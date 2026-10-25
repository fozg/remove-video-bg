"""Lõi xử lý: tách nền video bằng BiRefNet (chất lượng cao) hoặc RobustVideoMatting (nhanh, người)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ENGINES = {
    "birefnet": "Chất lượng cao – mọi vật thể, giữ cả mũ/đồ cầm tay (BiRefNet, ~2–3 khung/giây)",
    "rvm": "Nhanh – chỉ người (RobustVideoMatting, ~18 khung/giây)",
}

OUTPUTS = {
    "mov": "Nền trong suốt – ProRes 4444 .mov (Premiere, Final Cut, CapCut)",
    "webm": "Nền trong suốt – VP9 .webm (web, OBS)",
    "green": "Phông xanh – .mp4",
    "color": "Màu nền tuỳ chọn – .mp4",
    "image": "Ảnh nền tuỳ chọn – .mp4",
    "matte": "Mặt nạ đen trắng (alpha matte) – .mp4",
}

ProgressFn = Callable[[int, int], None]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_color(value: str | None) -> tuple[int, int, int]:
    """Nhận '#00ff00', '#0f0' hoặc 'rgba(0, 255, 0, 1)' -> (r, g, b)."""
    if not value:
        return (0, 255, 0)
    value = value.strip()
    if value.startswith("rgb"):
        nums = [float(n) for n in re.findall(r"[\d.]+", value)[:3]]
        return tuple(int(round(n)) for n in nums)  # type: ignore[return-value]
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}  # HLG (iPhone), PQ

# Chuyển HDR -> SDR; nếu không, model nhận ảnh xám nhạt và tách sai (mất đầu, mất mũ...)
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=pc")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    hdr: bool


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames,color_transfer:stream_side_data=rotation",
         path],
        capture_output=True, text=True,
    )
    streams = json.loads(out.stdout or "{}").get("streams")
    if out.returncode != 0 or not streams:
        raise ValueError(f"Không mở được video: {path}")
    st = streams[0]
    w, h = st["width"], st["height"]
    rotation = next((int(d["rotation"]) for d in st.get("side_data_list", []) if "rotation" in d), 0)
    if abs(rotation) % 180 == 90:  # video dọc quay trên điện thoại
        w, h = h, w
    num, den = (st.get("avg_frame_rate") or "30/1").split("/")
    return VideoInfo(
        width=w,
        height=h,
        fps=float(num) / float(den) if float(den) else 30.0,
        frames=int(st.get("nb_frames") or 0),
        hdr=st.get("color_transfer") in HDR_TRANSFERS,
    )


def read_frames(path: str, info: VideoInfo) -> Iterator[np.ndarray]:
    """Trả về từng khung hình RGB uint8 (ffmpeg tự xoay theo metadata)."""
    vf = [TONEMAP] if info.hdr else []
    cmd = ["ffmpeg", "-loglevel", "error", "-i", path]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = info.width * info.height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------- Matting engines


class RVMEngine:
    """RobustVideoMatting: có bộ nhớ hồi quy nên viền ổn định giữa các khung hình.
    Chỉ nhận diện người; vật đội/đeo lớn (mũ rộng vành...) có thể bị coi là nền."""

    _model = None

    def __init__(self, variant: str = "resnet50"):
        self.device = pick_device()
        if RVMEngine._model is None:
            model = torch.hub.load("PeterL1n/RobustVideoMatting", variant, trust_repo=True)
            RVMEngine._model = model.eval().to(self.device)
        self.model = RVMEngine._model
        self.rec = [None] * 4

    @staticmethod
    def downsample_ratio(h: int, w: int) -> float:
        return min(512 / max(h, w), 1.0)

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        src = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].float() / 255
        fgr, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio(h, w))
        fgr = (fgr[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        pha = pha[0, 0].clamp(0, 1).cpu().numpy()
        return fgr, pha


class BiRefNetEngine:
    """BiRefNet: tách vật thể nổi bật từng khung, hiểu cả mũ, áo choàng, đồ cầm tay."""

    _model = None
    SIZE = 1024
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, repo: str = "ZhengPeng7/BiRefNet_lite"):
        from transformers import AutoModelForImageSegmentation

        self.device = pick_device()
        self.dtype = torch.float16 if self.device.type in ("mps", "cuda") else torch.float32
        if BiRefNetEngine._model is None:
            model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
            BiRefNetEngine._model = model.eval().to(self.device, self.dtype)
        self.model = BiRefNetEngine._model
        self.mean = torch.tensor(self.MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        x = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = F.interpolate(x, (self.SIZE, self.SIZE), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        pha = self.model(x)[-1].sigmoid().float()
        pha = F.interpolate(pha, (h, w), mode="bilinear", align_corners=False)
        return frame, pha[0, 0].clamp(0, 1).cpu().numpy()


def make_engine(name: str):
    if name == "rvm":
        return RVMEngine()
    if name == "birefnet":
        return BiRefNetEngine()
    raise ValueError(f"Engine không hợp lệ: {name}")


# ---------------------------------------------------------------- Output


def output_suffix(mode: str) -> str:
    return {"mov": ".mov", "webm": ".webm"}.get(mode, ".mp4")


def ffmpeg_cmd(src: str, dst: str, info: VideoInfo, mode: str, keep_audio: bool) -> list[str]:
    alpha = mode in ("mov", "webm")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24",
        "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}", "-i", "-",
    ]
    if keep_audio:
        cmd += ["-i", src, "-map", "0:v", "-map", "1:a?", "-shortest"]
    if mode == "mov":
        cmd += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                "-c:a", "pcm_s16le"]
    elif mode == "webm":
        cmd += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30",
                "-row-mt", "1", "-c:a", "libopus"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k"]
    return cmd + [dst]


def build_background(mode: str, info: VideoInfo, color: str | None, image: str | None) -> np.ndarray | None:
    if mode == "green":
        rgb = (0, 177, 64)
    elif mode == "color":
        rgb = parse_color(color)
    elif mode == "image":
        if not image:
            raise ValueError("Chưa chọn ảnh nền.")
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        # Phủ kín khung hình (cover), cắt phần thừa ở giữa
        scale = max(info.width / img.shape[1], info.height / img.shape[0])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y = (img.shape[0] - info.height) // 2
        x = (img.shape[1] - info.width) // 2
        return img[y : y + info.height, x : x + info.width].astype(np.float32)
    else:
        return None
    return np.full((info.height, info.width, 3), rgb, dtype=np.float32)


def render(fgr: np.ndarray, pha: np.ndarray, mode: str, bg: np.ndarray | None) -> np.ndarray:
    if mode in ("mov", "webm"):
        a = (pha * 255).astype(np.uint8)[..., None]
        return np.concatenate([fgr, a], axis=2)
    if mode == "matte":
        return np.repeat((pha * 255).astype(np.uint8)[..., None], 3, axis=2)
    a = pha[..., None]
    return (fgr.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)


def remove_background(
    src: str,
    dst: str,
    engine: str = "birefnet",
    mode: str = "mov",
    color: str | None = None,
    image: str | None = None,
    keep_audio: bool = True,
    progress: ProgressFn | None = None,
) -> str:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Chưa cài ffmpeg (brew install ffmpeg).")
    if mode not in OUTPUTS:
        raise ValueError(f"Kiểu xuất không hợp lệ: {mode}")

    info = probe(src)
    bg = build_background(mode, info, color, image)
    matte = make_engine(engine)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(ffmpeg_cmd(src, dst, info, mode, keep_audio),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    done = 0
    try:
        for frame in read_frames(src, info):
            fgr, pha = matte(frame)
            proc.stdin.write(render(fgr, pha, mode, bg).tobytes())
            done += 1
            if progress:
                progress(done, info.frames)
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg lỗi:\n{err}")
    return dst

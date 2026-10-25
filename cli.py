"""Dòng lệnh: python cli.py input.mp4 -o out.mov --engine birefnet --mode mov"""

import argparse
from pathlib import Path

from tqdm import tqdm

from bgremove import ENGINES, OUTPUTS, output_suffix, remove_background


def main():
    p = argparse.ArgumentParser(description="Xoá background video")
    p.add_argument("input", help="Video đầu vào")
    p.add_argument("-o", "--output", help="File đầu ra (mặc định: <tên>_nobg.<ext>)")
    p.add_argument("--engine", choices=ENGINES, default="birefnet", help="birefnet = chất lượng cao, rvm = nhanh (chỉ người)")
    p.add_argument("--mode", choices=OUTPUTS, default="mov", help="Kiểu xuất")
    p.add_argument("--color", default="#00ff00", help="Màu nền khi --mode color")
    p.add_argument("--bg-image", help="Ảnh nền khi --mode image")
    p.add_argument("--no-audio", action="store_true", help="Bỏ âm thanh")
    a = p.parse_args()

    src = Path(a.input)
    dst = a.output or str(src.with_name(f"{src.stem}_nobg{output_suffix(a.mode)}"))

    bar = tqdm(unit="frame")

    def progress(done, total):
        bar.total = total or None
        bar.n = done
        bar.refresh()

    remove_background(str(src), dst, a.engine, a.mode, a.color, a.bg_image,
                      keep_audio=not a.no_audio, progress=progress)
    bar.close()
    print(f"Xong: {dst}")


if __name__ == "__main__":
    main()

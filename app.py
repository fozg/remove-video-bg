"""Giao diện web: python app.py rồi mở http://127.0.0.1:7860"""

import tempfile
import time
from pathlib import Path

import gradio as gr

from bgremove import ENGINES, OUTPUTS, output_suffix, remove_background

OUT_DIR = Path(__file__).parent / "outputs"


def process(video, engine, mode, color, bg_image, keep_audio, progress=gr.Progress()):
    if not video:
        raise gr.Error("Hãy tải lên một video.")
    if mode == "image" and not bg_image:
        raise gr.Error("Hãy chọn ảnh nền.")

    stem = Path(video).stem
    dst = OUT_DIR / f"{stem}_nobg_{int(time.time())}{output_suffix(mode)}"
    progress(0, desc="Đang tải model...")

    def on_progress(done, total):
        progress((done, total or None), desc="Đang tách nền", unit="khung")

    try:
        remove_background(video, str(dst), engine, mode, color, bg_image, keep_audio, on_progress)
    except Exception as e:
        raise gr.Error(str(e))

    # .mov ProRes / .webm alpha không xem trước được trong mọi trình duyệt,
    # nên luôn trả file để tải; chỉ preview với mp4/webm.
    preview = str(dst) if dst.suffix in (".mp4", ".webm") else None
    return preview, str(dst)


# Tự xoá file upload/tạm của Gradio sau 1 giờ để không đầy ổ cứng
with gr.Blocks(title="Xoá nền video", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# 🎬 Xoá background video\nChạy hoàn toàn trên máy bạn, không upload lên đâu cả.")
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video gốc", sources=["upload"])
            engine = gr.Radio([(v, k) for k, v in ENGINES.items()], value="birefnet", label="Chế độ")
            mode = gr.Radio([(v, k) for k, v in OUTPUTS.items()], value="mov", label="Kiểu xuất")
            color = gr.ColorPicker(value="#00ff00", label="Màu nền", visible=False)
            bg_image = gr.Image(type="filepath", label="Ảnh nền", visible=False)
            keep_audio = gr.Checkbox(value=True, label="Giữ âm thanh")
            run = gr.Button("Xoá nền", variant="primary")
        with gr.Column():
            preview = gr.Video(label="Xem trước", interactive=False)
            file = gr.File(label="Tải về")

    mode.change(
        lambda m: (gr.update(visible=m == "color"), gr.update(visible=m == "image")),
        mode, [color, bg_image],
    )
    run.click(process, [video, engine, mode, color, bg_image, keep_audio], [preview, file])


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    demo.queue().launch(inbrowser=True, allowed_paths=[str(OUT_DIR), tempfile.gettempdir()])

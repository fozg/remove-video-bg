# Xoá background video

Chạy offline trên máy (Apple Silicon dùng GPU qua MPS/CoreML).

## Chạy giao diện web

```bash
./run.sh
```

Trình duyệt sẽ mở http://127.0.0.1:7860. File kết quả lưu trong `outputs/`.

## Dòng lệnh

```bash
.venv/bin/python cli.py video.mp4                          # -> video_nobg.mov (trong suốt)
.venv/bin/python cli.py video.mp4 --mode green             # phông xanh
.venv/bin/python cli.py video.mp4 --mode color --color "#ffffff"
.venv/bin/python cli.py video.mp4 --mode image --bg-image nen.jpg
.venv/bin/python cli.py video.mp4 --engine rvm             # nhanh hơn, chỉ người
```

## Engine

| Engine     | Dùng cho                                   | Tốc độ (M4, 1080p) |
|------------|--------------------------------------------|--------------------|
| `birefnet` | Mặc định. Mọi vật thể, giữ cả mũ, áo choàng, đồ cầm tay | ~2–3 khung/giây |
| `rvm`      | Chỉ người, nhanh, viền rất ổn định         | ~18 khung/giây     |

`rvm` có thể coi mũ rộng vành, đồ vật to là nền. Gặp trường hợp đó thì dùng `birefnet`.

Video HDR của iPhone được tự chuyển sang SDR trước khi tách nền.

## Kiểu xuất

- `mov`: ProRes 4444 có alpha, dùng cho Premiere, Final Cut, DaVinci, CapCut desktop. File rất nặng (~60 MB/giây video 1080p)
- `webm`: VP9 có alpha, dùng cho web và OBS
- `green`, `color`, `image`: MP4 đã ghép nền mới
- `matte`: video mặt nạ đen trắng

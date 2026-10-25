#!/bin/bash
# Khởi động giao diện web xoá nền video
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python app.py

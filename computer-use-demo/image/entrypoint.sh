#!/bin/bash
set -e

./start_all.sh
./novnc_startup.sh

python -m uvicorn computer_use_demo.fastapi:app --port 8501 --host 0.0.0.0
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="weights"
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
MODEL_PATH="${MODEL_DIR}/yolov8n.onnx"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
else
    echo "Downloading YOLOv8n ONNX model..."
    curl -L -o "$MODEL_PATH" "$MODEL_URL"
    echo "Downloaded to $MODEL_PATH"
fi

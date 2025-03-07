#!/bin/bash

# Detect architecture and GPU
if [ "$(uname -m)" = "arm64" ]; then
    echo "Detected Apple Silicon (ARM64)"
    export ARCH=arm
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "Detected NVIDIA GPU"
    export ARCH=gpu
else
    echo "No GPU detected, using CPU"
    export ARCH=cpu
fi

# Make the script executable
chmod +x "$0"

# Run docker-compose
docker-compose up --build 
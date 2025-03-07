#!/bin/bash

echo "Checking environment for AI Lip Sync App..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
else
    echo "✅ Docker is installed"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
else
    echo "✅ Docker Compose is installed"
fi

# Check available memory
total_mem=$(free -g | awk '/^Mem:/{print $2}')
if [ "$total_mem" -lt 8 ]; then
    echo "⚠️ Warning: Less than 8GB RAM available ($total_mem GB)"
else
    echo "✅ Sufficient memory available ($total_mem GB)"
fi

# Check GPU availability
if [ "$(uname -m)" = "arm64" ]; then
    echo "✅ Apple Silicon detected - MPS acceleration available"
elif command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
else
    echo "ℹ️ No GPU detected - will use CPU"
fi

echo "Environment check complete!" 
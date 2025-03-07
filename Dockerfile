# Use Python 3.12.7 as base image
FROM python:3.12.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies based on architecture and GPU availability
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        # ARM64 (Apple Silicon) - use MPS backend \
        pip install --no-cache-dir -r requirements.txt && \
        echo "Installing for Apple Silicon (ARM64)" ; \
    elif command -v nvidia-smi >/dev/null 2>&1; then \
        # NVIDIA GPU available - install CUDA version \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
        pip install --no-cache-dir -r requirements.txt && \
        echo "Installing with CUDA support" ; \
    else \
        # CPU only \
        pip install --no-cache-dir -r requirements.txt && \
        echo "Installing CPU-only version" ; \
    fi

# Copy the rest of the application
COPY . .

# Download Wav2Lip model
RUN mkdir -p wav2lip_checkpoints && \
    pip install gdown && \
    gdown --folder https://drive.google.com/drive/folders/1Sy5SHRmI3zgg2RJaOttNsN3iJS9VVkbg?usp=sharing -O wav2lip_checkpoints

# Download avatar videos
RUN mkdir -p data/avatars/samples && \
    gdown --folder https://drive.google.com/drive/folders/1h9pkU5wenrS2vmKqXBfFmrg-1hYw5s4q?usp=sharing -O data/avatars/samples

# Create directories for uploads and results
RUN mkdir -p uploads wav2lip/results

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Set PyTorch environment variables based on architecture
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> /etc/profile.d/pytorch.sh ; \
    elif command -v nvidia-smi >/dev/null 2>&1; then \
        echo "export CUDA_VISIBLE_DEVICES=0" >> /etc/profile.d/pytorch.sh ; \
    else \
        echo "export PYTORCH_DEVICE=cpu" >> /etc/profile.d/pytorch.sh ; \
    fi

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 
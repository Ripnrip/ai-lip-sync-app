# AI Lip Sync App

A powerful application that uses AI to synchronize lip movements with audio input, creating realistic talking head videos. Built with Wav2Lip and optimized for multiple platforms.

## Quick Start 🚀

### Easiest: Automated Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-lip-sync-app.git
cd ai-lip-sync-app

# One-step setup (creates .venv, installs dependencies)
./setup_venv.sh

# Run the app
source .venv/bin/activate
streamlit run app.py
```

### Or use Makefile (if you have make):
```bash
make setup
make run
```

Then open http://localhost:8501 in your browser.

### Prerequisites

- Python 3.12+
- pip
- 8GB RAM minimum (16GB recommended)
- For GPU acceleration:
  - Apple Silicon Mac: No additional setup needed
  - NVIDIA GPU: Not supported natively, use Docker if needed

### Environment Configuration

You can customize the app's behavior by copying `.env.example` to `.env` and adjusting the values:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

Available configurations:
- Memory limits for Docker (if using Docker)
- Custom port for the web interface
- Force specific architecture (CPU/GPU/ARM)

## Features

- High-quality full-face animation for realistic results
- Support for both image and video inputs
- Live microphone recording or audio file upload
- Built-in avatar selection or custom image/video upload
- Automatic video/audio duration matching
- Progress tracking with detailed status updates
- Video trimming and post-processing options
- Hardware acceleration support (Apple Silicon, NVIDIA GPU, CPU)

## Running the App

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-lip-sync-app
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

### Docker Installation

The app includes a Docker configuration that automatically detects and utilizes the best available hardware:

1. Run with the automatic configuration script:
```bash
./run.sh
```

This will automatically:
- Detect Apple Silicon (M1/M2) and use MPS acceleration
- Detect NVIDIA GPUs and use CUDA acceleration
- Fall back to CPU optimization if no GPU is available

Or manually specify the configuration:
```bash
# For Apple Silicon
ARCH=arm docker-compose up --build

# For NVIDIA GPU
ARCH=gpu docker-compose up --build

# For CPU only
ARCH=cpu docker-compose up --build
```

The app will be available at http://localhost:8501

### Docker Configuration Files

The project uses a modular Docker configuration:
- `docker-compose.yml` - Base configuration
- `docker-compose.arm.yml` - Apple Silicon specific settings
- `docker-compose.gpu.yml` - NVIDIA GPU specific settings
- `docker-compose.cpu.yml` - CPU fallback settings

## Hardware Acceleration

The app automatically detects and uses the best available hardware:

- **Apple Silicon (M1/M2)**: Uses Metal Performance Shaders (MPS) for GPU acceleration
- **NVIDIA GPU**: Uses CUDA for GPU acceleration
- **CPU Only**: Optimized for CPU performance

## Usage

### Web Interface

1. Choose your avatar:
   - Select from built-in avatars
   - Upload your own image or video

2. Provide audio:
   - Record directly using your microphone
   - Upload an audio file (WAV or MP3)

3. Click "Start Animation" to begin processing
   - Processing time is approximately 30 minutes for a 30-second video
   - Progress bar shows current status and estimated time remaining

4. Post-processing options:
   - Trim the generated video
   - Download the result
   - Adjust video/audio synchronization

### Command Line Interface

The CLI provides direct access to the lip sync functionality without the web interface:

```bash
python cli.py --face <input_video/image> --audio <input_audio> [options]
```

Required arguments:
- `--face`: Path to video/image file for lip sync
- `--audio`: Path to audio file (WAV or MP3)

Optional arguments:
- `--output`: Path to save the output video (default: output.mp4)
- `--device`: Choose processing device [auto|cuda|mps|cpu] (default: auto)
- `--resize`: Resize factor for faster processing (default: 1.0)
- `--fps`: Force output video FPS
- `--trim-start`: Trim start time in seconds
- `--trim-end`: Trim end time in seconds

Examples:

```bash
# Basic usage
python cli.py --face input.mp4 --audio speech.wav

# Specify output file and device
python cli.py --face input.mp4 --audio speech.wav --output result.mp4 --device cuda

# Process faster with lower resolution
python cli.py --face input.mp4 --audio speech.wav --resize 0.5

# Trim output video
python cli.py --face input.mp4 --audio speech.wav --trim-start 1.5 --trim-end 10.0
```

## Best Practices

### Video Input
- Use clear, well-lit frontal views of faces
- Ensure faces take up a reasonable portion of the frame
- Avoid extreme angles or partial face views
- Recommended minimum resolution: 480p
- Stable videos with minimal camera movement work best

### Audio Input
- Clear speech with minimal background noise
- Consistent audio volume
- Supported formats: WAV, MP3
- Audio length doesn't need to match video length (will be automatically adjusted)

## Development

The project uses Docker for consistent development environments. Key files:

- `Dockerfile`: Multi-architecture support with automatic hardware detection
- `docker-compose.yml`: Container orchestration with resource management
- `run.sh`: Automatic platform detection and configuration script

## Troubleshooting

1. **Memory Issues**: Adjust memory limits in docker-compose.yml
2. **GPU Not Detected**: Ensure proper drivers are installed
3. **Performance Issues**: Check hardware acceleration settings
4. **Video Quality**: Follow the best practices for input videos

## License

[Your License Here]

## Acknowledgments

- Wav2Lip for the core lip-sync technology
- Streamlit for the web interface
- PyTorch for deep learning capabilities

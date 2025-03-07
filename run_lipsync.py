#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import cv2
import subprocess
import numpy as np
import tempfile
from wav2lip import inference
from wav2lip import audio

# Monkey patch sys.argv to prevent inference.py from processing command line arguments
original_argv = sys.argv
sys.argv = [sys.argv[0]]

# GPU/Device selection logic
# Check CUDA (NVIDIA) first, then MPS (Apple Silicon), then fall back to CPU
if torch.cuda.is_available():
    device = 'cuda'
    # Get CUDA device info
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "Unknown"
    print(f"Using {device} for inference. Device: {cuda_device_name}")
    # Free CUDA memory
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = 'mps'
    # Enable memory optimization for Apple Silicon
    torch.mps.empty_cache()
    print(f"Using {device} (Apple Silicon) for inference.")
else:
    device = 'mps'
    print(f"Using {device} for inference. No GPU detected or available.")

# Make device accessible to the wav2lip module
inference.device = device

def get_video_duration(video_path):
    """Get the duration of a video file in seconds"""
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        
        duration = frame_count / fps if fps > 0 else 0
        return duration
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        return 0

def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds"""
    try:
        # Use ffprobe to get audio duration
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{audio_path}"'
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        return float(output)
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return 0

def trim_video(input_path, output_path, start_time, end_time):
    """
    Trim a video using ffmpeg from start_time to end_time.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the trimmed video
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Input video not found: {input_path}")
            return False
            
        # Format the command - use -ss before -i for faster seeking
        command = f'ffmpeg -y -ss {start_time} -i "{input_path}" -to {end_time-start_time} -c:v copy -c:a copy "{output_path}"'
        
        # Use subprocess.run for better error handling
        result = subprocess.run(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"FFMPEG error: {result.stderr}")
            return False
            
        # Verify the output file exists and has a size greater than 0
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            print("Output file was not created correctly")
            return False
            
    except Exception as e:
        print(f"Error trimming video: {str(e)}")
        return False

def trim_audio(input_path, output_path, start_time, end_time):
    """
    Trim an audio file using ffmpeg.
    
    Args:
        input_path: Path to the input audio
        output_path: Path to save the trimmed audio
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Format the command
        command = f'ffmpeg -y -ss {start_time} -i "{input_path}" -to {end_time-start_time} -c copy "{output_path}"'
        
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"FFMPEG error: {result.stderr}")
            return False
            
        # Verify the output file exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            print("Output audio file was not created correctly")
            return False
            
    except Exception as e:
        print(f"Error trimming audio: {str(e)}")
        return False

def configure_inference_args():
    """Configure the arguments for the inference module"""
    # Create a new args object with default values
    class Args:
        pass
    
    args = Args()
    
    # Set default values for inference
    args.outfile = 'wav2lip/results/result_voice_3.mp4'
    args.static = False
    args.fps = 25.0
    args.pads = [0, 10, 0, 0]
    args.face_det_batch_size = 32
    args.wav2lip_batch_size = 512
    args.resize_factor = 1
    args.crop = [0, -1, 0, -1]
    args.box = [-1, -1, -1, -1]
    args.rotate = False
    args.nosmooth = False
    args.img_size = 96
    
    # Replace the global args in inference module
    inference.args = args
    
    return args

def main():
    # Restore original command line arguments
    sys.argv = original_argv
    
    print("Arguments received:", sys.argv)
    parser = argparse.ArgumentParser(description='Lip-sync a face in a video or image to match audio')
    
    # Required arguments
    parser.add_argument('--face', required=True, type=str,
                        help='Path to video/image that contains faces to use')
    parser.add_argument('--audio', required=True, type=str,
                        help='Path to audio file to use as speech source')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, 
                        default='wav2lip_checkpoints/wav2lip_gan.pth',
                        help='Path to the Wav2Lip model checkpoint')
    parser.add_argument('--outfile', type=str, 
                        default='wav2lip/results/result_voice.mp4',
                        help='Path to save the output video')
    parser.add_argument('--resize_factor', type=int, default=1,
                        help='Resize the input video by this factor')
    parser.add_argument('--fps', type=float, default=None,
                        help='FPS of the output video')
    parser.add_argument('--max_frames', type=int, default=1000,
                        help='Maximum number of frames to process')
    parser.add_argument('--slow_mode', action='store_true',
                        help='Use slow mode for better quality but slower processing')
    parser.add_argument('--nosmooth', action='store_true',
                        help='Disable face detection smoothing')
    parser.add_argument('--disable_partial_save', action='store_true',
                        help='Disable saving of partial results during processing')
    
    # Video trimming options
    parser.add_argument('--trim', action='store_true',
                        help='Trim the output video')
    parser.add_argument('--trim_start', type=float, default=0.0,
                        help='Start time for trimming in seconds')
    parser.add_argument('--trim_end', type=float, default=None,
                        help='End time for trimming in seconds')
    parser.add_argument('--trim_output', type=str, default=None,
                        help='Output path for trimmed video')
    
    print("Parser set up, parsing arguments...")
    args = parser.parse_args()
    print("Arguments parsed:", args)
    
    # Check if the face and audio file exist
    if not os.path.exists(args.face):
        print(f"Error: Face video/image file '{args.face}' does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' does not exist")
        sys.exit(1)
    
    # Create required directories
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    os.makedirs('wav2lip/temp', exist_ok=True)
    
    # Run inference
    print(f"Running lip sync with face: {args.face} and audio: {args.audio}")
    print(f"Max frames: {args.max_frames}, Resize factor: {args.resize_factor}")
    
    try:
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for processing.")
        else:
            print("CUDA is not available. Using CPU for processing (this may be slow).")
    except ImportError:
        print("PyTorch not found or CUDA status could not be determined.")
    
    # Load the model
    model = inference._load(args.checkpoint)
    
    # Run inference
    output_path = inference.main(args.face, args.audio, model, slow_mode=args.slow_mode)
    
    # Trim video if requested
    if args.trim and output_path and os.path.exists(output_path):
        trim_output = args.trim_output or output_path.replace('.mp4', '_trimmed.mp4')
        trim_start = args.trim_start
        trim_end = args.trim_end or ''  # If None, don't specify end time
        
        trim_cmd = f"ffmpeg -y -i {output_path} -ss {trim_start}"
        if trim_end:
            trim_cmd += f" -to {trim_end}"
        trim_cmd += f" -c copy {trim_output}"
        
        print(f"Trimming video from {trim_start}s to {trim_end or 'end'}")
        subprocess.call(trim_cmd, shell=True)
        print(f"Trimmed video saved to {trim_output}")
    
    print(f"Output saved to: {output_path}")
    return 0
    
if __name__ == "__main__":
    sys.exit(main()) 
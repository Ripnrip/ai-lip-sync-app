#!/usr/bin/env python3

import argparse
import os
import torch
from wav2lip import inference
from wav2lip.models import Wav2Lip
import gdown

def download_model_if_needed(model_path="wav2lip_checkpoints/wav2lip_gan.pth"):
    """Download the model if it doesn't exist"""
    if not os.path.exists(model_path):
        print("Downloading Wav2Lip model...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://drive.google.com/drive/folders/1Sy5SHRmI3zgg2RJaOttNsN3iJS9VVkbg?usp=sharing"
        gdown.download_folder(url, quiet=False, use_cookies=False)
        if not os.path.exists(model_path):
            raise RuntimeError("Failed to download model")

def main():
    parser = argparse.ArgumentParser(description='AI Lip Sync CLI')
    parser.add_argument('--face', required=True, help='Path to video/image file for lip sync')
    parser.add_argument('--audio', required=True, help='Path to audio file (wav or mp3)')
    parser.add_argument('--output', default='output.mp4', help='Path to save the output video')
    parser.add_argument('--model', default='wav2lip_checkpoints/wav2lip_gan.pth', help='Path to model checkpoint')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto', help='Device to use')
    parser.add_argument('--resize', type=float, default=1, help='Resize factor for faster processing')
    parser.add_argument('--fps', type=float, help='Force output video FPS')
    parser.add_argument('--trim-start', type=float, help='Trim start time in seconds')
    parser.add_argument('--trim-end', type=float, help='Trim end time in seconds')
    
    args = parser.parse_args()

    # Download model if needed
    download_model_if_needed(args.model)

    # Determine device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("Using NVIDIA GPU (CUDA)")
        else:
            device = 'cpu'
            print("Using CPU")
    else:
        device = args.device
        print(f"Using specified device: {device}")

    # Set environment variables
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load model
    print("Loading model...")
    model = Wav2Lip()
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Process the video
    print("Processing... This may take a while.")
    inference.main(
        args.face,
        args.audio,
        model,
        device=device,
        resize_factor=args.resize,
        fps=args.fps,
        output_path=args.output,
        trim_start=args.trim_start,
        trim_end=args.trim_end
    )
    print(f"Done! Output saved to: {args.output}")

if __name__ == "__main__":
    main() 
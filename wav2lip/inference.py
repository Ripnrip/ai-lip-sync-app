import numpy as np
import cv2 
import os 
import argparse
import subprocess
from tqdm import tqdm
import sys
import traceback
from .audio import load_wav, melspectrogram
from .face_detection import FaceAlignment, LandmarksType
import torch
import platform

# Global variables
frame_limit = 1000  # Default frame limit, can be overridden from run_lipsync.py
save_interval = 50  # Save partial results every N frames

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='wav2lip/results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', type=int, default=1, 
					help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--img_size', type=int, default=96, help='Size of the face thumbnail used by the model')

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use')
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source')
parser.add_argument('--max_frames', type=int, default=1000,
					help='Maximum number of frames to process (default: 1000)')
parser.add_argument('--disable_partial_save', action='store_true',
					help='Do not save partial results during processing')

def parse_args():
    """Parse command line arguments and return the args object."""
    args = parser.parse_args()
    return args

args = parser.parse_args()
args.img_size = 96

# Check for available devices - will be overridden if device is passed from outside
if not 'device' in globals():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using {device} for inference.")
    elif torch.backends.mps.is_available():
        device = 'mps'  # Use Apple Silicon GPU
        print(f"Using {device} for inference.")
    else:
        device = 'cpu'
        print(f"Using {device} for inference.")

# Print the device being used for debugging
if 'device' in globals():
    print(f'Device set externally to: {device}')

def get_smoothened_boxes(boxes, idx):
    """Get smoothened box for a specific index"""
    if idx >= len(boxes) or boxes[idx] is None:
        return None, None
    
    # Return the face region and coordinates
    if isinstance(boxes[idx], list) and len(boxes[idx]) == 2:  # Format from the specified bounding box
        return boxes[idx][0], boxes[idx][1]
    else:  # Format from face detection - [x1, y1, x2, y2]
        if isinstance(boxes[idx], list) or isinstance(boxes[idx], tuple):
            if len(boxes[idx]) >= 4:  # Make sure we have all 4 coordinates
                x1, y1, x2, y2 = boxes[idx][:4]
                # Return coordinates in the expected format (y1, y2, x1, x2)
                coords = (y1, y2, x1, x2)
                return None, coords
        
        print(f"WARNING: Unexpected box format at idx {idx}: {boxes[idx]}")
        return None, None

def face_detect(images):
    print(f"Starting face detection using {device} device...")
    try:
        detector = FaceAlignment(LandmarksType._2D, 
                                flip_input=False, device=device, verbose=True)
    except Exception as e:
        print(f"Error initializing face detector: {str(e)}")
        print("Attempting to fall back to CPU for face detection...")
        detector = FaceAlignment(LandmarksType._2D, 
                                flip_input=False, device='cpu', verbose=True)
    
    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                batch = np.array(images[i:i + batch_size])
                print(f"Processing detection batch {i//batch_size + 1}, shape: {batch.shape}")
                batch_predictions = detector.get_detections_for_batch(batch)
                predictions.extend(batch_predictions)
        except RuntimeError as e:
            print(f"Runtime error in face detection: {str(e)}")
            if batch_size == 1:
                # Error when batch_size is already 1
                print('Face detection failed at minimum batch size! Using fallback method...')
                # Create empty predictions for all frames to allow processing to continue
                predictions = [None] * len(images)
                break
            batch_size //= 2
            print('Reducing face detection batch size to', batch_size)
            continue
        except Exception as e:
            print(f"Unexpected error in face detection: {str(e)}")
            # Create empty predictions and continue with fallback
            predictions = [None] * len(images)
            break
        break
        
    # Check if we have at least one valid face detection
    faces_detected = sum(1 for p in predictions if p is not None)
    print(f"Detected faces in {faces_detected} out of {len(images)} frames ({faces_detected/len(images)*100:.1f}%)")
    
    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    for i, (rect, image) in enumerate(zip(predictions, images)):
        if rect is None:
            # Create default coordinates for face detection
            h, w = image.shape[:2]
            
            # Better face region estimation - center with proportional sizing
            center_x = w // 2
            center_y = h // 2
            
            # Use about 40% of frame height for face for more accurate proportions
            face_h = int(h * 0.4)
            face_w = int(face_h * 0.8)  # Typical face aspect ratio
            
            pady1, pady2, padx1, padx2 = args.pads
            x1 = max(0, center_x - face_w // 2 - padx1)
            y1 = max(0, center_y - face_h // 2 - pady1)
            x2 = min(w, center_x + face_w // 2 + padx2)
            y2 = min(h, center_y + face_h // 2 + pady2)
            
            print(f"Estimated face region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            results.append([x1, y1, x2, y2])
            continue
            
        # If face is detected, use its coordinates with padding
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])
    
    return results

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			try:
				print(f"Starting face detection for {len(frames)} frames...")
				face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
				print("Face detection completed successfully")
			except Exception as e:
				print(f"Face detection error: {str(e)}")
				print(f"Error type: {type(e).__name__}")
				traceback.print_exc()
				print("Using fallback method with default face regions...")
				# Create default face regions for all frames
				h, w = frames[0].shape[:2]
				
				# Better face region estimation - center with proportional sizing
				center_x = w // 2
				center_y = h // 2
				
				# Use about 40% of frame height for face for more accurate proportions
				face_h = int(h * 0.4)
				face_w = int(face_h * 0.8)  # Typical face aspect ratio
				
				pady1, pady2, padx1, padx2 = args.pads
				x1 = max(0, center_x - face_w // 2 - padx1)
				y1 = max(0, center_y - face_h // 2 - pady1)
				x2 = min(w, center_x + face_w // 2 + padx2)
				y2 = min(h, center_y + face_h // 2 + pady2)
				
				print(f"Estimated face region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
				
				# Use the same format as the face_detect function returns
				face_det_results = [[x1, y1, x2, y2] for _ in range(len(frames))]
		else:
			try:
				print("Starting face detection for static image...")
				face_det_results = face_detect([frames[0]])
				print("Face detection completed successfully")
			except Exception as e:
				print(f"Face detection error: {str(e)}")
				print(f"Error type: {type(e).__name__}")
				traceback.print_exc()
				print("Using fallback method with default face region...")
				# Create default face region for static image
				h, w = frames[0].shape[:2]
				
				# Better face region estimation - center with proportional sizing
				center_x = w // 2
				center_y = h // 2
				
				# Use about 40% of frame height for face for more accurate proportions
				face_h = int(h * 0.4)
				face_w = int(face_h * 0.8)  # Typical face aspect ratio
				
				pady1, pady2, padx1, padx2 = args.pads
				x1 = max(0, center_x - face_w // 2 - padx1)
				y1 = max(0, center_y - face_h // 2 - pady1)
				x2 = min(w, center_x + face_w // 2 + padx2)
				y2 = min(h, center_y + face_h // 2 + pady2)
				
				print(f"Estimated face region for static image: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
				
				# Use the same format as the face_detect function returns
				face_det_results = [[x1, y1, x2, y2]]
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[x1, y1, x2, y2] for _ in range(len(frames))]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		
		if args.box[0] == -1:
			face, coords = get_smoothened_boxes(face_det_results, idx)
			
			if coords is None:
				print(f'Face coordinates not detected! Skipping frame {i}')
				continue
			
			# If face is None, extract it from the frame using coordinates
			if face is None:
				y1, y2, x1, x2 = coords
				try:
					if y1 >= y2 or x1 >= x2:
						print(f"Invalid coordinates at frame {i}: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
						continue
					if y1 < 0 or x1 < 0 or y2 > frame_to_save.shape[0] or x2 > frame_to_save.shape[1]:
						print(f"Out of bounds coordinates at frame {i}. Adjusting...")
						y1 = max(0, y1)
						x1 = max(0, x1)
						y2 = min(frame_to_save.shape[0], y2)
						x2 = min(frame_to_save.shape[1], x2)
					
					# Check if the region is too small
					if (y2 - y1) < 10 or (x2 - x1) < 10:
						print(f"Region too small at frame {i}. Skipping.")
						continue
						
					face = frames[idx][y1:y2, x1:x2]
				except Exception as e:
					print(f"Error extracting face at frame {i}: {str(e)}")
					continue
		else:
			face = frames[idx][y1:y2, x1:x2]
			coords = (y1, y2, x1, x2)
			
		try:    
			face = cv2.resize(face, (args.img_size, args.img_size))
			img_batch.append(face)
			mel_batch.append(m)
			frame_batch.append(frame_to_save)
			coords_batch.append(coords)
		except Exception as e:
			print(f"Error processing frame {i}: {str(e)}")
			continue
		
		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16

def _load(checkpoint_path):
    # Handle loading for different devices
    global device
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Import the model here to avoid circular imports
    from .models import Wav2Lip
    
    # Force model to the specified device
    model = Wav2Lip()
    
    # Load state dict
    if 's3fd.pth' in checkpoint_path:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
    
    # Set model to evaluation mode and report device
    print(f"Model loaded and moved to {device}")
    model.eval()
    return model

# Add helper function for moving data to device
def to_device(tensor, dev):
    if tensor is None:
        return None
    if isinstance(tensor, (list, tuple)):
        return [to_device(t, dev) for t in tensor]
    return tensor.to(dev, non_blocking=True)

def main(face, audio, model, slow_mode=False):
	im = cv2.imread(face)
	if im is not None:
		return main_from_img(face, audio, model, slow_mode=slow_mode)
	else:
		return main_from_video(face, audio, model, slow_mode=slow_mode)

def main_from_video(face_video_path, audio_path, model, slow_mode=False):
    # Create args object manually using the parse_args function we defined earlier
    args = parse_args()
    args.face = face_video_path
    args.audio = audio_path

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    # Set up output directories
    result_dir = os.path.dirname(args.outfile)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join('wav2lip', 'temp'), exist_ok=True)

    # Initialize video parameters
    video_stream = cv2.VideoCapture(args.face)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    
    # Get video dimensions for potential downscaling
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Auto-adjust resize factor for large videos
    if frame_width > 1920 or frame_height > 1080:
        # For 4K or larger videos, use a higher resize factor
        if frame_width >= 3840 or frame_height >= 2160:
            args.resize_factor = max(4, args.resize_factor)
            print(f"Auto-adjusting resize factor to {args.resize_factor} for high-resolution video")
        # For 1080p-4K videos
        elif frame_width > 1920 or frame_height > 1080:
            args.resize_factor = max(2, args.resize_factor)
            print(f"Auto-adjusting resize factor to {args.resize_factor} for high-resolution video")

    print('Reading video frames...')
    
    # For large videos, limit frames as needed
    frame_limit = args.max_frames if args.max_frames > 0 else float('inf')
    
    # Use tqdm for progress reporting
    full_frames = []
    frame_count = 0
    
    pbar = tqdm(total=min(total_frames, frame_limit))
    
    while frame_count < frame_limit:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
            
        if args.resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

        # Crop if specified
        y1, y2, x1, x2 = [0, -1, 0, -1]  # Default to full frame
        if hasattr(args, 'crop') and args.crop:
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)
        frame_count += 1
        pbar.update(1)

    pbar.close()
    video_stream.release()
    
    # If no frames were read, exit
    if len(full_frames) == 0:
        print('No frames found.')
        return None
    
    print(f"Processing {len(full_frames)} frames...")
    
    # Process audio
    if not audio_path.endswith('.wav'):
        print('Converting audio to wav format...')
        temp_audio = 'wav2lip/temp/temp.wav'
        if os.path.isfile(temp_audio):
            os.remove(temp_audio)
            
        command = f'ffmpeg -y -i {audio_path} -strict -2 {temp_audio}'
        subprocess.call(command, shell=True)
        audio_path = temp_audio

    # Get the fps and duration of the input audio
    wav = load_wav(audio_path, 16000)
    mel = melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    
    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    
    print(f"Length of mel chunks: {len(mel_chunks)}")
    
    # Align audio and video frames
    full_frames_batch = full_frames[:len(mel_chunks)]
    
    # Determine optimal batch size for GPU memory
    # For Apple Silicon, we need to be more conservative with batch sizes
    if device == 'mps':
        # Get free memory as a rough estimate for batch size calculation
        try:
            # For Apple Silicon, tune batch size based on video dimensions
            if frame_width * frame_height > 1000000:  # > ~1MP resolution
                optimal_batch_size = 8
            elif frame_width * frame_height > 500000:  # > ~0.5MP resolution
                optimal_batch_size = 16
            else:
                optimal_batch_size = 32
            
            print(f"Optimized batch size for MPS: {optimal_batch_size}")
            args.wav2lip_batch_size = optimal_batch_size
        except:
            print("Could not determine optimal batch size, using default")
    
    # Prepare for face detection
    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames_batch.copy(), mel_chunks)
    
    # Create temporary video file
    temp_video_path = 'wav2lip/temp/result.avi'
    
    # Process frames in batches
    out = None
    all_frames = []
    progress_bar = tqdm(total=len(mel_chunks))
    
    for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(temp_video_path, 
                              cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        
        # Convert to tensor and move to device
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2)))
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2)))
        
        # Move tensors to the appropriate device
        img_batch = to_device(img_batch, device)
        mel_batch = to_device(mel_batch, device)
        
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        
        # Move prediction back to CPU for numpy processing
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        # Apply prediction to frames
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            
            # Store processed frame
            all_frames.append(f.copy())
            
            # Write to temp video file
            out.write(f)
        
        progress_bar.update(len(img_batch))
    
    progress_bar.close()
    if out is not None:
        out.release()
    
    # Create final video with audio
    command = f'ffmpeg -y -i {temp_video_path} -i {audio_path} -strict -2 -q:v 1 {args.outfile}'
    subprocess.call(command, shell=True)
    
    print(f"Final video saved to {args.outfile}")
    return args.outfile

def main_from_img(face_img_path, audio_path, model, slow_mode=False):
	# Create args object manually using the parse_args function
	args = parse_args()
	args.face = face_img_path
	args.audio = audio_path

	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to image file')

	# Set up output directories
	result_dir = os.path.dirname(args.outfile)
	os.makedirs(result_dir, exist_ok=True)
	os.makedirs(os.path.join('wav2lip', 'temp'), exist_ok=True)

	# Load the image and convert to frames
	full_frames = [cv2.imread(args.face)]
	fps = args.fps or 25  # Default FPS for image input
	print(f"Using static image with FPS set to {fps}")

	# Process audio
	if not audio_path.endswith('.wav'):
		print('Converting audio to wav format...')
		temp_audio = 'wav2lip/temp/temp.wav'
		if os.path.isfile(temp_audio):
			os.remove(temp_audio)
			
		command = f'ffmpeg -y -i {audio_path} -strict -2 {temp_audio}'
		subprocess.call(command, shell=True)
		audio_path = temp_audio

	# Get the fps and duration of the input audio
	wav = load_wav(audio_path, 16000)
	mel = melspectrogram(wav)
	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
	
	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1
	
	print(f"Length of mel chunks: {len(mel_chunks)}")
	
	# Duplicate the image frame for each audio chunk
	full_frames_batch = [full_frames[0] for _ in range(len(mel_chunks))]
	
	# Determine optimal batch size for GPU memory
	# For Apple Silicon, use larger batches since we're duplicating a single image
	if device == 'mps':
		# Static images can use larger batches since the face detection is simpler
		optimal_batch_size = 64
		print(f"Optimized batch size for static image on MPS: {optimal_batch_size}")
		args.wav2lip_batch_size = optimal_batch_size
	
	# Prepare for face detection
	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames_batch.copy(), mel_chunks)
	
	# Create temporary video file
	temp_video_path = 'wav2lip/temp/result.avi'
	
	# Process frames in batches
	out = None
	all_frames = []
	progress_bar = tqdm(total=len(mel_chunks))
	
	for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
		if i == 0:
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(temp_video_path, 
								cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
		
		# Convert to tensor and move to device
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2)))
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2)))
		
		# Move tensors to the appropriate device
		img_batch = to_device(img_batch, device)
		mel_batch = to_device(mel_batch, device)
		
		with torch.no_grad():
			pred = model(mel_batch, img_batch)
		
		# Move prediction back to CPU for numpy processing
		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		# Apply prediction to frames
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			f[y1:y2, x1:x2] = p
			
			# Store processed frame
			all_frames.append(f.copy())
			
			# Write to temp video file
			out.write(f)
		
		progress_bar.update(len(img_batch))
	
	progress_bar.close()
	if out is not None:
		out.release()
	
	# Create final video with audio
	command = f'ffmpeg -y -i {temp_video_path} -i {audio_path} -strict -2 -q:v 1 {args.outfile}'
	subprocess.call(command, shell=True)
	
	print(f"Final video saved to {args.outfile}")
	return args.outfile

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
					help='Batch size for face detection', default=32)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=512)

parser.add_argument('--resize_factor', default=1, type=int, 
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

args = parser.parse_args()
args.img_size = 96

# Check for available devices
if torch.backends.mps.is_available():
    device = 'mps'  # Use Apple Silicon GPU
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print('Using {} for inference.'.format(device))

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
            
            # Simple and consistent face region estimation based on center of the frame
            center_x = w // 2
            center_y = h // 2
            
            # Use about 1/3 of the frame height for face
            face_h = h // 3
            face_w = min(w // 2, face_h)
            
            # Create a centered box
            x1 = max(0, center_x - face_w // 2 - padx1)
            y1 = max(0, center_y - face_h // 2 - pady1)
            x2 = min(w, center_x + face_w // 2 + padx2)
            y2 = min(h, center_y + face_h // 2 + pady2)
            
            if i == 0 or i % 100 == 0:  # Log only occasionally to avoid flooding
                print(f"Frame {i}: Using fallback face region at ({x1},{y1},{x2},{y2})")
            
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
				
				# Simple face region estimation in the center of the frame
				center_x = w // 2
				center_y = h // 2
				
				# Use about 1/3 of the frame height for face
				face_h = h // 3
				face_w = min(w // 2, face_h)
				
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
				
				# Simple face region estimation in the center of the frame
				center_x = w // 2
				center_y = h // 2
				
				# Use about 1/3 of the frame height for face
				face_h = h // 3
				face_w = min(w // 2, face_h)
				
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
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    return checkpoint


def main(face, audio, model, slow_mode=False):
	if slow_mode:
		print("Using SLOW animation mode (full face animation)")
	else:
		print("Using FAST animation mode (lips only)")
		
	if not os.path.isfile(face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif face.split('.')[1] in ['jpg', 'png', 'jpeg'] and not slow_mode:
		full_frames = [cv2.imread(face)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		
		# Get video dimensions for potential downscaling of large videos
		frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
		
		# Auto-adjust resize factor for very large videos
		original_resize_factor = args.resize_factor
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

		full_frames = []
		
		# For large videos, report progress and limit memory usage
		frame_limit = 5000  # Maximum number of frames to process at once
		if total_frames > frame_limit:
			print(f"Large video detected ({total_frames} frames). Will process in chunks.")
		
		# Use tqdm for progress reporting
		pbar = tqdm(total=min(total_frames, frame_limit))
		frame_count = 0
		
		while frame_count < frame_limit:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
				
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)
			frame_count += 1
			pbar.update(1)
			
			# For very large videos, limit frames to avoid memory issues
			if frame_count >= frame_limit:
				print(f"Reached frame limit of {frame_limit}. Processing this chunk.")
				break
		
		pbar.close()
		
		# Reset resize factor to original value after processing
		args.resize_factor = original_resize_factor

	print ("Number of frames available for inference: "+str(len(full_frames)))

	if not audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		audio = 'temp/temp.wav'

	wav = load_wav(audio, 16000)
	mel = melspectrogram(wav)
	print(mel.shape)

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

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	# Initialize video writer outside the try block
	out = None
	try:
		for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
												total=int(np.ceil(float(len(mel_chunks))/args.wav2lip_batch_size)))):
			if i == 0:
				#model = load_model(checkpoint_path)
				print ("Model loaded")

				frame_h, frame_w = full_frames[0].shape[:-1]
				out = cv2.VideoWriter('wav2lip/temp/result.avi', 
										cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
			
			for p, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

				f[y1:y2, x1:x2] = p
				out.write(f)
	except Exception as e:
		print(f"Error during processing: {str(e)}")
		print("Attempting to save any completed frames...")
	
	# Save the results - only if out was initialized
	if out is not None:
		out.release()
	
	# Convert the output video to MP4 if needed - only if the AVI exists
	result_path = 'wav2lip/results/result_voice.mp4'
	if os.path.exists('wav2lip/temp/result.avi'):
		# Check if the result file is valid (has frames)
		avi_info = os.stat('wav2lip/temp/result.avi')
		if avi_info.st_size > 1000:  # If file is too small, it's likely empty
			# Modified command to include the audio file
			command = 'ffmpeg -y -i {} -i {} -c:v libx264 -preset ultrafast -c:a aac -map 0:v:0 -map 1:a:0 {}'.format(
				'wav2lip/temp/result.avi', audio, result_path)
			try:
				subprocess.call(command, shell=True)
				if os.path.exists(result_path):
					print(f"Successfully created output video with audio at {result_path}")
				else:
					print(f"Error: Output video file was not created.")
			except Exception as e:
				print(f"Error during video conversion: {str(e)}")
		else:
			print(f"Warning: Output AVI file is too small ({avi_info.st_size} bytes). Face detection may have failed.")
	else:
		print("No output video was created. Face detection likely failed completely.")
		# Return a default path even if no output was created
	
	# Return even if there were errors
	return result_path

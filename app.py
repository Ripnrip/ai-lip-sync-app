import os
import streamlit as st
from streamlit_image_select import image_select
import torch
from streamlit_mic_recorder import mic_recorder
from wav2lip import inference 
from wav2lip.models import Wav2Lip
import gdown
import warnings
import cv2
import numpy as np
import librosa
from pathlib import Path
import subprocess
import time
from PIL import Image
import matplotlib.pyplot as plt
import sys
import threading
import concurrent.futures

# Suppress warnings
warnings.filterwarnings('ignore')

# More comprehensive fix for Streamlit file watcher issues with PyTorch
os.environ['STREAMLIT_WATCH_IGNORE'] = 'torch'
if 'torch' in sys.modules:
    sys.modules['torch'].__path__ = type('', (), {'_path': []})()

# Check if MPS (Apple Silicon GPU) is available, otherwise use CPU
if torch.backends.mps.is_available():
    device = 'mps'
    # Enable memory optimization for Apple Silicon
    torch.mps.empty_cache()
    # Set the memory format to optimize for M2 Max
    torch._C._set_cudnn_benchmark(True)
    st.success("Using Apple M2 Max GPU for acceleration with optimized settings!")
else:
    device = 'cpu'
    st.warning("Using CPU for inference (slower). GPU acceleration not available.")

print(f"Using {device} for inference.")

# Add functions to analyze video and audio quality
def analyze_video_quality(file_path):
    """Analyze video quality and detect faces for better user guidance"""
    try:
        # Open the video file
        video = cv2.VideoCapture(file_path)
        
        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Read a frame for face detection
        success, frame = video.read()
        if not success:
            return {
                "resolution": f"{width}x{height}",
                "fps": fps,
                "duration": f"{duration:.1f} seconds",
                "quality": "Unknown",
                "face_detected": False,
                "message": "Could not analyze video content."
            }
        
        # Detect faces using OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Determine quality score based on resolution and face detection
        quality_score = 0
        
        # Resolution assessment
        if width >= 1920 or height >= 1080:  # 1080p or higher
            resolution_quality = "Excellent"
            quality_score += 3
        elif width >= 1280 or height >= 720:  # 720p
            resolution_quality = "Good"
            quality_score += 2
        elif width >= 640 or height >= 480:  # 480p
            resolution_quality = "Fair"
            quality_score += 1
        else:
            resolution_quality = "Low"
            
        # Overall quality assessment
        face_detected = len(faces) > 0
        
        if face_detected:
            quality_score += 2
            face_message = "Face detected! ✅"
            
            # Check face size relative to frame
            for (x, y, w, h) in faces:
                face_area_ratio = (w * h) / (width * height)
                if face_area_ratio > 0.1:  # Face takes up at least 10% of frame
                    quality_score += 1
                    face_size = "Good face size"
                else:
                    face_size = "Face may be too small"
        else:
            face_message = "No face detected! ⚠️ Lip sync results may be poor."
            face_size = "N/A"
        
        # Determine overall quality
        if quality_score >= 5:
            quality = "Excellent"
        elif quality_score >= 3:
            quality = "Good"
        elif quality_score >= 1:
            quality = "Fair"
        else:
            quality = "Poor"
            
        # Release video resource
        video.release()
        
        return {
            "resolution": f"{width}x{height}",
            "fps": f"{fps:.1f}",
            "duration": f"{duration:.1f} seconds",
            "quality": quality,
            "resolution_quality": resolution_quality,
            "face_detected": face_detected,
            "face_message": face_message,
            "face_size": face_size,
            "message": get_video_recommendation(quality, face_detected, width, height)
        }
        
    except Exception as e:
        return {
            "quality": "Error",
            "message": f"Could not analyze video: {str(e)}"
        }

def analyze_audio_quality(file_path):
    """Analyze audio quality for better user guidance"""
    try:
        # Load audio file using librosa
        y, sr = librosa.load(file_path, sr=None)
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate audio features
        rms = librosa.feature.rms(y=y)[0]
        mean_volume = np.mean(rms)
        
        # Simple speech detection (using energy levels)
        has_speech = np.max(rms) > 0.05
        
        # Check for silence periods
        silence_threshold = 0.01
        silence_percentage = np.mean(rms < silence_threshold) * 100
        
        # Calculate quality score
        quality_score = 0
        
        # Volume assessment
        if 0.05 <= mean_volume <= 0.2:
            volume_quality = "Good volume levels"
            quality_score += 2
        elif mean_volume > 0.2:
            volume_quality = "Audio might be too loud"
            quality_score += 1
        else:
            volume_quality = "Audio might be too quiet"
            
        # Speech detection
        if has_speech:
            speech_quality = "Speech detected ✅"
            quality_score += 2
        else:
            speech_quality = "Speech may not be clear ⚠️"
        
        # Silence assessment (some silence is normal)
        if silence_percentage < 40:
            silence_quality = "Good speech-to-silence ratio"
            quality_score += 1
        else:
            silence_quality = "Too much silence detected"
        
        # Determine overall quality
        if quality_score >= 4:
            quality = "Excellent"
        elif quality_score >= 2:
            quality = "Good"
        elif quality_score >= 1:
            quality = "Fair"
        else:
            quality = "Poor"
            
        return {
            "duration": f"{duration:.1f} seconds",
            "quality": quality,
            "volume_quality": volume_quality,
            "speech_quality": speech_quality,
            "silence_quality": silence_quality,
            "message": get_audio_recommendation(quality, has_speech, mean_volume, silence_percentage)
        }
        
    except Exception as e:
        return {
            "quality": "Error",
            "message": f"Could not analyze audio: {str(e)}"
        }

def get_video_recommendation(quality, face_detected, width, height):
    """Get recommendations based on video quality"""
    if not face_detected:
        return "⚠️ No face detected. For best results, use a video with a clear, well-lit face looking toward the camera."
    
    if quality == "Poor":
        return "⚠️ Low quality video. Consider using a higher resolution video with better lighting and a clearly visible face."
    
    if width < 640 or height < 480:
        return "⚠️ Video resolution is low. For better results, use a video with at least 480p resolution."
    
    if quality == "Excellent":
        return "✅ Great video quality! This should work well for lip syncing."
    
    return "✅ Video quality is acceptable for lip syncing."

def get_audio_recommendation(quality, has_speech, volume, silence_percentage):
    """Get recommendations based on audio quality"""
    if not has_speech:
        return "⚠️ Speech may not be clearly detected. For best results, use audio with clear speech."
    
    if quality == "Poor":
        return "⚠️ Low quality audio. Consider using clearer audio with consistent volume levels."
    
    if volume < 0.01:
        return "⚠️ Audio volume is very low. This may result in poor lip sync."
    
    if volume > 0.3:
        return "⚠️ Audio volume is very high. This may cause distortion in lip sync."
    
    if silence_percentage > 50:
        return "⚠️ Audio contains a lot of silence. Lip sync will only work during speech sections."
    
    if quality == "Excellent":
        return "✅ Great audio quality! This should work well for lip syncing."
    
    return "✅ Audio quality is acceptable for lip syncing."

#@st.cache_data is used to only load the model once
#@st.cache_data 
@st.cache_resource
def load_model(path):
    st.write("Please wait for the model to be loaded or it will cause an error")
    wav2lip_checkpoints_url = "https://drive.google.com/drive/folders/1Sy5SHRmI3zgg2RJaOttNsN3iJS9VVkbg?usp=sharing"
    if not os.path.exists(path):
        gdown.download_folder(wav2lip_checkpoints_url, quiet=True, use_cookies=False)
    st.write("Please wait")
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    
    # Optimize model loading for M2 Max
    if device == 'mps':
        # Clear cache before loading model
        torch.mps.empty_cache()
        
    # Load model with device mapping
    checkpoint = torch.load(path, map_location=torch.device(device))
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    
    # Set model to evaluation mode and optimize for inference
    model.eval()
    if device == 'mps':
        # Attempt to optimize the model for inference
        try:
            # Use torch's inference mode for optimized inference
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
            print("Applied M2 Max optimizations")
        except:
            print("Could not apply all M2 Max optimizations")
            
    st.write(f"Model loaded successfully on {device} with optimized settings for M2 Max!")
    return model
@st.cache_resource
def load_avatar_videos_for_slow_animation(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
            
            avatar_videos_url = "https://drive.google.com/drive/folders/1h9pkU5wenrS2vmKqXBfFmrg-1hYw5s4q?usp=sharing"
            print(f"Downloading avatar videos from: {avatar_videos_url}")
            gdown.download_folder(avatar_videos_url, quiet=False, use_cookies=False)
            print(f"Avatar videos downloaded successfully to: {path}")
        except Exception as e:
            print(f"Error downloading avatar videos: {str(e)}")
            # Create default empty videos if download fails
            for avatar_file in ["avatar1.mp4", "avatar2.mp4", "avatar3.mp4"]:
                video_path = os.path.join(path, avatar_file)
                if not os.path.exists(video_path):
                    print(f"Creating empty video file: {video_path}")
                    # Get the matching image
                    img_key = f"avatars_images/{os.path.splitext(avatar_file)[0]}" + (".jpg" if avatar_file != "avatar3.mp4" else ".png")
                    try:
                        # Create a video from the image
                        img = cv2.imread(img_key)
                        if img is not None:
                            # Create a short 5-second video from the image
                            print(f"Creating video from image: {img_key}")
                            height, width = img.shape[:2]
                            output_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                            for _ in range(150):  # 5 seconds at 30 fps
                                output_video.write(img)
                            output_video.release()
                        else:
                            print(f"Could not read image: {img_key}")
                    except Exception as e:
                        print(f"Error creating video from image: {str(e)}")
    else:
        print(f"Avatar videos directory already exists: {path}")
        # Check if files exist in the directory
        files = os.listdir(path)
        if not files:
            print(f"No files found in {path}, directory exists but is empty")
        else:
            print(f"Found {len(files)} files in {path}: {', '.join(files)}")



image_video_map = {
      				"avatars_images/avatar1.jpg":"avatars_videos/avatar1.mp4",
                    "avatars_images/avatar2.jpg":"avatars_videos/avatar2.mp4",
                    "avatars_images/avatar3.png":"avatars_videos/avatar3.mp4"
                              }
def streamlit_look():
    """
    Modest front-end code:)
    """
    data={}
    st.title("Welcome to AI Lip Sync :)")
    
    # Add a brief app description
    st.markdown("""
    This app uses AI to synchronize a person's lip movements with any audio file. 
    You can choose from built-in avatars or upload your own image/video, then provide audio 
    to create realistic lip-synced videos. Powered by Wav2Lip and optimized for Apple Silicon.
    """)
    
    # Add a guidelines section with an expander for best practices
    with st.expander("📋 Guidelines & Best Practices (Click to expand)", expanded=False):
        st.markdown("""
        ### Guidelines for Best Results

        #### Audio and Video Length
        - Audio and video don't need to be exactly the same length
        - If audio is shorter than video: Only the matching portion will be lip-synced
        - If audio is longer than video: Audio will be trimmed to match video length

        #### Face Quality
        - Clear, well-lit frontal views of faces work best
        - Faces should take up a reasonable portion of the frame
        - Avoid extreme angles, heavy shadows, or partial face views

        #### Audio Quality
        - Clear speech with minimal background noise works best
        - Consistent audio volume improves synchronization
        - Supported formats: WAV, MP3

        #### Video Quality
        - Stable videos with minimal camera movement
        - The person's mouth should be clearly visible
        - Videos at 480p or higher resolution work best
        - Very high-resolution videos will be automatically downscaled

        #### Processing Tips
        - Shorter videos process faster and often give better results
        - "Fast animation" only moves the lips (quicker processing)
        - "Slow animation" animates the full face (better quality, slower)
        - Your M2 Max GPU will significantly speed up processing
        """)
    
    # Option to choose between built-in avatars or upload a custom one
    avatar_source = st.radio("Choose avatar source:", ["Upload my own image/video", "Use built-in avatars"])
    
    if avatar_source == "Use built-in avatars":
        st.write("Please choose your avatar from the following options:")
        avatar_img = image_select("", 
                                ["avatars_images/avatar1.jpg",
                                "avatars_images/avatar2.jpg",
                                "avatars_images/avatar3.png",
                                        ])
        data["imge_path"] = avatar_img
    else:
        st.write("Upload an image or video file for your avatar:")
        uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"], key="avatar_uploader")
        
        if uploaded_file is not None:
            # Save the uploaded file
            file_path = os.path.join("uploads", uploaded_file.name)
            os.makedirs("uploads", exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Set the file path as image path
            data["imge_path"] = file_path
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Preview the uploaded image/video
            if uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
                st.image(file_path, caption="Uploaded Image")
            elif uploaded_file.name.endswith('.mp4'):
                st.video(file_path)
                
                # Analyze video quality for MP4 files
                with st.spinner("Analyzing video quality..."):
                    video_analysis = analyze_video_quality(file_path)
                
                # Display video quality analysis in a nice box
                with st.expander("📊 Video Quality Analysis", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Resolution:** {video_analysis['resolution']}")
                        st.markdown(f"**FPS:** {video_analysis['fps']}")
                        st.markdown(f"**Duration:** {video_analysis['duration']}")
                    
                    with col2:
                        quality_color = {
                            "Excellent": "green",
                            "Good": "lightgreen",
                            "Fair": "orange",
                            "Poor": "red",
                            "Error": "red"
                        }.get(video_analysis['quality'], "gray")
                        
                        st.markdown(f"**Quality:** <span style='color:{quality_color};font-weight:bold'>{video_analysis['quality']}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Face Detection:** {'✅ Detected' if video_analysis.get('face_detected', False) else '❌ Not detected'}")
                    
                    # Display the recommendation
                    st.info(video_analysis['message'])
    
    # Option to choose between mic recording or upload audio file
    audio_source = st.radio("Choose audio source:", ["Upload audio file", "Record with microphone"])
    
    if audio_source == "Record with microphone":
        audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording", 
            just_once=False,
            use_container_width=False,
            callback=None,
            args=(),
            kwargs={},
            key=None)
        
        if audio:
            st.audio(audio["bytes"])
            data["audio"] = audio["bytes"]
    else:
        st.write("Upload an audio file:")
        uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3"], key="audio_uploader")
        
        if uploaded_audio is not None:
            # Save the uploaded audio file
            audio_path = os.path.join("uploads", uploaded_audio.name)
            os.makedirs("uploads", exist_ok=True)
            
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
            
            # Preview the uploaded audio
            st.audio(audio_path)
            
            # Read the file into bytes for consistency with microphone recording
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            data["audio"] = audio_bytes
            st.success(f"Audio file uploaded successfully: {uploaded_audio.name}")
            
            # Analyze audio quality
            with st.spinner("Analyzing audio quality..."):
                audio_analysis = analyze_audio_quality(audio_path)
            
            # Display audio quality analysis in a nice box
            with st.expander("🎵 Audio Quality Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Duration:** {audio_analysis['duration']}")
                    st.markdown(f"**Volume:** {audio_analysis['volume_quality']}")
                
                with col2:
                    quality_color = {
                        "Excellent": "green",
                        "Good": "lightgreen",
                        "Fair": "orange",
                        "Poor": "red",
                        "Error": "red"
                    }.get(audio_analysis['quality'], "gray")
                    
                    st.markdown(f"**Quality:** <span style='color:{quality_color};font-weight:bold'>{audio_analysis['quality']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Speech:** {audio_analysis['speech_quality']}")
                
                # Display the recommendation
                st.info(audio_analysis['message'])
    
    return data

def main():  
    # Initialize session state to track processing status
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        
    data = streamlit_look()
    
    # Add debug information
    st.write("Debug info:")
    if "imge_path" in data:
        st.write(f"Image/Video path: {data['imge_path']}")
    else:
        st.write("No image/video selected yet")
        
    if "audio" in data:
        st.write("Audio file selected ✓")
    else:
        st.write("No audio selected yet")
    
    # Only proceed if we have both image/video and audio data
    if "imge_path" in data and "audio" in data:
        st.write("This app will automatically save your audio when you click animate.")
        save_record = st.button("save record manually")
        st.write("With fast animation only the lips of the avatar will move, and it will take probably less than a minute for a record of about 30 seconds, but with slow animation choice, the full face of the avatar will move and it will take about 30 minutes for a record of about 30 seconds to get ready.")
        model = load_model("wav2lip_checkpoints/wav2lip_gan.pth")
        
        # Check for duration mismatches between video and audio
        if data["imge_path"].endswith('.mp4'):
            # Save audio to temp file for analysis
            if not os.path.exists('record.wav'):
                with open('record.wav', mode='wb') as f:
                    f.write(data["audio"])
            
            # Get durations
            video_duration = get_video_duration(data["imge_path"])
            audio_duration = get_audio_duration('record.wav')
            
            # Check for significant duration mismatch (more than 2 seconds difference)
            if abs(video_duration - audio_duration) > 2:
                st.warning(f"⚠️ Duration mismatch detected: Video is {video_duration:.1f}s and Audio is {audio_duration:.1f}s")
                
                # Create a tab for handling duration mismatches
                with st.expander("Duration Mismatch Options (Click to expand)", expanded=True):
                    st.info("The video and audio have different durations. Choose an option below:")
                    
                    if video_duration > audio_duration:
                        if st.button("Trim Video to Match Audio Duration"):
                            # Update duration values to match
                            output_path = 'uploads/trimmed_input_video.mp4'
                            with st.spinner(f"Trimming video from {video_duration:.1f}s to {audio_duration:.1f}s..."):
                                success = trim_video(data["imge_path"], output_path, 0, audio_duration)
                                
                            if success:
                                st.success("Video trimmed to match audio duration!")
                                # Update the image path to use the trimmed video
                                data["imge_path"] = output_path
                                st.video(output_path)
                    else:  # audio_duration > video_duration
                        if st.button("Trim Audio to Match Video Duration"):
                            # Update duration values to match
                            output_path = 'uploads/trimmed_input_audio.wav'
                            with st.spinner(f"Trimming audio from {audio_duration:.1f}s to {video_duration:.1f}s..."):
                                success = trim_audio('record.wav', output_path, 0, video_duration)
                                
                            if success:
                                st.success("Audio trimmed to match video duration!")
                                # Update the audio data with the trimmed audio
                                with open(output_path, "rb") as f:
                                    data["audio"] = f.read()
                                # Save the trimmed audio as record.wav
                                with open('record.wav', mode='wb') as f:
                                    f.write(data["audio"])
                                st.audio(output_path)
        
        # Animation buttons
        fast_animate = st.button("fast animate")
        slower_animate = st.button("slower animate")
        
        # Function to save the audio record
        def save_audio_record():
            if os.path.exists('record.wav'):
                os.remove('record.wav')
            with open('record.wav', mode='wb') as f:
                f.write(data["audio"])
            st.write("Audio record saved!")
        
        if save_record:
            save_audio_record()
        
        # Show previously generated results if they exist and we're not generating new ones
        if os.path.exists('wav2lip/results/result_voice.mp4') and st.session_state.processed and not (fast_animate or slower_animate):
            st.video('wav2lip/results/result_voice.mp4')
            display_trim_options('wav2lip/results/result_voice.mp4')
        
        if fast_animate:
            # Automatically save the record before animation
            save_audio_record()
            
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            progress_bar = progress_placeholder.progress(0, text="Processing: 0% complete")
            status_placeholder.info("Preparing to process...")
            
            # Call the inference function inside a try block with progress updates at key points
            try:
                # Initialize a progress tracker
                progress_steps = [
                    (0, "Starting processing..."),
                    (15, "Step 1/4: Loading and analyzing video frames"),
                    (30, "Step 2/4: Performing face detection (this may take a while for long videos)"),
                    (60, "Step 3/4: Generating lip-synced frames"),
                    (80, "Step 4/4: Creating final video with audio"),
                    (100, "Processing complete!")
                ]
                current_step = 0
                
                # Redirect stdout to capture progress information
                import io
                sys.stdout = io.StringIO()
                
                # Update progress for the initial step
                progress, message = progress_steps[current_step]
                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                status_placeholder.info(message)
                current_step += 1
                
                # Run the inference in a background thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Start the inference process
                    future = executor.submit(inference.main, data["imge_path"], "record.wav", model)
                    
                    # Monitor the output for progress indicators
                    while not future.done():
                        captured_output = sys.stdout.getvalue()
                        
                        # Check for progress indicators and update UI
                        if current_step < len(progress_steps):
                            # Check for stage 1 completion: frames read
                            if current_step == 1 and "Number of frames available for inference" in captured_output:
                                progress, message = progress_steps[current_step]
                                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                                status_placeholder.info(message)
                                current_step += 1
                            # Check for stage 2 completion: face detection
                            elif current_step == 2 and "Face detection completed successfully" in captured_output:
                                progress, message = progress_steps[current_step]
                                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                                status_placeholder.info(message)
                                current_step += 1
                            # Check for stage 3 completion: ffmpeg started
                            elif current_step == 3 and "ffmpeg" in captured_output:
                                progress, message = progress_steps[current_step]
                                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                                status_placeholder.info(message)
                                current_step += 1
                        
                        # Sleep to avoid excessive CPU usage
                        time.sleep(0.5)
                    
                    try:
                        # Get the result or propagate exceptions
                        future.result()
                        
                        # Show completion
                        progress, message = progress_steps[-1]
                        progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                        status_placeholder.success("Lip sync complete! Your video is ready.")
                    except Exception as e:
                        raise e
                
                # Restore stdout
                sys.stdout = sys.__stdout__
                
                if os.path.exists('wav2lip/results/result_voice.mp4'):
                    st.video('wav2lip/results/result_voice.mp4')
                    display_trim_options('wav2lip/results/result_voice.mp4')
                    # Set processed flag to True after successful processing
                    st.session_state.processed = True
                    
            except Exception as e:
                # Restore stdout in case of error
                sys.stdout = sys.__stdout__
                
                progress_placeholder.empty()
                status_placeholder.error(f"Error during processing: {str(e)}")
                st.error("Failed to generate video. Please try again or use a different image/audio.")
        
        if slower_animate:
            # Automatically save the record before animation
            save_audio_record()
            
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            progress_bar = progress_placeholder.progress(0, text="Processing: 0% complete")
            status_placeholder.info("Preparing to process...")
            
            # Derive the video path from the selected avatar
            if data["imge_path"].endswith('.mp4'):
                video_path = data["imge_path"]
            else:
                # Get the avatar video path for the selected avatar
                avatar_list = load_avatar_videos_for_slow_animation("./data/avatars/samples")
                video_path = avatar_list[available_avatars_for_slow.index(avatar_choice)]
            
            try:
                # Initialize a progress tracker
                progress_steps = [
                    (0, "Starting processing..."),
                    (15, "Step 1/4: Loading and analyzing video frames"),
                    (30, "Step 2/4: Performing face detection (this may take a while for long videos)"),
                    (60, "Step 3/4: Generating lip-synced frames with full-face animation"),
                    (80, "Step 4/4: Creating final video with audio"),
                    (100, "Processing complete!")
                ]
                current_step = 0
                
                # Redirect stdout to capture progress information
                import io
                sys.stdout = io.StringIO()
                
                # Update progress for the initial step
                progress, message = progress_steps[current_step]
                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                status_placeholder.info(message)
                current_step += 1
                
                # Run the inference in a background thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Start the inference process
                    future = executor.submit(inference.main, video_path, "record.wav", model, slow_mode=True)
                    
                    # Monitor the output for progress indicators
                    while not future.done():
                        captured_output = sys.stdout.getvalue()
                        
                        # Check for progress indicators and update UI
                        if current_step < len(progress_steps):
                            # Check for stage 1 completion: frames read
                            if current_step == 1 and "Number of frames available for inference" in captured_output:
                                progress, message = progress_steps[current_step]
                                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                                status_placeholder.info(message)
                                current_step += 1
                            # Check for stage 2 completion: face detection
                            elif current_step == 2 and "Face detection completed successfully" in captured_output:
                                progress, message = progress_steps[current_step]
                                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                                status_placeholder.info(message)
                                current_step += 1
                            # Check for stage 3 completion: ffmpeg started
                            elif current_step == 3 and "ffmpeg" in captured_output:
                                progress, message = progress_steps[current_step]
                                progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                                status_placeholder.info(message)
                                current_step += 1
                        
                        # Sleep to avoid excessive CPU usage
                        time.sleep(0.5)
                    
                    try:
                        # Get the result or propagate exceptions
                        future.result()
                        
                        # Show completion
                        progress, message = progress_steps[-1]
                        progress_bar.progress(progress, text=f"Processing: {progress}% complete")
                        status_placeholder.success("Lip sync complete! Your video is ready.")
                    except Exception as e:
                        raise e
                
                # Restore stdout
                sys.stdout = sys.__stdout__
                
                if os.path.exists('wav2lip/results/result_voice.mp4'):
                    st.video('wav2lip/results/result_voice.mp4')
                    display_trim_options('wav2lip/results/result_voice.mp4')
                    # Set processed flag to True after successful processing
                    st.session_state.processed = True
            except Exception as e:
                # Restore stdout in case of error
                sys.stdout = sys.__stdout__
                
                progress_placeholder.empty()
                status_placeholder.error(f"Error during processing: {str(e)}")
                st.error("Failed to generate video. Please try again or use a different video/audio.")
    else:
        if "imge_path" not in data and "audio" not in data:
            st.warning("Please upload both an image/video AND provide audio to continue.")
        elif "imge_path" not in data:
            st.warning("Please select or upload an image/video to continue.")
        else:
            st.warning("Please provide audio to continue.")

# Function to display trim options and handle video trimming
def display_trim_options(video_path):
    """Display options to trim the video and handle the trimming process"""
    st.subheader("Video Processing Options")
    
    # Check if the video exists first
    if not os.path.exists(video_path):
        st.error(f"Video file not found at {video_path}. Try running the animation again.")
        return
    
    # Add tabs for different operations
    download_tab, trim_tab = st.tabs(["Download Original", "Trim Video"])
    
    with download_tab:
        st.write("Download the original generated video:")
        try:
            st.video(video_path)
            st.download_button(
                label="Download Original Video",
                data=open(video_path, 'rb').read(),
                file_name="original_lip_sync_video.mp4",
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
    
    with trim_tab:
        st.write("You can trim the generated video to remove unwanted parts from the beginning or end.")
        
        duration = get_video_duration(video_path)
        if duration <= 0:
            st.error("Could not determine video duration")
            return
        
        # Display video duration
        st.write(f"Video duration: {duration:.2f} seconds")
        
        # Create a slider for selecting start and end times
        col1, col2 = st.columns(2)
        
        with col1:
            start_time = st.slider("Start time (seconds)", 
                                min_value=0.0, 
                                max_value=float(duration), 
                                value=0.0,
                                step=0.1)
            st.write(f"Start at: {start_time:.1f}s")
        
        with col2:
            end_time = st.slider("End time (seconds)", 
                                min_value=0.0, 
                                max_value=float(duration), 
                                value=float(duration),
                                step=0.1)
            st.write(f"End at: {end_time:.1f}s")
        
        # Display trim duration
        trim_duration = end_time - start_time
        st.info(f"Trimmed video duration will be: {trim_duration:.1f} seconds")
        
        # Validate the selected range
        if start_time >= end_time:
            st.error("Start time must be less than end time")
            return
        
        # Button to perform trimming
        if st.button("Trim Video"):
            # Generate output path
            output_path = 'wav2lip/results/trimmed_video.mp4'
            
            # Show progress
            with st.spinner("Trimming video..."):
                success = trim_video(video_path, output_path, start_time, end_time)
            
            if success:
                st.success("Video trimmed successfully!")
                try:
                    st.video(output_path)
                    
                    # Add download button for trimmed video
                    st.download_button(
                        label="Download Trimmed Video",
                        data=open(output_path, 'rb').read(),
                        file_name="trimmed_lip_sync_video.mp4",
                        mime="video/mp4"
                    )
                except Exception as e:
                    st.error(f"Error displaying trimmed video: {str(e)}")
            else:
                st.error("Failed to trim video. Try again with different timing parameters.")

# Function to trim video using ffmpeg
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
            st.error(f"Input video not found: {input_path}")
            return False
            
        # Format the command - use -ss before -i for faster seeking
        # Add quotes around file paths to handle spaces and special characters
        command = f'ffmpeg -y -ss {start_time} -i "{input_path}" -to {end_time} -c:v copy -c:a copy "{output_path}"'
        
        # Use subprocess.run for better error handling
        result = subprocess.run(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            st.error(f"FFMPEG error: {result.stderr}")
            return False
            
        # Verify the output file exists and has a size greater than 0
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            st.error("Output file was not created correctly")
            return False
            
    except Exception as e:
        st.error(f"Error trimming video: {str(e)}")
        return False

# Function to get video duration
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
        st.error(f"Error getting video duration: {str(e)}")
        return 0

# Function to get audio duration
def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        st.error(f"Error getting audio duration: {str(e)}")
        return 0

# Function to trim audio file
def trim_audio(input_path, output_path, start_time, end_time):
    """Trim an audio file to the specified start and end times"""
    try:
        # Command to trim audio using ffmpeg
        command = f'ffmpeg -y -i "{input_path}" -ss {start_time} -to {end_time} -c copy "{output_path}"'
        
        # Execute the command
        subprocess.call(command, shell=True)
        
        # Check if output file exists
        if os.path.exists(output_path):
            return True
        else:
            st.error("Output audio file was not created correctly")
            return False
            
    except Exception as e:
        st.error(f"Error trimming audio: {str(e)}")
        return False

if __name__ == "__main__":
    main()
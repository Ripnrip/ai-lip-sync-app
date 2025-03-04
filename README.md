# AI Lip Sync

![Screenshot 2024-01-22 at 03-03-09 app · Streamlit](https://github.com/Aml-Hassan-Abd-El-hamid/AI-Lip-Sync/assets/66205928/d35f379f-f1ca-46e5-a113-bfa3f3c4c2f9)

An AI-powered application that synchronizes lip movements with audio input, built with Wav2Lip and Streamlit.

## Features

- **Multiple Avatar Options**: Choose from built-in avatars or upload your own image/video
- **Audio Input Flexibility**: Record audio directly or upload WAV/MP3 files
- **Quality Assessment**: Automatic analysis of video and audio quality with recommendations
- **GPU Acceleration**: Optimized for Apple Silicon (M1/M2) GPUs
- **Two Animation Modes**: Fast (lips only) or Slow (full face animation)
- **Video Trimming**: Trim the output video to remove unwanted portions

## Quick Setup Guide

### Prerequisites

- Python 3.9+ 
- ffmpeg (for audio processing)
- Git LFS (optional, for handling large model files)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-lip-sync-app.git
   cd ai-lip-sync-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On macOS/Linux
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install system dependencies:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install $(cat packages.txt)
   
   # On macOS with Homebrew
   brew install ffmpeg
   ```

5. Run the application:
   ```bash
   python -m streamlit run app.py
   ```
   
   > **Note**: If you encounter a "streamlit: command not found" error, always use `python -m streamlit run app.py` instead of `streamlit run app.py`

The application will automatically download the required model files on first run.

## Usage Guide

1. **Choose Avatar Source**:
   - Select from built-in avatars or upload your own image/video
   - For best results, use clear frontal face images/videos

2. **Provide Audio**:
   - Record directly using your microphone
   - Upload WAV or MP3 files

3. **Quality Assessment**:
   - The app will automatically analyze your uploaded video and audio
   - Review the quality analysis and recommendations
   - Make adjustments if needed for better results

4. **Generate Animation**:
   - Choose "Fast animate" for quicker processing (lips only)
   - Choose "Slower animate" for more realistic results (full face)

5. **View and Edit Results**:
   - The generated video will appear in the app
   - Use the trim feature to remove unwanted portions from the start or end
   - Download the original or trimmed version to your computer

## Video Trimming Feature

The app now includes a video trimming capability:

- After generating a lip-sync video, you'll see trimming options below the result
- Use the sliders to select the start and end times for your trimmed video
- Click "Trim Video" to create a shortened version
- Both original and trimmed videos can be downloaded directly from the app

## Quality Assessment Feature

The app now includes automatic quality assessment for uploaded videos and audio:

### Video Analysis:
- Resolution check (higher resolution = better results)
- Face detection (confirms a face is present and properly sized)
- Frame rate analysis
- Overall quality score with specific recommendations

### Audio Analysis:
- Speech detection (confirms speech is present)
- Volume level assessment
- Silence detection
- Overall quality score with specific recommendations

## Troubleshooting

- **"No face detected" error**: Ensure your video has a clear, well-lit frontal face
- **Poor lip sync results**: Try using higher quality audio with clear speech
- **Performance issues**: For large videos, try the "Fast animate" option or use a smaller video clip
- **Memory errors**: Close other applications to free up memory, or use a machine with more RAM

## Technical Details

The project is built on the Wav2Lip model with several optimizations:
- Apple Silicon (M1/M2) GPU acceleration using MPS backend
- Automatic video resolution scaling for large videos
- Memory optimizations for processing longer videos
- Quality assessment using OpenCV and librosa

## Original Project Background

The project started as a part of an interview process with some company, I received an email with the following task:

Assignment Object:<br>
          &emsp;&emsp;&emsp;&emsp;Your task is to develop a lip-syncing model using machine learning
          techniques. It takes an input image and audio and then generates a video
          where the image appears to lip sync with the provided audio. You have to
          develop this task using python3.

Requirements:<br>
        &emsp;&emsp;&emsp;&emsp;● Avatar / Image : Get one AI-generated avatar, the avatar may be for a<br>
        &emsp;&emsp;&emsp;&emsp;man, woman, old man, old lady or a child. Ensure that the avatar is<br>
        &emsp;&emsp;&emsp;&emsp;created by artificial intelligence and does not represent real<br>
        &emsp;&emsp;&emsp;&emsp;individuals.<br>
        &emsp;&emsp;&emsp;&emsp;● Audio : Provide two distinct and clear audio recordings—one in Arabic<br>
        &emsp;&emsp;&emsp;&emsp;and the other in English. The duration of each audio clip should be<br>
        &emsp;&emsp;&emsp;&emsp;no less than 30 seconds and no more than 1 minute.<br>
        &emsp;&emsp;&emsp;&emsp;● Lip-sync model: Develop a lip-syncing model to synchronise the lip<br>
        &emsp;&emsp;&emsp;&emsp;movements of the chosen avatar with the provided audio. Ensure the<br>
        &emsp;&emsp;&emsp;&emsp;model demonstrates proficiency in accurately aligning lip motions<br>
        &emsp;&emsp;&emsp;&emsp;with the spoken words in both Arabic and English.<br>
        &emsp;&emsp;&emsp;&emsp;Hint : You can refer to state of the art models in lip-syncing.<br>
         
I was given about 96 hours to accomplish this task, I spent the first 12 hours sick with a very bad flu and no proper internet connection so I had 84 hours!<br>
After submitting the task on time, I took more time to deploy the project on Streamlight, as I thought it was a fun project and would be a nice addition to my CV:)  

Given the provided hint from the company, "You can refer to state-of-the-art models in lip-syncing.", I started looking into the available open-source pre-trained model that can accomplish this task and most available resources pointed towards **Wav2Lip**. I found a couple of interesting tutorials for that model that I will share below.

### How to run the application locally:<br>

1- clone the repo to your local machine.<br>
2- open your terminal inside the project folder and run the following command: `pip install -r requirements.txt` and then run this command `sudo xargs -a packages.txt apt-get install` to install the needed modules and packages.<br>
3- open your terminal inside the project folder and run the following command: `streamlit run app.py` to run the streamlit application.<br>

### Things I changed in the wav2lip and why:<br>

In order to work with and deploy the wav2lip model I had to make the following changes:<br>
1- Changed the `_build_mel_basis()` function in `audio.py`, I had to do that to be able to work with `librosa>=0.10.0` package, check this [issue](https://github.com/Rudrabha/Wav2Lip/issues/550) for more details.<br>
2- Changed the `main()` function at the `inferance.py` to directly take an output from the `app.py` instead of using the command line arguments.<br>
3- I took the `load_model(path)` function and added it to `app.py` and added `@st.cache_data` in order to only load the model once, instead of using it multiple times, I also modified it<br>
4- Deleted the unnecessary files like the checkpoints to make the Streamlit website deployment easier.<br>
5- Since I'm using Streamlit for deployment and Streamlit Cloud doesn't support GPU, I had to change the device to work with `cpu` instead of `cuda`.<br>
6- I made other minor changes like changing the path to a file or modifying import statements.

### Issues I had with Streamlit, during the deployment:

This part is a documentation for me, just in case, I need to face an issue in the future and also could be helpful for any poor soul who would have to work with Streamlit:

1- 
```
Error downloading object: wav2lip/checkpoints/wav2lip_gan.pth (ca9ab7b): Smudge error: Error downloading wav2lip/checkpoints/wav2lip_gan.pth (ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8): batch request: git@github.com: Permission denied (publickey).: exit status 255

Errors logged to /mount/src/ai-lip-sync/.git/lfs/logs/20240121T212252.496674
```
This essentially Streamlit telling you that it can't handle that big file, upload it to Google Drive, and then load it using Python code later, and no `git lfs` won't solve the problem :)<br> 
A ground rule that I learned here is: that the lighter you make your app, the better and faster it is to deploy it.<br>
I opened a topic with that issue on the Streamlit forum, right [here](https://discuss.streamlit.io/t/file-upload-fails-with-error-downloading-object-wav2lip-checkpoints-wav2lip-gan-pth-ca9ab7b/60261)<br>

2- Other issues that I faced a lot were dependency issues -lots of them- and that was mostly due to the fact that I depended on `pipreqs` to write down my `requirements.txt`, that `pipreqs` missed up my modules, it added unneeded ones and missed others, unfortunately, it took me some time to discover that and really slowed me down.

3- 
```
 ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
I faced that problem during importing `cv2` -`openCv`- and the solution was to install `libgl1-mesa-dev` and some other packages using `apt`, you can't just add such packages to the `requirements.txt`, you need to create a file named `packages.txt` to do so.

4- Streamlit can't handle heavy processing, I discovered that when I tried to deploy the `slow animation` button to process video input alongside recording to get more accurate lip-syncing, the application failed directly when I used that button -and I tried to use it twice :)-, and that kinda make sense as Streamlit doesn't have a GPU or even a high ram space -I don't have a good GPU but I have about 64GB ram which was enough to run that function locally- and to solve that issue, I initiated another branch to contain the deployment version that doesn't have the `slow animation` button and used that branch for deployment while kept the main branch containing that button.

**Pushing the checkpoints files:**<br>

Given the size of those kind of files, There are 2 ways to handle that. 

At the start, I had to use git lfs, here's how to do it:<br>

1- Follow the installation instructions that are suitable for your system from [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) <br>
2- Use the command `git lfs track "*.pth"` to let git lfs know that those are your big files.<br>
3- When pushing from the command line -I usually use VS code but it usually doesn't work with big files like `.pth` files- you need to generate a personal access token, to do so, follow the instructions from [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token), and then copy the token<br>
4- When pushing the file from the terminal you will be asked to pass a password, don't pass your GitHub profile password, instead pass your personal access token that you got from step 3.

But then Streamlit wasn't capable of even pulling the repo! so I uploaded the model checkpoints and some other files to Google Drive, put them in a public folder, and then used a module called gdown to download those folders when needed! here's a [link](https://github.com/wkentaro/gdown) to that gdown, it's straightforward to use and install.


**Video preview of the application:**<br>

**fast animation version**<br>
Notice how only the lips are moving.

English version:

https://github.com/Aml-Hassan-Abd-El-hamid/AI-Lip-Sync/assets/66205928/36577ccb-5ec6-4bb4-b7ff-44bb52a4f984

Arabic version:



https://github.com/Aml-Hassan-Abd-El-hamid/ai-lip-sync-app/assets/66205928/4346aa6d-ea4e-400e-9124-1cce06b049df



**slower animation version**<br>
Notice how the eye and the whole face are moving instead of only the lips.<br>

Unfortunately, Streamlit can't handle the computational power that the slower animation version requires and that's why I made it only available on the offline version, which means that you need to run the application locally to try that version.

English version:

https://github.com/Aml-Hassan-Abd-El-hamid/AI-Lip-Sync/assets/66205928/26740856-52e5-4fe7-868d-3b9341e97064

Arabic version:



https://github.com/Aml-Hassan-Abd-El-hamid/ai-lip-sync-app/assets/66205928/ba97daca-b30d-4179-9387-a382abbca3ba



The only difference between the fast and slow versions of animation here is the fact that the fast version passes only a photo while the slow version passes a video instead.

## Command Line Interface

In addition to the Streamlit web interface, this project now includes a command-line interface that allows you to run the lip sync directly:

```bash
./run_lipsync.py --face <path_to_face_file> --audio <path_to_audio_file> [options]
```

### Required Arguments:
- `--face`: Path to video/image file for lip sync
- `--audio`: Path to audio file (wav or mp3)

### Optional Arguments:
- `--model`: Path to Wav2Lip model checkpoint file (default: wav2lip_checkpoints/wav2lip_gan.pth)
- `--outfile`: Path to save the output video (default: wav2lip/results/result_voice.mp4)
- `--slow_mode`: Use slow mode for better quality (full face animation)
- `--resize_factor`: Reduce resolution for faster processing (default: 1)
- `--fps`: Set output frames per second (default: same as input)
- `--no_smooth`: Disable face detection smoothing

### Video Trimming:
The CLI also supports automatic video trimming:

```bash
./run_lipsync.py --face input.mp4 --audio speech.wav --trim --trim_start 1.5 --trim_end 10.0
```

- `--trim`: Enable video trimming
- `--trim_start`: Start time in seconds (default: 0)
- `--trim_end`: End time in seconds (default: end of video)
- `--trim_output`: Path for trimmed output (default: original_trimmed.mp4)

This command-line interface is particularly useful for batch processing or integration with other tools and scripts.

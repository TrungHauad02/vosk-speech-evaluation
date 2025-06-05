import os
import wave
import librosa
import soundfile as sf
import tempfile
import shutil
import time
import uuid
from datetime import datetime

def convert_to_wav(input_path, output_path):
    """
    Convert any audio file to WAV format using librosa
    """
    try:
        # Load the audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Save as WAV
        sf.write(output_path, y, sr, format='WAV', subtype='PCM_16')
        return True
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return False

def prepare_audio_file(audio_path):
    """
    Check if file is WAV or needs conversion, and ensure it has the correct format
    """
    # Check if file is WAV or needs conversion
    file_ext = os.path.splitext(audio_path)[1].lower()
    temp_files = []
    
    if file_ext != '.wav':
        # Convert to WAV if not already in WAV format
        temp_wav_path = audio_path + ".wav"
        success = convert_to_wav(audio_path, temp_wav_path)
        if not success:
            return None, temp_files, "Failed to convert audio file to WAV format"
        audio_path = temp_wav_path
        temp_files.append(temp_wav_path)
    
    try:
        wf = wave.open(audio_path, "rb")
    except Exception as e:
        return None, temp_files, f"Cannot open audio file: {str(e)}"
    
    # Check audio format
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        # Try to convert to mono PCM if needed
        try:
            temp_mono_path = audio_path + ".mono.wav"
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            sf.write(temp_mono_path, y, sr, format='WAV', subtype='PCM_16')
            audio_path = temp_mono_path
            temp_files.append(temp_mono_path)
            wf = wave.open(audio_path, "rb")
        except Exception as e:
            return None, temp_files, f"Failed to convert audio to required format: {str(e)}"
    
    return wf, temp_files, None

def process_audio_input(audio_input):
    """
    Process audio input (either recorded or uploaded) and save to a temporary file
    """
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_path = None
    
    try:
        # Generate a unique filename to avoid collisions and special character issues
        unique_filename = f"audio_{uuid.uuid4().hex}"
        
        # Determine input type and process accordingly
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            # This is numpy array input (recorded audio)
            sample_rate, audio_data = audio_input
            temp_path = os.path.join(temp_dir, f"{unique_filename}.wav")
            sf.write(temp_path, audio_data, sample_rate)
        else:
            # This is a file upload
            if hasattr(audio_input, 'name'):
                # Try to preserve file extension if possible
                original_ext = os.path.splitext(getattr(audio_input, 'filename', ''))[1]
                if not original_ext:
                    original_ext = '.tmp'
                temp_path = os.path.join(temp_dir, f"{unique_filename}{original_ext}")
                
                # Gradio file upload object
                with open(temp_path, "wb") as f:
                    f.write(audio_input.read())
            else:
                # Direct file path or bytes
                temp_path = os.path.join(temp_dir, f"{unique_filename}.tmp")
                
                if isinstance(audio_input, str) and os.path.isfile(audio_input):
                    # It's a file path
                    shutil.copy(audio_input, temp_path)
                else:
                    # Assume it's bytes
                    with open(temp_path, "wb") as f:
                        f.write(audio_input)
        
        return temp_dir, temp_path
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

def cleanup_temp_files(temp_dir, files_list=None):
    """
    Clean up temporary files and directory
    """
    if files_list:
        for file_path in files_list:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to remove temp file: {e}")
    
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Failed to remove temp directory: {e}")
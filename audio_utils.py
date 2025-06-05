import os
import wave
import librosa
import soundfile as sf
import tempfile
import shutil
import time
import uuid
from datetime import datetime
import numpy as np

def convert_to_wav(input_path, output_path):
    """
    Convert any audio file to WAV format using librosa
    Optimized for speech recognition with Vosk
    """
    try:
        print(f"Converting {input_path} to WAV format...")
        
        # Load audio with librosa (handles many formats)
        # sr=None preserves original sample rate
        y, sr = librosa.load(input_path, sr=None, mono=True)
        
        print(f"Loaded audio - Sample rate: {sr}Hz, Duration: {len(y)/sr:.2f}s")
        
        # Check if audio is silent or too quiet
        max_amplitude = np.max(np.abs(y))
        if max_amplitude < 0.001:
            print("Warning: Audio is very quiet, amplifying...")
            y = y * 10.0
            max_amplitude = np.max(np.abs(y))
        
        # Normalize to prevent clipping but maintain good volume
        if max_amplitude > 0:
            # Normalize to 90% to avoid clipping
            y = y / max_amplitude * 0.9
        
        # Vosk works best with 16kHz for English models
        # But the lgraph model can handle higher rates
        if sr > 48000:
            print(f"Sample rate {sr}Hz is very high, resampling to 16kHz...")
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Apply noise reduction if needed
        # Simple high-pass filter to remove low-frequency noise
        if sr >= 16000:
            y = librosa.effects.preemphasis(y)
        
        # Convert to 16-bit PCM
        y_16bit = np.int16(y * 32767)
        
        # Write WAV file
        sf.write(output_path, y_16bit, sr, format='WAV', subtype='PCM_16')
        
        # Verify the output
        with wave.open(output_path, 'rb') as wf:
            print(f"Output WAV - Channels: {wf.getnchannels()}, "
                  f"Sample rate: {wf.getframerate()}Hz, "
                  f"Duration: {wf.getnframes()/wf.getframerate():.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def prepare_audio_file(audio_path):
    """
    Check if file is WAV or needs conversion, and ensure it has the correct format
    """
    # Check if file exists and is not empty
    if not os.path.exists(audio_path):
        return None, [], f"Audio file not found: {audio_path}"
    
    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        return None, [], "Audio file is empty"
    
    print(f"Processing audio file: {audio_path} ({file_size} bytes)")
    
    # Check if file is WAV or needs conversion
    file_ext = os.path.splitext(audio_path)[1].lower()
    temp_files = []
    
    # Always try to open as WAV first
    try:
        wf = wave.open(audio_path, "rb")
        
        # Check if format is suitable for Vosk
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        comp_type = wf.getcomptype()
        framerate = wf.getframerate()
        
        print(f"WAV info - Channels: {channels}, Sample width: {sample_width}, "
              f"Compression: {comp_type}, Sample rate: {framerate}")
        
        # Vosk requires: mono, 16-bit PCM, no compression
        if channels == 1 and sample_width == 2 and comp_type == "NONE":
            # Format is already correct
            return wf, temp_files, None
        else:
            print("WAV format needs conversion for Vosk")
            wf.close()
            
    except Exception as e:
        print(f"Not a valid WAV file or needs conversion: {e}")
    
    # Convert to proper WAV format
    temp_wav_path = audio_path + ".converted.wav"
    success = convert_to_wav(audio_path, temp_wav_path)
    
    if not success:
        return None, temp_files, "Failed to convert audio file to WAV format"
    
    temp_files.append(temp_wav_path)
    
    # Try to open the converted file
    try:
        wf = wave.open(temp_wav_path, "rb")
        return wf, temp_files, None
    except Exception as e:
        return None, temp_files, f"Cannot open converted audio file: {str(e)}"

def process_audio_input(audio_input):
    """
    Process audio input (either recorded or uploaded) and save to a temporary file
    """
    # If it's already a file path, just return it
    if isinstance(audio_input, str) and os.path.isfile(audio_input):
        print(f"Using existing file: {audio_input}")
        return os.path.dirname(audio_input), audio_input
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_path = None
    
    try:
        # Generate a unique filename to avoid collisions
        unique_filename = f"audio_{uuid.uuid4().hex}_{int(time.time())}"
        
        # Determine input type and process accordingly
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            # This is numpy array input (recorded audio)
            sample_rate, audio_data = audio_input
            temp_path = os.path.join(temp_dir, f"{unique_filename}.wav")
            
            # Ensure audio data is in the right format
            if audio_data.dtype != np.int16:
                audio_data = np.int16(audio_data)
            
            sf.write(temp_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            
        elif hasattr(audio_input, 'read'):
            # It's a file-like object
            temp_path = os.path.join(temp_dir, f"{unique_filename}.tmp")
            with open(temp_path, "wb") as f:
                content = audio_input.read()
                f.write(content)
                
        elif isinstance(audio_input, bytes):
            # It's raw bytes
            temp_path = os.path.join(temp_dir, f"{unique_filename}.tmp")
            with open(temp_path, "wb") as f:
                f.write(audio_input)
                
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        print(f"Created temporary file: {temp_path}")
        return temp_dir, temp_path
        
    except Exception as e:
        # Clean up in case of error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def cleanup_temp_files(temp_dir, files_list=None):
    """
    Clean up temporary files and directory
    """
    if files_list:
        for file_path in files_list:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to remove temp file {file_path}: {e}")
    
    # Only remove temp directory if it's actually a temp directory
    if temp_dir and os.path.exists(temp_dir) and tempfile.gettempdir() in temp_dir:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Failed to remove temp directory {temp_dir}: {e}")
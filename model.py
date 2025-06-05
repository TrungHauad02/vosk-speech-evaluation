import os
import requests
import zipfile
from vosk import Model, KaldiRecognizer, SetLogLevel

# Reduce Vosk verbosity
SetLogLevel(-1)

# Global model instance
_model = None

def download_and_initialize_model():
    """
    Download and initialize the Vosk model for speech recognition
    """
    global _model
    
    # Return cached model if already loaded
    if _model is not None:
        return _model
    
    MODEL_PATH = "vosk-model-small-en-us-0.15"
    
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            r = requests.get("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", stream=True)
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open("model.zip", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Download progress: {percent:.1f}%", end='\r')
            
            print("\nExtracting model...")
            with zipfile.ZipFile("model.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove("model.zip")
            print("Model ready")
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    
    # Initialize and cache model
    print("Loading model...")
    _model = Model(MODEL_PATH)
    print("Model loaded successfully")
    
    return _model

def get_model():
    """Get the initialized model (with caching)"""
    global _model
    if _model is None:
        _model = download_and_initialize_model()
    return _model
import os
import requests
import zipfile
from vosk import Model, KaldiRecognizer

def download_and_initialize_model():
    """
    Download and initialize the Vosk model for speech recognition
    """
    MODEL_PATH = "vosk-model-small-en-us-0.15"
    
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", stream=True)
        with open("model.zip", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        print("Extracting model...")
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove("model.zip")
        print("Model ready")
    
    # Initialize and return model
    model = Model(MODEL_PATH)
    print("Model loaded successfully")
    return model
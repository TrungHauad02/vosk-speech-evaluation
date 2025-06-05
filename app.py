from fastapi import FastAPI, File, UploadFile, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
import os
from api import api_evaluate, api_evaluate_pronunciation_only, api_evaluate_multiple

# Create a FastAPI instance
app = FastAPI(title="Speech Evaluation API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an APIRouter for the API endpoints
router = APIRouter(prefix="/api", tags=["speech"])

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    topic: Optional[str] = Form("")
):
    """
    Evaluate speech from audio file with optional topic.
    
    Parameters:
    - file: The audio file (WAV or MP3)
    - topic: Optional expected text or topic for evaluation
    
    Returns:
    - JSON with evaluation results
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the audio file using the existing API function
        result = api_evaluate(temp_file_path, topic)
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return result
    
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {
            "error": str(e),
            "score": "N/A",
            "transcript": "",
            "strengths": [],
            "areas_to_improve": [],
            "feedback": f"Error processing audio: {str(e)}",
            "timestamp": None
        }

@router.post("/predict_pronunciation")
async def predict_pronunciation(
    file: UploadFile = File(...)
):
    """
    Evaluate pronunciation quality without topic comparison.
    
    Parameters:
    - file: The audio file (WAV or MP3)
    
    Returns:
    - JSON with pronunciation evaluation results
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the audio file for pronunciation only
        result = api_evaluate_pronunciation_only(temp_file_path)
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return result
    
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {
            "error": str(e),
            "score": "N/A",
            "transcript": "",
            "strengths": [],
            "areas_to_improve": [],
            "feedback": f"Error processing audio: {str(e)}",
            "timestamp": None
        }

@router.post("/predict_multiple")
async def predict_multiple(
    files: List[UploadFile] = File(...),
    topics: str = Form("")
):
    """
    Evaluate multiple audio files with corresponding topics.
    
    Parameters:
    - files: List of audio files (WAV or MP3)
    - topics: String with topics separated by newlines (one per file)
    
    Returns:
    - JSON with consolidated evaluation results
    """
    try:
        # Split topics by newline
        topic_list = [t.strip() for t in topics.split('\n') if t.strip()]
        
        # Save the uploaded files temporarily
        temp_file_paths = []
        for file in files:
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
            temp_file_paths.append(temp_path)
        
        # Process the multiple audio files
        result = api_evaluate_multiple(temp_file_paths, topic_list)
        
        # Clean up the temporary files
        for path in temp_file_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return result
    
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_paths' in locals():
            for path in temp_file_paths:
                if os.path.exists(path):
                    os.remove(path)
        
        return {
            "error": str(e),
            "score": "N/A",
            "transcripts": [],
            "strengths": [],
            "areas_to_improve": [],
            "feedback": f"Error processing audio files: {str(e)}",
            "timestamp": None
        }

# Include the router in the FastAPI app
app.include_router(router)

# Add a root endpoint for basic health check
@app.get("/")
async def root():
    return {"message": "Speech Evaluation API is running", "status": "ok"}

# Run the FastAPI app when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
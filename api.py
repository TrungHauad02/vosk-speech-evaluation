import os
import numpy as np
import soundfile as sf
import json
import time
from datetime import datetime
import wave
from vosk import Model, KaldiRecognizer

from speech_evaluation import evaluate_speech
from feedback_generation import (
    generate_feedback, 
    generate_detailed_feedback,
    generate_pronunciation_feedback
)
from audio_utils import (
    prepare_audio_file, 
    process_audio_input, 
    cleanup_temp_files
)

def process_audio(audio_input, topic=""):
    """
    Process recorded audio and return evaluation results
    """
    try:
        # Process the audio input
        temp_dir, temp_path = process_audio_input(audio_input)
        
        # Open the audio file
        wf, temp_files, error = prepare_audio_file(temp_path)
        if error:
            cleanup_temp_files(temp_dir, temp_files)
            return f"Error: {error}", "", [], [], "Error processing audio"
        
        # Get global model
        from model import download_and_initialize_model
        model = download_and_initialize_model()
        
        # Evaluate speech
        results = evaluate_speech(wf, model, topic)
        wf.close()
        
        # Generate feedback using AI model
        strengths, areas = generate_feedback(
            results["word_details"],
            results["transcript"],
            results["expected_text"],
            float(results["overall_confidence"]),
            float(results["relevance"]) if results["relevance"] is not None else 0.5,
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        detailed_feedback = generate_detailed_feedback(
            results["transcript"],
            results["expected_text"],
            float(results["score"]) / 10,  # Convert back to 0-1 scale
            float(results["overall_confidence"]),
            float(results["relevance"]) if results["relevance"] is not None else 0.5,
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        # Clean up temporary files
        cleanup_temp_files(temp_dir, temp_files)
        
        return results["score"], results["transcript"], strengths, areas, detailed_feedback
    
    except Exception as e:
        return f"Error: {str(e)}", "", [], [], "Error processing audio"

def process_uploaded_file(file_input, topic=""):
    """
    Process uploaded audio file and return evaluation results
    """
    try:
        # Process the file
        temp_dir, temp_path = process_audio_input(file_input)
        
        # Open the audio file
        wf, temp_files, error = prepare_audio_file(temp_path)
        if error:
            cleanup_temp_files(temp_dir, temp_files)
            return f"Error: {error}", "", [], [], "Error processing audio"
        
        # Get global model
        from model import download_and_initialize_model
        model = download_and_initialize_model()
        
        # Evaluate speech
        results = evaluate_speech(wf, model, topic)
        wf.close()
        
        # Generate feedback using AI model
        strengths, areas = generate_feedback(
            results["word_details"],
            results["transcript"],
            results["expected_text"],
            float(results["overall_confidence"]),
            float(results["relevance"]) if results["relevance"] is not None else 0.5,
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        detailed_feedback = generate_detailed_feedback(
            results["transcript"],
            results["expected_text"],
            float(results["score"]) / 10,  # Convert back to 0-1 scale
            float(results["overall_confidence"]),
            float(results["relevance"]) if results["relevance"] is not None else 0.5,
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        # Clean up temporary files
        cleanup_temp_files(temp_dir, temp_files)
        
        return results["score"], results["transcript"], strengths, areas, detailed_feedback
    
    except Exception as e:
        return f"Error: {str(e)}", "", [], [], "Error processing audio"

def api_evaluate(audio_file, topic=""):
    """
    API endpoint for evaluating speech from audio file
    """
    try:
        # Process the audio file
        temp_dir, temp_path = process_audio_input(audio_file)
        
        # Open the audio file
        wf, temp_files, error = prepare_audio_file(temp_path)
        if error:
            cleanup_temp_files(temp_dir, temp_files)
            return {
                "error": error,
                "score": "N/A",
                "transcript": "",
                "strengths": [],
                "areas_to_improve": [],
                "feedback": "Error processing audio",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get global model
        from model import download_and_initialize_model
        model = download_and_initialize_model()
        
        # Evaluate speech
        results = evaluate_speech(wf, model, topic)
        wf.close()
        
        # Generate feedback using AI model
        strengths, areas = generate_feedback(
            results["word_details"],
            results["transcript"],
            results["expected_text"],
            float(results["overall_confidence"]),
            float(results["relevance"]) if results["relevance"] is not None else 0.5,
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        detailed_feedback = generate_detailed_feedback(
            results["transcript"],
            results["expected_text"],
            float(results["score"]) / 10,  # Convert back to 0-1 scale
            float(results["overall_confidence"]),
            float(results["relevance"]) if results["relevance"] is not None else 0.5,
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        # Clean up temporary files
        cleanup_temp_files(temp_dir, temp_files)
        
        # Return JSON response
        return {
            "score": results["score"],
            "transcript": results["transcript"],
            "strengths": strengths,
            "areas_to_improve": areas,
            "feedback": detailed_feedback,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "score": "N/A",
            "transcript": "",
            "strengths": [],
            "areas_to_improve": [],
            "feedback": "Error processing audio",
            "timestamp": datetime.now().isoformat()
        }

def api_evaluate_pronunciation_only(audio_file):
    """
    API endpoint for evaluating pronunciation only (no topic)
    """
    try:
        # Process the audio file
        temp_dir, temp_path = process_audio_input(audio_file)
        
        # Open the audio file
        wf, temp_files, error = prepare_audio_file(temp_path)
        if error:
            cleanup_temp_files(temp_dir, temp_files)
            return {
                "error": error,
                "score": "N/A",
                "transcript": "",
                "strengths": [],
                "areas_to_improve": [],
                "feedback": "Error processing audio",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get global model
        from model import download_and_initialize_model
        model = download_and_initialize_model()
        
        # Evaluate speech (no topic)
        results = evaluate_speech(wf, model)
        wf.close()
        
        # Generate feedback specific to pronunciation using AI model
        strengths, areas = generate_pronunciation_feedback(
            results["word_details"],
            results["transcript"],
            float(results["overall_confidence"]),
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        detailed_feedback = generate_detailed_feedback(
            results["transcript"],
            "",  # Empty expected text
            float(results["score"]) / 10,
            float(results["overall_confidence"]),
            0.5,  # Neutral relevance score
            float(results["rhythm"]),
            float(results["intonation"]),
            float(results["speech_rate"])
        )
        
        # Clean up temporary files
        cleanup_temp_files(temp_dir, temp_files)
        
        # Return JSON response
        return {
            "score": results["score"],
            "transcript": results["transcript"],
            "strengths": strengths,
            "areas_to_improve": areas,
            "feedback": detailed_feedback,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "score": "N/A",
            "transcript": "",
            "strengths": [],
            "areas_to_improve": [],
            "feedback": "Error processing audio",
            "timestamp": datetime.now().isoformat()
        }

def api_evaluate_multiple(audio_files, expected_texts):
    """
    API endpoint for evaluating multiple audio files with their expected texts
    Returns consolidated feedback with individual transcripts
    """
    try:
        if len(audio_files) != len(expected_texts):
            return {
                "error": "Number of audio files must match number of expected texts",
                "score": "N/A",
                "transcripts": [],
                "strengths": [],
                "areas_to_improve": [],
                "feedback": "Error: Input mismatch",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get global model once for all evaluations
        from model import download_and_initialize_model
        model = download_and_initialize_model()
        
        # Process each file and collect results
        all_results = []
        all_transcripts = []
        all_word_details = []
        cumulative_score = 0
        temp_directories = []
        temp_files_list = []
        
        for i, (audio_file, expected_text) in enumerate(zip(audio_files, expected_texts)):
            try:
                # Process the audio file
                temp_dir, temp_path = process_audio_input(audio_file)
                temp_directories.append(temp_dir)
                
                # Open the audio file
                wf, temp_files, error = prepare_audio_file(temp_path)
                temp_files_list.extend(temp_files)
                
                if error:
                    all_transcripts.append(f"Error with file {i+1}: {error}")
                    continue
                
                # Evaluate speech
                results = evaluate_speech(wf, model, expected_text)
                wf.close()
                
                # Collect results
                all_results.append(results)
                all_transcripts.append(results["transcript"])
                all_word_details.extend(results["word_details"])
                cumulative_score += float(results["score"])
                
            except Exception as e:
                all_transcripts.append(f"Error processing file {i+1}: {str(e)}")
        
        # Clean up all temporary files
        for temp_dir in temp_directories:
            cleanup_temp_files(temp_dir, [])
        for temp_file in temp_files_list:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        # If no valid results, return error
        if not all_results:
            return {
                "error": "No audio files could be processed",
                "score": "N/A",
                "transcripts": all_transcripts,
                "strengths": [],
                "areas_to_improve": [],
                "feedback": "Error processing audio files",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate average scores
        avg_score = cumulative_score / len(all_results)
        avg_pronunciation = sum(float(r["overall_confidence"]) for r in all_results) / len(all_results)
        avg_relevance = sum(float(r["relevance"]) for r in all_results if r["relevance"] is not None) / len(all_results)
        avg_rhythm = sum(float(r["rhythm"]) for r in all_results) / len(all_results)
        avg_intonation = sum(float(r["intonation"]) for r in all_results) / len(all_results)
        avg_speech_rate = sum(float(r["speech_rate"]) for r in all_results) / len(all_results)
        
        # Generate consolidated feedback using AI model
        consolidated_transcript = " ".join(all_transcripts)
        consolidated_expected_text = " ".join(expected_texts)
        
        strengths, areas = generate_feedback(
            all_word_details,
            consolidated_transcript,
            consolidated_expected_text,
            avg_pronunciation,
            avg_relevance,
            avg_rhythm,
            avg_intonation,
            avg_speech_rate
        )
        
        detailed_feedback = generate_detailed_feedback(
            consolidated_transcript,
            consolidated_expected_text,
            avg_score / 10,
            avg_pronunciation,
            avg_relevance,
            avg_rhythm,
            avg_intonation,
            avg_speech_rate
        )
        
        # Return JSON response with consolidated results and individual transcripts
        return {
            "score": f"{avg_score:.1f}",
            "transcripts": all_transcripts,
            "strengths": strengths,
            "areas_to_improve": areas,
            "feedback": detailed_feedback,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "score": "N/A",
            "transcripts": [],
            "strengths": [],
            "areas_to_improve": [],
            "feedback": "Error processing audio files",
            "timestamp": datetime.now().isoformat()
        }
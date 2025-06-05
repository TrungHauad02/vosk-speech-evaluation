import os
import json
import re
from openai import OpenAI

# Get API key from environment variable - you'll need to set this
OPENROUTER_API_KEY = "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"
SITE_URL = os.environ.get("SITE_URL", "https://speech-evaluation-app.com")
SITE_NAME = os.environ.get("SITE_NAME", "Speech Evaluation App")

# Initialize the OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def extract_json_from_text(text):
    """
    Extract JSON object from text that might contain additional content
    """
    # Try to find JSON content between curly braces
    json_pattern = r'(\{[\s\S]*\})'
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1)
        try:
            # Try to parse the extracted JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If still not valid JSON, use more aggressive extraction
            # Look for content between the first { and the last }
            if '{' in text and '}' in text:
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                json_str = text[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None

def generate_ai_feedback(word_scores, transcript, expected_text, pronunciation_score, relevance_score, 
                     rhythm_score, intonation_score, speech_rate_score):
    """
    Generate AI-powered feedback using DeepSeek model via OpenRouter
    Returns strengths, area_to_improve, and detailed_feedback
    """
    # Prepare the data to send to the model
    if not expected_text:
        expected_text = "(No specific topic provided - pronunciation-only evaluation)"
    
    # Create word-level details
    word_details = ""
    if word_scores:
        problem_words = [w["word"] for w in word_scores if w["confidence"] < 0.4]
        good_words = [w["word"] for w in word_scores if w["confidence"] > 0.9 and len(w["word"]) > 3]
        
        if problem_words:
            word_details += f"Words with pronunciation issues: {', '.join(problem_words[:5])}\n"
        if good_words:
            word_details += f"Well-pronounced words: {', '.join(good_words[:5])}\n"
    
    # Calculate fluency metrics if possible
    fluency_info = ""
    if len(word_scores) > 10:
        avg_time_between_words = sum((word_scores[i+1]["start"] - word_scores[i]["end"]) 
                                    for i in range(len(word_scores) - 1)) / (len(word_scores) - 1)
        fluency_info = f"Average time between words: {avg_time_between_words:.2f} seconds\n"
    
    # Build prompt for the AI model with explicit JSON format instructions
    prompt = f"""
You are an expert English pronunciation coach. Based on the following speech analysis data, provide an assessment of the speaker's strengths and areas for improvement. You must return your analysis ONLY in valid JSON format with no additional text before or after the JSON.

SPEECH ANALYSIS DATA:
- Transcript: "{transcript}"
- Expected Content: "{expected_text}"
- Pronunciation Score: {pronunciation_score:.2f}/1.0
- Content Relevance Score: {relevance_score:.2f}/1.0
- Speech Rhythm Score: {rhythm_score:.2f}/1.0
- Intonation Score: {intonation_score:.2f}/1.0
- Speaking Rate Score: {speech_rate_score:.2f}/1.0
- Word Count: {len(transcript.split())}
{word_details}
{fluency_info}

Your response must be ONLY a JSON object with this exact structure:
{{
  "strengths": [
    "strength 1",
    "strength 2",
    "strength 3"
  ],
  "area_to_improve": [
    "area 1",
    "area 2",
    "area 3"
  ],
  "detailed_feedback": "Detailed paragraph of feedback with specific observations and recommendations."
}}

Do not include any text, explanations, or markdown outside of this JSON structure. The response should start with '{{' and end with '}}' with no additional characters.
Ensure each array has at least one item. Keep strengths and area_to_improve concise (one sentence each). The detailed_feedback should be 2-3 paragraphs at most.
"""

    try:
        # Call the OpenRouter API with the DeepSeek model
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert speech evaluation assistant. You provide feedback in JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=800,   # Limit the response length
            response_format={"type": "json_object"}  # Request JSON response format if supported
        )
        
        # Extract the response
        ai_response = completion.choices[0].message.content
        
        # Log the raw response for debugging (you can remove this in production)
        print(f"Raw AI response: {ai_response[:100]}...")  # Show just the first 100 chars
        
        # Try to parse the JSON response directly
        try:
            # First attempt: direct parsing
            feedback_data = json.loads(ai_response)
            
        except json.JSONDecodeError:
            print("Error with direct JSON parsing. Attempting to extract JSON from text.")
            # Second attempt: extract JSON from the response text
            feedback_data = extract_json_from_text(ai_response)
            
            if not feedback_data:
                raise ValueError("Failed to extract valid JSON from AI response")
        
        # Validate that the expected keys exist
        if not all(key in feedback_data for key in ["strengths", "area_to_improve", "detailed_feedback"]):
            missing_keys = [key for key in ["strengths", "area_to_improve", "detailed_feedback"] if key not in feedback_data]
            print(f"Missing required keys in AI response: {missing_keys}")
            raise ValueError(f"Missing required keys in AI response: {missing_keys}")
        
        # Validate types of each key
        if not isinstance(feedback_data["strengths"], list):
            feedback_data["strengths"] = [str(feedback_data["strengths"])]
        
        if not isinstance(feedback_data["area_to_improve"], list):
            feedback_data["area_to_improve"] = [str(feedback_data["area_to_improve"])]
        
        if not isinstance(feedback_data["detailed_feedback"], str):
            feedback_data["detailed_feedback"] = str(feedback_data["detailed_feedback"])
        
        # Ensure we have at least one item in each list
        if not feedback_data["strengths"]:
            feedback_data["strengths"] = ["Good attempt at communication"]
        
        if not feedback_data["area_to_improve"]:
            feedback_data["area_to_improve"] = ["Continue practicing to improve your speaking skills"]
        
        return feedback_data["strengths"], feedback_data["area_to_improve"], feedback_data["detailed_feedback"]
            
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in generate_ai_feedback: {str(e)}")
        
        # Return default feedback when all else fails
        return ["Communication attempt noted"], ["Continue practicing English pronunciation"], "The speech was analyzed, but detailed feedback could not be generated due to a technical issue. Please try again later."

# Public interface functions that use the AI feedback generator

def generate_feedback(word_scores, transcript, expected_text, pronunciation_score, relevance_score, 
                     rhythm_score, intonation_score, speech_rate_score):
    """
    Generate a list of strengths and areas to improve using AI model
    Returns arrays of strings for strengths and areas to improve
    """
    strengths, areas, _ = generate_ai_feedback(
        word_scores, transcript, expected_text, pronunciation_score, relevance_score,
        rhythm_score, intonation_score, speech_rate_score
    )
    return strengths, areas

def generate_detailed_feedback(transcript, expected_text, overall_score, pronunciation_score, 
                              relevance_score, rhythm_score, intonation_score, speech_rate_score):
    """
    Generate detailed feedback using AI model
    """
    # Get the detailed feedback from the AI model
    # For this function, we don't have word_scores, so we'll pass an empty list
    _, _, detailed_feedback = generate_ai_feedback(
        [], transcript, expected_text, pronunciation_score, relevance_score,
        rhythm_score, intonation_score, speech_rate_score
    )
    return detailed_feedback

def generate_pronunciation_feedback(word_scores, transcript, pronunciation_score, 
                     rhythm_score, intonation_score, speech_rate_score):
    """
    Generate a list of strengths and areas to improve focusing only on pronunciation using AI model
    Returns arrays of strings for strengths and areas to improve
    """
    # For pronunciation-only feedback, we set relevance_score to a neutral value
    # and leave expected_text empty to indicate pronunciation-only evaluation
    strengths, areas, _ = generate_ai_feedback(
        word_scores, transcript, "", pronunciation_score, 0.5,
        rhythm_score, intonation_score, speech_rate_score
    )
    return strengths, areas
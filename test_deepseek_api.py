import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
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

def test_deepseek_api():
    """
    Test function to verify OpenRouter API connection and JSON parsing
    """
    # Check if API key is available
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return False
    
    # Sample speech data for testing
    transcript = "hello my name is john and i like to play soccer"
    expected_text = "Hello, my name is John and I like to play soccer."
    pronunciation_score = 0.75
    relevance_score = 0.85
    rhythm_score = 0.70
    intonation_score = 0.65
    speech_rate_score = 0.80
    
    # Build test prompt
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
        print("Testing OpenRouter API connection...")
        
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
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}  # Request JSON response format if supported
        )
        
        # Extract the response
        ai_response = completion.choices[0].message.content
        
        print("\n--- Raw AI Response ---")
        print(ai_response)
        print("--- End of Raw Response ---\n")
        
        # Try direct JSON parsing
        print("Attempting direct JSON parsing...")
        try:
            feedback_data = json.loads(ai_response)
            print("✅ Direct JSON parsing successful!")
            
            # Validate the JSON structure
            if all(key in feedback_data for key in ["strengths", "area_to_improve", "detailed_feedback"]):
                print("✅ JSON contains all required keys")
                print("\n--- Parsed JSON Data ---")
                print(json.dumps(feedback_data, indent=2))
                print("--- End of Parsed Data ---\n")
                return True
            else:
                print("❌ JSON missing required keys")
                return False
            
        except json.JSONDecodeError as e:
            print(f"❌ Direct JSON parsing failed: {str(e)}")
            
            # Try extracting JSON from text
            print("Attempting to extract JSON from text...")
            feedback_data = extract_json_from_text(ai_response)
            
            if feedback_data:
                print("✅ JSON extraction successful!")
                print("\n--- Extracted JSON Data ---")
                print(json.dumps(feedback_data, indent=2))
                print("--- End of Extracted Data ---\n")
                return True
            else:
                print("❌ JSON extraction failed")
                return False
    
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_deepseek_api()
    
    if success:
        print("\n✅ TEST SUCCESSFUL: OpenRouter API connection and JSON parsing are working correctly")
    else:
        print("\n❌ TEST FAILED: There was an issue with the OpenRouter API or JSON parsing")
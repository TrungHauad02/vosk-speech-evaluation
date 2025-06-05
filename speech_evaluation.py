import json
import wave
from vosk import KaldiRecognizer

def evaluate_speech(wf, model, expected_text=""):
    """
    Evaluate pronunciation from audio file
    """
    # Initialize recognizer
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    
    # Process audio
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            if "result" in part_result:
                results.extend(part_result["result"])
    
    # Get final result
    final_result = json.loads(rec.FinalResult())
    if "result" in final_result:
        results.extend(final_result["result"])
    
    transcript = final_result.get("text", "")
    
    # Analyze results
    word_scores = []
    for word_info in results:
        word_scores.append({
            "word": word_info["word"],
            "confidence": word_info["conf"],
            "start": word_info["start"],
            "end": word_info["end"]
        })
    
    # Calculate average score of words
    if word_scores:
        avg_confidence = sum(w["confidence"] for w in word_scores) / len(word_scores)
    else:
        avg_confidence = 0.0
    
    # Evaluate relevance to the topic
    relevance_score = evaluate_relevance(transcript, expected_text) if expected_text else 0.5
    
    # Calculate rhythm and intonation scores if enough words
    rhythm_score = evaluate_rhythm(word_scores) if len(word_scores) > 3 else 0.5
    intonation_score = evaluate_intonation(word_scores) if len(word_scores) > 3 else 0.5
    
    # Additional speech quality metrics
    speech_rate_score = evaluate_speech_rate(word_scores, transcript) if word_scores else 0.5
    
    # Calculate total score with weighted factors
    if expected_text:
        # With topic comparison
        overall_score = (
            0.4 * avg_confidence +    # Pronunciation accuracy
            0.2 * relevance_score +   # Content relevance
            0.15 * rhythm_score +     # Speech rhythm
            0.15 * intonation_score + # Intonation patterns
            0.1 * speech_rate_score   # Speech rate
        )
    else:
        # Without topic comparison (pronunciation only)
        overall_score = (
            0.5 * avg_confidence +    # Pronunciation accuracy (increased weight)
            0.2 * rhythm_score +      # Speech rhythm
            0.2 * intonation_score +  # Intonation patterns
            0.1 * speech_rate_score   # Speech rate
        )
    
    return {
        "score": f"{overall_score * 10:.1f}",  # Convert to scale of 10
        "pronunciation_score": f"{avg_confidence * 10:.1f}",
        "relevance_score": f"{relevance_score * 10:.1f}" if expected_text else "N/A",
        "rhythm_score": f"{rhythm_score * 10:.1f}",
        "intonation_score": f"{intonation_score * 10:.1f}",
        "speech_rate_score": f"{speech_rate_score * 10:.1f}",
        "transcript": transcript,
        "expected_text": expected_text if expected_text else "",
        "word_details": word_scores,
        "overall_confidence": avg_confidence,
        "relevance": relevance_score if expected_text else None,
        "rhythm": rhythm_score,
        "intonation": intonation_score,
        "speech_rate": speech_rate_score
    }

def evaluate_relevance(transcript, expected_text):
    """
    Evaluate the relevance between transcript and expected text
    Using a more sophisticated algorithm based on word frequency and order
    """
    if not transcript or not expected_text:
        return 0.0
    
    # Convert to lowercase and split into words
    transcript_words = transcript.lower().split()
    expected_words = expected_text.lower().split()
    
    # Calculate common words (with order consideration)
    common_words = set(transcript_words).intersection(set(expected_words))
    
    # Calculate score based on the ratio of common words
    if not expected_words:
        return 0.0
    
    # Base score from word overlap
    base_score = len(common_words) / len(expected_words)
    
    # Bonus for word order similarity (simple implementation)
    order_bonus = 0.0
    if len(common_words) > 1:
        # Check if the order of words is preserved
        expected_order = {word: i for i, word in enumerate(expected_words) if word in common_words}
        transcript_order = {word: i for i, word in enumerate(transcript_words) if word in common_words}
        
        # Count pairs of words that appear in the same order in both texts
        correct_order_pairs = 0
        total_pairs = 0
        
        sorted_common_words = sorted(common_words)
        for i in range(len(sorted_common_words)):
            for j in range(i+1, len(sorted_common_words)):
                word1, word2 = sorted_common_words[i], sorted_common_words[j]
                total_pairs += 1
                if (expected_order[word1] < expected_order[word2]) == (transcript_order[word1] < transcript_order[word2]):
                    correct_order_pairs += 1
        
        if total_pairs > 0:
            order_bonus = 0.2 * (correct_order_pairs / total_pairs)
    
    # Return combined score (capped at 1.0)
    return min(1.0, base_score + order_bonus)

def evaluate_rhythm(word_scores):
    """
    Evaluate rhythm based on timing between words
    """
    if len(word_scores) < 4:
        return 0.5  # Not enough data for meaningful rhythm analysis
    
    # Calculate time between words
    gaps = []
    for i in range(len(word_scores) - 1):
        gaps.append(word_scores[i+1]["start"] - word_scores[i]["end"])
    
    # Check for consistency in timing (standard deviation of gaps)
    if not gaps:
        return 0.5
    
    avg_gap = sum(gaps) / len(gaps)
    variance = sum((gap - avg_gap) ** 2 for gap in gaps) / len(gaps)
    std_dev = variance ** 0.5
    
    # Calculate rhythm score - lower standard deviation means more consistent rhythm
    # Normalize to [0-1] range
    consistency_score = max(0, min(1, 1 - (std_dev / avg_gap) if avg_gap > 0 else 0))
    
    # Check if there are appropriate pauses at syntactic boundaries
    # This is a simplified approximation
    pause_score = 0.5
    if avg_gap > 0.1 and avg_gap < 0.8:
        pause_score = 0.8  # Reasonable average pause duration
    
    return (consistency_score * 0.7) + (pause_score * 0.3)

def evaluate_intonation(word_scores):
    """
    Evaluate intonation based on confidence patterns
    Note: This is an approximation since we don't have actual pitch data
    """
    if len(word_scores) < 4:
        return 0.5  # Not enough data
    
    # Use confidence scores as a proxy for emphasis
    # In a better implementation, we would use actual pitch data
    confidences = [w["confidence"] for w in word_scores]
    
    # Calculate variance in confidence (as a proxy for intonation variation)
    avg_conf = sum(confidences) / len(confidences)
    variance = sum((conf - avg_conf) ** 2 for conf in confidences) / len(confidences)
    
    # Some variation is good, but too much or too little is not ideal
    # This is a simplified model
    if variance < 0.01:
        # Too monotone
        return 0.4
    elif variance > 0.2:
        # Too erratic
        return 0.6
    else:
        # Good variation
        return 0.8

def evaluate_speech_rate(word_scores, transcript):
    """
    Evaluate speech rate (words per minute)
    """
    if not word_scores or len(word_scores) < 2:
        return 0.5  # Not enough data
    
    # Calculate total speech duration
    total_duration = word_scores[-1]["end"] - word_scores[0]["start"]
    if total_duration <= 0:
        return 0.5
    
    # Calculate words per minute
    words_count = len(transcript.split())
    wpm = (words_count / total_duration) * 60
    
    # Evaluate based on typical English speech rates
    # Optimal range: 120-160 WPM for clear speech
    if wpm < 80:
        return 0.4  # Too slow
    elif 80 <= wpm < 120:
        return 0.7  # Slightly slow but acceptable
    elif 120 <= wpm <= 160:
        return 0.9  # Optimal rate
    elif 160 < wpm <= 200:
        return 0.7  # Slightly fast but acceptable
    else:
        return 0.4  # Too fast
import time
import re
import requests
from flask import Flask, request, jsonify, send_from_directory
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_url_path='', static_folder='static')

# --------------- Config ---------------
MAX_RETRIES = 3
BASE_DELAY = 5  # seconds
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

# --------------- Gemini Helper ---------------
def gemini_call(client, model_id, contents):
    """Call Gemini API with retry + exponential backoff for 429 errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents
            )
            return response
        except Exception as e:
            error_str = str(e)
            is_rate_limit = '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                match = re.search(r'retry in ([\d.]+)s', error_str, re.IGNORECASE)
                delay = float(match.group(1)) if match else BASE_DELAY * (2 ** attempt)
                print(f"[Rate Limited] Attempt {attempt+1}/{MAX_RETRIES}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise

# --------------- Groq Helper ---------------
def groq_call(api_key, contents):
    """Call Groq API (OpenAI-compatible) as a backup provider."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": contents}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "tokens": data.get("usage", {}).get("total_tokens", 0)
    }

# --------------- Unified LLM Call ---------------
def llm_call(provider, api_key, contents, gemini_client=None):
    """
    Unified LLM call. Uses the specified provider.
    Returns dict: { text, tokens, provider_used }
    """
    if provider == "gemini":
        try:
            response = gemini_call(gemini_client, "gemini-2.0-flash", contents)
            tokens = response.usage_metadata.total_token_count if response.usage_metadata else len(response.text.split())
            return {"text": response.text, "tokens": tokens, "provider_used": "gemini"}
        except Exception as e:
            error_str = str(e)
            is_quota = '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str
            # Auto-fallback to Groq if quota exhausted and GROQ key is available
            groq_key = os.getenv("GROQ_API_KEY")
            if is_quota and groq_key:
                print("[Fallback] Gemini quota exhausted. Switching to Groq...")
                result = groq_call(groq_key, contents)
                result["provider_used"] = "groq (fallback)"
                return result
            raise
    elif provider == "groq":
        result = groq_call(api_key, contents)
        result["provider_used"] = "groq"
        return result
    else:
        raise ValueError(f"Unknown provider: {provider}")

# --------------- Feature Extraction ---------------
def calculate_features(text):
    """Extracts features from the prompt text based on our ML analysis."""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Feature 1: Token Length
    token_length = len(words)
    
    # Feature 2: Instruction Count
    instruction_verbs = ['always','never','do','don\'t','start','follow','include','end','keep','ensure','must','strictly','exactly']
    instruction_count = 0
    for line in text_lower.split('\n'):
        line = line.strip()
        if any(line.startswith(str(i)) for i in range(10)) or any(line.startswith(v) for v in instruction_verbs):
            instruction_count += 1
            
    # Feature 3: Specificity Score
    specificity_keywords = ["must", "only", "strictly", "require", "exactly", "specific", "detailed", "ensure", "always", "never"]
    num_constraints = sum(1 for word in words if word in specificity_keywords)
    specificity_score = (num_constraints / token_length * 100) if token_length > 0 else 0
    
    # Feature 4: Example Count
    example_count = text_lower.count("example") + text_lower.count("e.g.")
    
    return {
        "token_length": token_length,
        "instruction_count": instruction_count,
        "specificity_score": round(specificity_score, 2),
        "example_count": example_count
    }

def calculate_score(features, api_quality_score, latency):
    """Applies the custom algorithm to score a prompt."""
    base_score = (features['instruction_count'] * 0.5) + (features['specificity_score'] * 0.2) + (features['example_count'] * 1.5)
    penalty = (features['token_length'] * 0.005) + (latency * 0.2)
    total_score = base_score + api_quality_score - penalty
    return round(total_score, 2)

# --------------- Routes ---------------
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    data = request.json
    api_key = data.get('api_key', '').strip()
    prompt_a = data.get('prompt_a', '')
    prompt_b = data.get('prompt_b', '')
    provider = data.get('provider', 'gemini')  # 'gemini' or 'groq'
    test_query = "Explain what machine learning is"
    
    if not api_key:
        return jsonify({"error": "API Key is required"}), 400

    # Initialize clients based on provider
    gemini_client = None
    if provider == "gemini":
        try:
            gemini_client = genai.Client(api_key=api_key)
        except Exception as e:
            return jsonify({"error": f"Failed to initialize Gemini Client: {str(e)}"}), 400

    results = {}
    actual_provider = provider  # Track which provider was actually used
    
    for prompt_name, prompt_text in [("A", prompt_a), ("B", prompt_b)]:
        features = calculate_features(prompt_text)
        
        # Call LLM
        start = time.time()
        try:
            llm_result = llm_call(provider, api_key, prompt_text + "\n\n" + test_query, gemini_client)
            response_text = llm_result["text"]
            actual_tokens = llm_result["tokens"] or features['token_length']
            actual_provider = llm_result["provider_used"]
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                return jsonify({
                    "error": "⚠️ API quota exhausted. Please wait a few minutes and try again, switch to Groq, or upgrade your plan."
                }), 429
            response_text = f"Error: {error_str}"
            actual_tokens = features['token_length']
            
        latency = round(time.time() - start, 2)
        
        # Judge quality
        judge_prompt = f"Rate this response on a scale of 1 to 10 for accuracy and clarity. Reply with ONLY a number.\n\nResponse: {response_text}"
        try:
            judge_result = llm_call(provider, api_key, judge_prompt, gemini_client)
            api_quality_score = float(re.search(r'(\d+)', judge_result["text"]).group(1))
            api_quality_score = min(max(api_quality_score, 1), 10)  # Clamp 1-10
        except:
            api_quality_score = 5.0
            
        features['token_length'] = actual_tokens
        final_score = calculate_score(features, api_quality_score, latency)
        
        results[prompt_name] = {
            "text": prompt_text,
            "response": response_text,
            "latency": latency,
            "api_quality_score": api_quality_score,
            "features": features,
            "final_score": final_score
        }

        # Small delay between prompts to help with rate limits
        time.sleep(1)
        
    # Determine winner
    winner = "Prompt A" if results["A"]["final_score"] >= results["B"]["final_score"] else "Prompt B"
    results["winner"] = winner
    results["provider_used"] = actual_provider
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

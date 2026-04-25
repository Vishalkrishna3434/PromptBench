import time
from flask import Flask, request, jsonify, send_from_directory
from google import genai
import os

app = Flask(__name__, static_url_path='', static_folder='static')

def calculate_features(text):
    """Extracts features from the prompt text based on our ML analysis."""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Feature 1: Token Length
    token_length = len(words)
    
    # Feature 2: Instruction Count
    # Capture command verbs and numbered/bulleted lists
    instruction_verbs = ['always','never','do','don\'t','start','follow','include','end','keep','ensure','must','strictly','exactly']
    instruction_count = 0
    for line in text_lower.split('\n'):
        line = line.strip()
        # Count if line starts with a number (e.g., "1. ") or a command verb
        if any(line.startswith(str(i)) for i in range(10)) or any(line.startswith(v) for v in instruction_verbs):
            instruction_count += 1
            
    # Feature 3: Specificity Score
    # Broaden the list of constraint keywords
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
    # Base score derived from ML positive features
    base_score = (features['instruction_count'] * 0.5) + (features['specificity_score'] * 0.2) + (features['example_count'] * 1.5)
    
    # Penalty derived from ML negative features
    penalty = (features['token_length'] * 0.005) + (latency * 0.2)
    
    # Total Score
    total_score = base_score + api_quality_score - penalty
    return round(total_score, 2)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    data = request.json
    api_key = data.get('api_key')
    prompt_a = data.get('prompt_a', '')
    prompt_b = data.get('prompt_b', '')
    test_query = "Explain what machine learning is"
    
    if not api_key:
        return jsonify({"error": "API Key is required"}), 400
        
    # Initialize genai client
    try:
        client = genai.Client(api_key=api_key)
        model_id = 'gemini-1.5-flash'
    except Exception as e:
        return jsonify({"error": f"Failed to initialize Gemini Client: {str(e)}"}), 400

    results = {}
    
    for prompt_name, prompt_text in [("A", prompt_a), ("B", prompt_b)]:
        features = calculate_features(prompt_text)
        
        # Test Gemini
        start = time.time()
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt_text + "\n\n" + test_query
            )
            response_text = response.text
            actual_tokens = response.usage_metadata.total_token_count if response.usage_metadata else features['token_length']
        except Exception as e:
            response_text = f"Error: {str(e)}"
            actual_tokens = features['token_length']
            
        latency = round(time.time() - start, 2)
        
        # Judge quality using Gemini
        judge_prompt = f"Rate this response on a scale of 1 to 10 for accuracy and clarity. Reply with ONLY a number.\n\nResponse: {response_text}"
        try:
            judge_res = client.models.generate_content(model=model_id, contents=judge_prompt)
            api_quality_score = float(judge_res.text.strip())
        except:
            api_quality_score = 5.0 # Fallback
            
        # Override calculated tokens with real Gemini tokens
        features['token_length'] = actual_tokens
        
        # Calculate custom final score
        final_score = calculate_score(features, api_quality_score, latency)
        
        results[prompt_name] = {
            "text": prompt_text,
            "response": response_text,
            "latency": latency,
            "api_quality_score": api_quality_score,
            "features": features,
            "final_score": final_score
        }
        
    # Determine winner
    winner = "Prompt A" if results["A"]["final_score"] >= results["B"]["final_score"] else "Prompt B"
    results["winner"] = winner
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

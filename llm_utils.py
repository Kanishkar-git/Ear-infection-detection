import requests

GEMINI_API_KEY = "AIzaSyC6NktkmKRpJTMpe10HaLplzAqBXkt7Kgw"

def query_gemini(class_name, confidence):
    """
    Send detected object info to Gemini LLM and get reasons & precautions in English & Tamil
    """
    prompt = f"""
    You are a medical assistant. A patient has an ear issue detected as:
    - Class: {class_name}
    - Confidence: {confidence}

    Explain why this happens and give precautions point by point.
    Provide the answer in **English and Tamil**.
    """

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gemini-1.5",
        "prompt": prompt,
        "temperature": 0.7,
        "max_output_tokens": 400
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("output_text", "No response from LLM")
    else:
        return f"Error: {response.status_code} - {response.text}"





import requests
import os
import time

# Load API Key securely
API_KEY = os.getenv("OPENROUTER_API_KEY") 

URL = "https://openrouter.ai/api/v1/chat/completions" 

INTENT_CATEGORIES = [
    "Need immediate crisis support",  # Feeling suicidal, overwhelmed, or in extreme distress
    "Looking for professional help",  # Seeking a therapist, psychologist, or psychiatrist
    "Need emotional support",  # Feeling lonely, sad, or needing someone to talk to
    "Want therapy recommendations",  # Asking about therapy types, effectiveness, or techniques
    "Struggling with anxiety or panic attacks",  # Dealing with anxiety, fear, or sudden panic
    "Feeling depressed or hopeless",  # Experiencing sadness, numbness, or lack of motivation
    "Seeking ways to cope with loneliness",  # Feeling isolated and wanting social connection
    "Have questions about medication or psychiatry",  # Curious about antidepressants or psychiatric treatments
    "Struggling with grief or loss",  # Coping with death, separation, or emotional pain
    "Need support for trauma or PTSD",  # Trying to manage past trauma or distressing memories
    "Facing financial stress or money issues",  # Dealing with job loss, financial anxiety, or debt
    "Need help with parenting or family stress",  # Struggling with raising kids or family conflicts
    "Experiencing bullying or harassment",  # Facing bullying at school, work, or online
    "Seeking advice on domestic abuse or legal issues",  # Needing support for abuse or legal concerns
    "Struggling with self-esteem or identity",  # Issues with self-worth, body image, or personal identity
    "Need motivation or productivity tips",  # Struggling with focus, procrastination, or lack of drive
    "Exploring spirituality or life purpose",  # Seeking meaning, purpose, or personal beliefs
    "Looking for mindfulness or relaxation techniques",  # Interested in meditation, stress relief, or breathing exercises
    "Feeling stressed about work or studies",  # Dealing with burnout, exams, or workplace pressure
    "Need career or job loss advice",  # Facing layoffs, career changes, or job uncertainty
    "Experiencing workplace toxicity or discrimination",  # Facing unfair treatment, bullying, or bias at work
    "Struggling with social anxiety or communication",  # Having trouble making friends or expressing thoughts
    "Need advice on relationships or social issues",  # Facing difficulties in friendships, family, or romantic relationships
    "Other concerns that don’t fit above"  # Any topic that doesn’t fall under these categories
]


def intent_recognition(user_message):
    messages = [
        {"role": "system", "content": f"""
        You are an AI that classifies user messages into one of the following categories:

        {", ".join(INTENT_CATEGORIES)}

        Respond **only** with the category name. Do not provide explanations, extra words, or alternative responses.
        """},
        {"role": "user", "content": user_message}
    ]
    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "yourwebsite.com",
        "X-Title": "LLaMA Intent Recognition"
    }
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = requests.post(URL, json=payload, headers=headers)
            response.raise_for_status() 
            result = response.json()
            if "choices" in result and result["choices"]:
                detected_intent = result["choices"][0]["message"]["content"].strip()
                if detected_intent in INTENT_CATEGORIES:
                    return detected_intent 
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request failed (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(3)  # Wait before retrying
    return "Error: Unable to determine intent."





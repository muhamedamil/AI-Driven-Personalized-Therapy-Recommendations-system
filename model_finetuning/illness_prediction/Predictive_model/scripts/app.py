# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import torch
import re
import streamlit as st
from openai import OpenAI

sys.path.append(os.path.abspath("C:/Users/amil/OneDrive/Documents/AI-Driven Personalized Therapy Recommendations system/Module_3/Predictive_model/scripts"))
from utils.model_loader import sentiment_tokenizer, sentiment_model, predective_model, predective_model_tokenizer
from utils.intent_recognition import intent_recognition

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mental Health and AI Response Mappings
mental_health_mapping = {
    0: "ADHD",
    1: "Anxiety",
    2: "BDD",
    3: "Bipolar",
    4: "BPD",
    5: "Depression",
    6: "Eating Disorder",
    7: "Hoarding Disorder",
    8: "Mental Illness",
    9: "Normal",
    10: "OCD",
    11: "Off My Chest",
    12: "Panic Disorder",
    13: "Personality Disorder",
    14: "PTSD",
    15: "Schizophrenia",
    16: "Social Anxiety",
    17: "Stress",
    18: "Suicidal"
}

ai_response_mapping = {
    0: "De-escalation & Validation",
    1: "Reframing & Encouragement",
    2: "Reassurance & Coping Strategies",
    3: "Encouragement & Positive Reinforcement",
    4: "Active Listening & Encouragement",
    5: "Compassion & Support",
    6: "Clarification & Stability"
}

# Text Cleaning Functions
contractions = {
    "don't": "do not",
    "can't": "cannot",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "there's": "there is",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "doesn't": "does not"
}

def expand_contractions(text):
    """ Expand contractions in the text """
    for contraction, expanded in contractions.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    return text

def clean_text(text):
    """ Preprocess the input text """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

def mental_health_pipeline(text):
    """ Perform intent recognition, sentiment analysis, and mental health prediction """
    intent = intent_recognition(text)

    inputs = sentiment_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment_idx = torch.argmax(sentiment_scores).item()
    sentiment_label = ai_response_mapping.get(sentiment_idx, "Unknown")

    # Mental Health Prediction
    pred_inputs = predective_model_tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    if "token_type_ids" in pred_inputs:
        del pred_inputs["token_type_ids"]

    predective_model.eval()
    with torch.no_grad():
        outputs = predective_model(**pred_inputs)

    logits = outputs["logits"]
    prediction_idx = torch.argmax(logits).item()
    prediction_label = mental_health_mapping.get(prediction_idx, "Unknown")

    return {
        "intent": intent,
        "sentiment": sentiment_label,
        "mental_health_prediction": prediction_label
    }


import streamlit as st
import requests
import json

API_KEY = "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" 
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_ai_response(intent, sentiment, mental_health_prediction, user_message):
    """ Generate a dynamic AI response using OpenRouter API """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": [
            {"role": "system", "content": (
                "You are a compassionate and empathetic friend, dedicated to offering emotional support. "
                "Your goal is to create a safe, non-judgmental space where the user feels heard and understood. "
                "Be warm, kind, and reassuring. Use gentle, encouraging language and validate their feelings. "
                "Offer comfort without being dismissive, and express genuine care. "
                "If appropriate, provide simple, practical suggestions to help them feel more at ease, "
                "while always prioritizing empathy over advice. "
                "Your tone should be friendly, patient, and soothing, making the user feel valued and supported."
            )},
            {"role": "user", "content": (
                f"Intent: {intent}\n"
                f"Sentiment: {sentiment}\n"
                f"Mental Health Prediction: {mental_health_prediction}\n"
                f"User Message: {user_message}"
            )}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit Interface
st.title("🧠 AI-Powered Mental Health Assistant")
st.markdown("### 🌿 Enter your thoughts or feelings below to get insights and receive compassionate support.")

# User Input Box
user_input = st.text_area("Enter your query here:", height=150)

if st.button("Analyze"):
    if user_input.strip():
        # Preprocess and run the pipeline
        preprocessed_text = clean_text(user_input)
        result = mental_health_pipeline(preprocessed_text)

        # Generate AI response
        ai_response = generate_ai_response(
            intent=result['intent'],
            sentiment=result['sentiment'],
            mental_health_prediction=result['mental_health_prediction'],
            user_message=user_input
        )

        # Display the result
        st.markdown("### 💡 **Analysis Result:**")
        st.write(f"**🛑 Intent:** {result['intent']}")
        st.write(f"**😊 Sentiment:** {result['sentiment']}")
        st.write(f"**🧠 Mental Health Prediction:** {result['mental_health_prediction']}")

        # Display AI's compassionate response
        st.markdown("### 💙 **AI's Supportive Response:**")
        st.write(ai_response)

    else:
        st.warning("⚠️ Please enter some text to analyze!")


st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

"""
Module: intent_recognition.py
Description:
    This module uses OpenRouter's language model to classify user input
    into predefined intent categories, particularly for mental health-related
    chat scenarios.

    The assistant responds with exactly one label from a structured intent taxonomy.

Created: 2025-07-08
"""

import os
import logging
import asyncio
from typing import List, Optional
from openai import OpenAI, OpenAIError


# -------------------------------
# INTENT CATEGORIES (Mental Health)
# -------------------------------
INTENT_CATEGORIES: List[str] = [

    # Crisis & Emergency
    "Need immediate crisis support",
    "Feeling suicidal or in danger",
    "Experiencing severe emotional breakdown",

    # Mental Health Conditions
    "Struggling with anxiety or panic attacks",
    "Feeling depressed or hopeless",
    "Struggling with emotional regulation",
    "Dealing with trauma or PTSD",
    "Experiencing obsessive or intrusive thoughts",
    "Struggling with self-esteem or identity",

    #  Social, Emotional & Relationship Support
    "Feeling lonely or isolated",
    "Having relationship difficulties",
    "Need emotional support",
    "Need advice on relationships or social issues",
    "Struggling with social anxiety or communication",
    "Dealing with bullying or harassment",
    "Facing discrimination or exclusion",

    # Family & Parenting
    "Need help with parenting or family stress",
    "Coping with family conflict or tension",

    #  Career, School & Financial
    "Feeling stressed about work or studies",
    "Need career guidance or job loss support",
    "Experiencing workplace toxicity or discrimination",
    "Facing financial stress or money issues",

    #  Personal Growth & Self-Help
    "Looking for mindfulness or relaxation techniques",
    "Exploring spirituality or life purpose",
    "Need motivation or productivity tips",
    "Want therapy or self-care recommendations",

    #  Medical, Psychiatric & Professional 
    "Have questions about medication or psychiatry",
    "Looking for professional help",
    "Need a diagnosis or clinical guidance",

    #  Grief & Loss
    "Struggling with grief or loss",

    # Casual, General & Check-ins
    "General inquiry or casual conversation",
    "Just checking in or saying hello",
    "No specific concern, just expressing myself",
    "Neutral or polite expression (e.g., 'okay', 'fine', 'thank you')",
    "Want to continue chatting but no emotional distress",
    "No issue, just small talk or closure",
    "Non-emotional affirmation or agreement",
    "Simple feedback or acknowledgment",
    "No mental health-related concern detected",
    "Unclear intent or ambiguous statement",

    # Miscellaneous
    "Other concerns that don’t fit above",
]

# -------------------------------
# Logging Configuration
# -------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------
# OpenRouter Client Configuration
# -------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# -------------------------------
# Intent Recognition Logic
# -------------------------------

async def intent_recognition(user_input: str, retries: int = 3) -> Optional[str]:
    """
    Detects the user’s intent category using OpenRouter LLM.

    Args:
        user_input (str): The user’s message to classify.
        retries (int): Number of retry attempts on failure (default is 3).

    Returns:
        Optional[str]: A matching category from INTENT_CATEGORIES or None if all retries fail.
    """
    system_prompt = (
        "You are a strict intent classification assistant.\n\n"
        "Your task is to read the user's message and classify it into exactly **one** of the following intent categories:\n"
        f"{', '.join(INTENT_CATEGORIES)}\n\n"
        " IMPORTANT RULES:\n"
        "1. You must respond with **only one** category name from the list.\n"
        "2. Do not add any explanations, punctuation, or extra words.\n"
        "3. If the intent is unclear or overlaps, choose the **closest matching** category.\n"
        "4. Do not say 'The user intent is...' or include quotation marks.\n\n"
        "Respond with only the exact category name."
    )

    for attempt in range(retries):
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct:free",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ]
                )
            )

            intent = response.choices[0].message.content.strip()
            if intent:
                logger.info(f" Detected intent: {intent}")
                return intent

        except OpenAIError as e:
            logger.warning(f" OpenRouter API error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during intent detection on attempt {attempt + 1}: {e}")

    logger.error(" All retries failed. Could not detect intent.")
    return None

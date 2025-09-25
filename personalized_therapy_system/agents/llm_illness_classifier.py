"""
Module: llm_illness_classifier.py
Description: Uses a large language model (LLM) to determine whether a user's message indicates the need for mental illness detection.
Created: 2025-06-29
Last Modified: 2025-07-08
"""

import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()


# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

__all__ = ['LLMIntentBasedIllnessClassifier']

# ----------------------------------------
# Prompt Template: Mental Distress Detector
# ----------------------------------------
ILLNESS_CLASSIFIER_PROMPT = PromptTemplate.from_template("""
You are a classifier that determines if a message indicates *significant mental distress* that requires further analysis.

Only respond with **TRUE** or **FALSE**.

Respond **TRUE** only if the message includes any of the following:
- Clear expressions of deep emotional suffering (e.g. "I want to disappear", "I can't handle this anymore")
- Mentions of suicidal thoughts, hopelessness, trauma, or panic attacks
- Signs of a mental health crisis (e.g. "I'm losing control", "Everything feels pointless")

Respond **FALSE** if the message:
- Expresses everyday stress, frustration, tiredness, or low mood without urgent language
- Is vague, casual, or does not suggest mental health risks

Be cautious and only flag truly concerning input as TRUE.

Message:
\"\"\"{query}\"\"\"

Should we run mental illness analysis?
""")



class LLMIntentBasedIllnessClassifier:
    """
    A classifier using Groq's LLM to determine whether mental illness analysis is needed.
    """

    def __init__(self):
        """
        Initialize the LLM client using Groq API and config values from environment.

        :raises ValueError: If the GROQ_API_KEY is missing in the environment.
        """
        api_key = os.getenv("LLM_ILLNESS_GROQ_API_KEY")
        model_name = os.getenv("LLM_ILLNESS_GROQ_MODEL_NAME")
        temperature = float(os.getenv("LLM_ILLNESS_GROQ_TEMPERATURE"))
        max_tokens = int(os.getenv("LLM_ILLNESS_GROQ_MAX_TOKENS"))

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")

        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )

    async def needs_illness_check(self, user_input: str) -> bool:
        """
        Uses LLM to decide if the input indicates a need for illness analysis.

        :param user_input: User's input message.
        :returns: True if LLM suggests the input indicates significant distress.
        """
        try:
            prompt = ILLNESS_CLASSIFIER_PROMPT.format(query=user_input)
            logger.info("asking Groq if illness detection is needed...")

            response = await self.llm.ainvoke(prompt)
            result = response.content.strip().lower()

            logger.info(f"Illness check response from Groq: {result}")
            return "true" in result

        except Exception as e:
            logger.warning(f"Illness LLM classification failed: {e}")
            return False

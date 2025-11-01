"""
Module: rag_use_classifier.py
Description: Determines whether to use RAG and whether the input query is vague or clear.
Author: TechTeam AI Labs
Last Modified: 2025-07-08
"""

import os
import logging
from enum import Enum
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from rag.utils.fallback_handler import FallbackHandler, FallbackType
from rag.utils.config import settings

# Load environment variables
load_dotenv()

# ---------------------------
# Logging Configuration
# ---------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_path = "logs/rag_use_classifier.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ---------------------------
# Enum Definitions
# ---------------------------
class RAGDecision(str, Enum):
    RAG = "RAG"
    NO_RAG = "NO_RAG"

class VagueDecision(str, Enum):
    VAGUE = "VAGUE"
    CLEAR = "CLEAR"

# ---------------------------
# Prompt Template
# ---------------------------
CLASSIFIER_PROMPT = PromptTemplate.from_template("""
You are an expert classification system for a mental health assistant.

Your job is to decide if the user’s query:
1️⃣ Requires Retrieval-Augmented Generation (RAG)
2️⃣ Is VAGUE or CLEAR

## Definitions and Rules

🔹 **RAG** — Use ONLY if the query asks for factual, external knowledge that the assistant cannot infer from the conversation.
   - Examples that REQUIRE RAG: 
     - “What are the side effects of fluoxetine?”
     - “Give me statistics about depression rates in India.”
   - Examples that DO NOT REQUIRE RAG: 
     - “I feel sad, what should I do?”
     - “I’m overwhelmed by work.”

🔹 **NO_RAG** — Use for ALL queries that can be answered from general mental health support knowledge and session context.

🔹 **VAGUE** — Use ONLY if the query is incomplete, ambiguous, or impossible to answer without more context.
   - Examples that are VAGUE:
     - “What about that?”
     - “Tell me more.”
     - “What do you mean?”

🔹 **CLEAR** — Use for ANY query that is self-contained or understandable on its own.

## Important:
- Default to **NO_RAG** and **CLEAR** unless you have strong evidence to choose otherwise.
- Be strict: do not overuse RAG or VAGUE.

## Response Format
Respond ONLY in this format:

RAG: <RAG or NO_RAG>
VAGUE: <VAGUE or CLEAR>

## Query to classify:
{query}
""")

# ---------------------------
# RAG Use Classifier
# ---------------------------
class RAGUseClassifier:
    """
    Determines whether to apply RAG and whether the input query is vague or clear.
    """

    def __init__(self):
        if not settings.RAG_GROQ_API_KEY:
            raise EnvironmentError(" GROQ_API_KEY not found in settings.")

        self.llm = ChatGroq(
            model_name=settings.RAG_GROQ_MODEL_NAME,
            temperature=settings.RAG_GROQ_TEMPERATURE,
            max_tokens=settings.RAG_GROQ_MAX_TOKENS,
            api_key=settings.RAG_GROQ_API_KEY
        )
    
    async def classify(self, query: str) -> tuple[RAGDecision, VagueDecision]:
        """
        Classifies the given query for RAG usage and vagueness.

        :param query: The user input string
        :return: Tuple of (RAGDecision, VagueDecision)
        """
        try:
            prompt = CLASSIFIER_PROMPT.format(query=query)
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            logger.debug(f"[Classifier] Raw response:\n{content}")

            rag_decision = RAGDecision.NO_RAG
            vague_decision = VagueDecision.CLEAR

            for line in content.splitlines():
                if line.upper().startswith("RAG:"):
                    val = line.split(":", 1)[1].strip().upper()
                    if val in RAGDecision.__members__:
                        rag_decision = RAGDecision[val]
                elif line.upper().startswith("VAGUE:"):
                    val = line.split(":", 1)[1].strip().upper()
                    if val in VagueDecision.__members__:
                        vague_decision = VagueDecision[val]

            logger.info(f"[Classifier] Query classified as: RAG={rag_decision}, VAGUE={vague_decision}")
            return rag_decision, vague_decision

        except Exception as e:
            logger.error(f"[Classifier] Failed to classify query: {e}")
            return FallbackHandler.handle(
                fallback_type=FallbackType.CLASSIFIER_FAILURE,
                default_value=(RAGDecision.NO_RAG, VagueDecision.CLEAR),
                context={"query": query},
                error=e,
                user_message="Unable to classify your query. Proceeding with default behavior."
            )

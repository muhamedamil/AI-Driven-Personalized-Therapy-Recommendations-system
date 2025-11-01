"""
Module: query_expander.py
Description:
    Handles expansion of user queries by leveraging recent conversation history
    and known mental health context to make queries clearer without changing their intent.

    This is particularly useful for improving downstream reasoning or retrieval
    in mental health chatbot systems.

Created: 2025-06-12
Last Modified: 2025-07-08
"""

import logging
import os
from dotenv import load_dotenv
from typing import Optional

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage

from services.memory_service import MemoryService
from rag.utils.fallback_handler import FallbackHandler, FallbackType

# ---------------------------------
#  Logger Configuration
# ---------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/query_expansion.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ---------------------------------
#  Prompt Template for Expansion
# ---------------------------------
EXPANSION_PROMPT = PromptTemplate.from_template("""
You are a careful and supportive assistant for a mental health chatbot.

Your task is to rewrite the user's original query to make it clearer, more specific, and easier to understand, while preserving the original intent exactly. Use relevant details from the recent conversation if needed to clarify vague references, but do NOT add new ideas or assumptions.

## Inputs:
- Original user query: "{query}"
- Recent conversation context: "{history}"
- Known or inferred mental health condition: "{illness}"

## Rules:
- Only clarify what the user meant; do not invent or add unrelated content.
- Keep the rewritten query short, direct, and polite.
- Do not add any explanatory text or formatting.
- If the query is already clear, repeat it exactly as is.

## Output:
Respond with ONLY the rewritten query, nothing else.
""")

# ---------------------------------
# QueryExpander Class
# ---------------------------------
class QueryExpander:
    """
    Expands and clarifies user queries using conversation memory and illness metadata.
    Helps improve reasoning and search quality by rephrasing vague input.

    Attributes:
        memory_service (MemoryService): Service to manage session memory.
        llm (ChatGroq): Language model used for expansion.
    """

    def __init__(self, memory_service: MemoryService):
        load_dotenv()

        self.memory_service = memory_service
        api_key = os.getenv("LLM_EXPAND_GROQ_API_KEY")
        model_name = os.getenv("LLM_EXPAND_GROQ_MODEL_NAME", "llama3-70b-8192")
        temperature = float(os.getenv("LLM_EXPAND_GROQ_TEMPERATURE", "0"))
        max_tokens = int(os.getenv("LLM_EXPAND_GROQ_MAX_TOKENS", "256"))

        if not api_key:
            raise ValueError("LLM_EXPAND_GROQ_API_KEY not set in environment.")

        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )

    async def expand_query(
        self,
        db,
        query: str,
        session_id: Optional[str],
        user_id: str,
        illness: Optional[str] = None
    ) -> str:
        """
        Expand the user's query using conversation context and illness data.

        Args:
            db: Async database session.
            query (str): The user's original input.
            session_id (Optional[str]): Session ID for context retrieval.
            user_id (str): User identifier.
            illness (Optional[str]): Mental health condition if already known.

        Returns:
            str: A clarified or expanded version of the original query.
        """
        try:
            # Load memory from session and determine illness
            session = await self.memory_service.load_memory(db, session_id)
            illness = session.illness if (session and session.illness) else illness

            # Extract summary and last 5 messages
            summary = self.memory_service.memory.moving_summary_buffer or ""
            recent_messages: list[BaseMessage] = self.memory_service.memory.chat_memory.messages[-5:]

            recent_history = "\n".join(
                f"{m.type.capitalize()}: {m.content.strip()}"
                for m in recent_messages if getattr(m, "content", "").strip()
            )

            history = "\n".join(filter(None, [summary, recent_history])).strip()

            if not history:
                logger.info(f"[expand_query] No conversation history for session={session_id}; using query & illness only.")

            # Format the expansion prompt
            prompt_text = EXPANSION_PROMPT.format(
                query=query.strip(),
                history=history,
                illness=(illness or "").strip()
            )

            logger.debug(f"[expand_query] Expansion prompt for session={session_id}: {prompt_text[:200]}...")

            # Call the LLM
            response = await self.llm.ainvoke(prompt_text)
            expanded_query = response.content.strip()

            if expanded_query:
                logger.info(f"[expand_query] Successfully expanded query for session={session_id}.")
                return expanded_query
            else:
                logger.warning(f"[expand_query] LLM returned empty; falling back to original query.")
                return query.strip()

        except Exception as e:
            logger.exception(f"[expand_query] Exception for session={session_id}: {e}")
            return await FallbackHandler.handle(
                fallback_type=FallbackType.QUERY_EXPANSION_FAILURE,
                default_value=query.strip(),
                context={
                    "original_query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "illness": illness
                },
                error=e,
                user_message="Couldn't expand the query. Proceeding with your original query."
            )


__all__ = ["QueryExpander"]

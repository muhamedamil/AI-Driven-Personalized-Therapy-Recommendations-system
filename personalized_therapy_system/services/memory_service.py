"""
memory_service.py

This module provides a `MemoryService` class to manage conversational memory and summaries
for a mental health chatbot application. It integrates with a database (via SQLAlchemy),
uses LangChain memory components, and supports summarizing conversations with an LLM model.

Responsibilities:
- Load and restore memory from DB sessions.
- Track and trim recent conversation turns.
- Generate and store conversation summaries.
- Persist updated memory and metadata into the database.

Dependencies:
- FastAPI
- SQLAlchemy (AsyncSession)
- LangChain
- OpenRouterLLM (custom LLM wrapper)
"""

import logging
import json
import os
from dotenv import load_dotenv
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.messages import messages_from_dict, messages_to_dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database.models import UserSession
from services.openrouter_llm import OpenRouterLLM

logger = logging.getLogger(__name__)

load_dotenv()

# === Constants from env ===
MODEL_NAME = os.getenv("MEMORY_LLM_MODEL_NAME")
RECENT_TURNS_TO_KEEP = int(os.getenv("RECENT_TURNS_TO_KEEP"))



# === Prompt template for summarizing conversation history ===
summary_prompt = PromptTemplate(
    input_variables=["history"],
    template="""
You are an empathetic AI designed to provide mental health support. You are tasked with summarizing a conversation between a user and an assistant.

Please provide a **concise** summary of the conversation, focusing on the following key elements:
1. **User's emotional state**
2. **Main concerns or topics**
3. **Assistant's response**
4. **Tone and intent**

Make the summary **short, informative**, and ensure all **important points are included**.

Example:
Conversation Summary:
User expressed feeling overwhelmed due to work pressure and difficulty managing stress. The assistant recommended mindfulness techniques, breathing exercises, and scheduled breaks to help reduce stress and improve focus.

{history}
"""
)


class MemoryService:
    """
    Handles conversational memory management and summarization for user sessions.

    Uses LangChain's ConversationSummaryBufferMemory to:
    - Track dialogue history.
    - Generate conversation summaries via LLM.
    - Load and persist memory state in a database.
    """

    def __init__(self):
        """Initializes memory service with LLM and summary buffer."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env file.")
        
        self.llm = OpenRouterLLM(api_key=api_key, model_name=MODEL_NAME)
        self.memory = ConversationSummaryBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            llm=self.llm,
            k=RECENT_TURNS_TO_KEEP,
            input_key="input",
            output_key="output",
        )

    async def load_memory(self, db: AsyncSession, session_id: str) -> Optional[UserSession]:
        """
        Loads previous session messages and summary into memory.

        Args:
            db (AsyncSession): The active DB session.
            session_id (str): The session ID to load.

        Returns:
            Optional[UserSession]: The session object if found, else None.
        """
        try:
            result = await db.execute(
                select(UserSession).where(UserSession.session_id == session_id)
            )
            session = result.scalars().first()

            if session and session.messages:
                try:
                    messages_json = json.loads(session.messages)
                    self.memory.chat_memory.messages = messages_from_dict(messages_json)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Malformed session messages for session {session_id}: {e}")
                    self.memory.chat_memory.messages = []
                
                self.memory.moving_summary_buffer = session.summary or ""
            else:
                self.memory.chat_memory.messages = []
                self.memory.moving_summary_buffer = ""

            return session

        except Exception as e:
            await db.rollback()
            logger.exception(f"Failed to load memory for session {session_id}: {e}")
            self.memory.chat_memory.messages = []
            self.memory.moving_summary_buffer = ""
            return None

    async def save_memory(
        self,
        db: AsyncSession,
        session_id: str,
        illness: Optional[str] = None,
        intent: Optional[str] = None,
        response_style: Optional[str] = None,
        illness_detected: bool = False
    ) -> Optional[str]:
        """
        Saves memory, summaries, and optional metadata back to the database.

        If memory length exceeds `RECENT_TURNS_TO_KEEP`, it summarizes using the LLM and stores trimmed history.

        Args:
            db (AsyncSession): The active DB session.
            session_id (str): The session ID to update.
            illness (Optional[str]): Detected illness label.
            intent (Optional[str]): User's intent classification.
            response_style (Optional[str]): Chosen response tone or style.
            illness_detected (bool): Flag if illness was identified.

        Returns:
            Optional[str]: The session ID if saved successfully, else None.
        """
        try:
            result = await db.execute(
                select(UserSession).where(UserSession.session_id == session_id)
            )
            session = result.scalars().first()

            if not session:
                logger.error(f"Session {session_id} not found.")
                return None

            msgs = self.memory.chat_memory.messages or []
            current_summary = session.summary or ""

            # Count total history (summary lines + chat turns)
            total_history_turns = current_summary.count('\n') + len(msgs)

            if total_history_turns > RECENT_TURNS_TO_KEEP:
                full_history = current_summary
                if msgs:
                    full_history += "\n" + "\n".join(f"{m.type}: {m.content}" for m in msgs)
                prompt_text = summary_prompt.format(history=full_history)

                try:
                    out = await self.llm.agenerate([prompt_text])
                    generations = getattr(out, "generations", None)
                    new_summary = generations[0][0].text.strip() if generations and generations[0] else ""

                    if new_summary:
                        session.summary = new_summary
                        self.memory.moving_summary_buffer = new_summary
                        logger.info(f"Summary generated and saved for session {session_id}")
                    else:
                        logger.warning("Generated summary is empty.")
                except Exception as e:
                    logger.warning(f"Summary generation failed for session {session_id}: {e}")
                    logger.debug(f"Raw LLM output: {out}")

                # Keep only latest turns
                trimmed_msgs = msgs[-RECENT_TURNS_TO_KEEP:]
                self.memory.chat_memory.messages = trimmed_msgs
            else:
                # Just prune without summarization
                trimmed_msgs = msgs[-RECENT_TURNS_TO_KEEP:]
                self.memory.chat_memory.messages = trimmed_msgs

            # Save messages (as JSON)
            try:
                session.messages = json.dumps(messages_to_dict(trimmed_msgs))
            except Exception as e:
                logger.warning(f"Failed to serialize messages for session {session_id}: {e}")

            # Store optional metadata
            if illness is not None:
                session.illness = illness
            if intent is not None:
                session.intent = intent
            if response_style is not None:
                session.response_style = response_style
            session.illness_detected = illness_detected

            await db.commit()
            logger.info(f"Session {session_id} saved successfully.")
            return session_id

        except Exception as e:
            await db.rollback()
            logger.exception(f"Error saving session {session_id}: {e}")
            return None

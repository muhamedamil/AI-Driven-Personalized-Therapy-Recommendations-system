"""
Module: llm_service.py

Description:
    This module defines the LLMService class for handling the conversational logic of a mental health chatbot.
    It integrates speech-to-text transcription, vague query expansion, agent tool metadata extraction,
    RAG-enhanced responses, memory saving, and text-to-speech synthesis.

    The service handles both standard and streaming response modes, including metadata tracking,
    conversation context loading, and adaptive system prompts based on emotional intensity.

Created: 2025-07-08
"""
import os
import logging
import asyncio
from typing import Dict, List
from io import BytesIO

from fastapi.responses import StreamingResponse
import ffmpeg
import numpy as np
import soundfile as sf
from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from services.speech_service import SAMPLE_RATE, transcribe_audio, tts_to_base64
from services.session_service import update_session_metadata
from agents.mental_health_graph import get_mental_health_graph
from rag.rag_use_classifier import VagueDecision


from datetime import datetime
import time
import wandb

wandb.login(key=os.getenv("WANDB_API_KEY"))

# -------------------- Logging Setup -------------------- #
def setup_logger() -> logging.Logger:
    class SessionFilter(logging.Filter):
        def filter(self, record):
            record.session_id = getattr(record, 'session_id', '-')
            return True

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(SessionFilter())
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [sess:%(session_id)s] %(message)s"
        ))
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# -------------------- Prompt Templates -------------------- #

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_name", "additional_context"],
    template=(
        "You are a deeply compassionate and emotionally intelligent companion. "
        "You act as a supportive and trusted friend, especially for people who are feeling mentally low, anxious, overwhelmed, or emotionally burdened. "
        "Your purpose is to provide a safe space where the user feels heard, valued, and never judged.\n\n"

        "Be fully present, warm, and understanding — like someone who genuinely cares. Reflect back emotions with kindness, offer gentle encouragement, "
        "and help the user feel a little lighter by simply being there.\n\n"

        "Always prioritize empathy over advice. You don’t need to solve their problems. Instead, focus on making them feel understood. "
        "Validate their feelings, acknowledge their struggles, and if appropriate, offer small, comforting suggestions that are easy to follow.\n\n"

        "This is an ongoing, natural conversation. Be consistent with past exchanges — never treat each message as a new session. "
        "Use memory and conversation summaries to recall and build on previous moments the user has shared. "
        "If the user asks what they shared earlier, try to reflect on the available history or summary and respond with what you remember.\n\n"

        "Note: The user's name is {user_name}, but **only mention it if the user asks about it or brings it up naturally**. "
        "Avoid starting with greetings like 'Hi {user_name}' unless it makes sense emotionally in the moment.\n\n"

        "Your tone should be gentle, calm, and emotionally supportive. Use clear, easy-to-read language. You can express empathy with phrases like "
        "'That sounds really difficult,' 'You’re not alone in this,' or 'It makes sense that you feel that way.'\n\n"

        "{additional_context}"
    )
)

LIGHT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_name", "additional_context"],
    template=(
        "You are a friendly and thoughtful companion in a relaxed conversation. "
        "The user is not in distress. Your role is to keep the tone natural, light, and polite. "
        "Avoid emotionally intense language unless the user clearly expresses distress. "
        "Let the conversation flow casually and calmly, without offering therapy-style support unless prompted.\n\n"
        "{additional_context}"
    )
)

# -------------------- LLM Service -------------------- #
class LLMService:
    """
    LLMService manages the conversational pipeline for the mental health chatbot.

    Responsibilities:
    - Transcribe audio input from users.
    - Classify and expand vague inputs.
    - Invoke a mental health reasoning agent to detect metadata (intent, illness, etc.).
    - Run a RAG (retrieval-augmented generation) pipeline to retrieve relevant context.
    - Generate empathetic or light conversational responses using a large language model.
    - Convert text responses to speech (audio).
    - Persist conversation history and session metadata to memory and database.
    """

    def __init__(self, request: Request):
        """
        Initializes the LLMService instance with required services.

        Args:
            request (Request): FastAPI request object to access application state.
        """
        self.app = request.app
        self.memory_service = request.app.state.memory_service
        self.rag_pipeline = request.app.state.rag_pipeline
        self.llm = request.app.state.llm_model

    async def _transcode_and_transcribe(self, raw_audio: bytes) -> str:
        """
        Transcodes and transcribes user audio input.

        This method:
        - Converts raw audio from WebM format to WAV using FFmpeg.
        - Resamples the audio to the required sample rate and mono channel.
        - Uses a speech-to-text model to generate a transcript from the audio.

        Args:
            raw_audio (bytes): Raw binary audio data in WebM format.

        Returns:
            str: The transcribed text from the audio.

        Raises:
            HTTPException:
                - 400 if the audio format is unsupported or transcription fails.
                - 504 if the transcription operation times out.
        """
        def _transcode() -> np.ndarray:
            out, _ = (
                ffmpeg
                .input('pipe:0', format='webm')
                .output('pipe:1', format='wav', ar=str(SAMPLE_RATE), ac='1')
                .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
            )
            audio_np, _ = sf.read(BytesIO(out), dtype='int16')
            return audio_np

        try:
            audio_np = await asyncio.to_thread(_transcode)
            return await asyncio.wait_for(transcribe_audio(audio_np), timeout=20)
        except ffmpeg.Error as e:
            stderr = (e.stderr or b"").decode(errors="ignore")
            logger.error(f"Transcode failed: {stderr}")
            raise HTTPException(400, "Unsupported audio format.")
        except asyncio.TimeoutError:
            logger.error("STT timeout")
            raise HTTPException(504, "Transcription timed out.")
        except Exception as e:
            logger.error(f"STT failed: {e}")
            raise HTTPException(400, "Could not transcribe audio.")

    async def chat(
        self,
        db: AsyncSession,
        session_id: str,
        user_name: str,
        newly_created_session: bool,
        raw_audio: bytes,
    ) -> Dict[str, str]:
        """
        Main chat method to handle user input, generate response, and return audio.

        Steps:
        1. Load memory if this is not a newly created session.
        2. Transcribe incoming WebM audio to text.
        3. Check if the input is vague using a classifier, and expand it if necessary.
        4. Run the mental health agent graph tool to extract metadata (intent, illness, response style).
        5. Update the session metadata in the database.
        6. Run RAG pipeline using processed query and extracted metadata.
        7. Generate system prompt and construct message history.
        8. Generate response using the main LLM model.
        9. Save conversation to memory and persist it to DB.
        10. Convert reply text to speech (base64 audio).
        11. Return the full result as a dictionary.

        Args:
            db (AsyncSession): SQLAlchemy async database session.
            session_id (str): Unique session ID.
            user_name (str): User's display name or identifier.
            newly_created_session (bool): Whether this session is freshly created.
            raw_audio (bytes): Raw WebM audio input from user.

        Returns:
            Dict[str, str]: A dictionary containing:
                - transcript (str): Transcribed user input.
                - response (str): LLM-generated response.
                - audio_base64 (str): Response audio encoded in base64.
                - use_rag (bool): Whether RAG was used to assist response.
        """
        logger.info("Starting chat pipeline", extra={"session_id": session_id})
        
        start_time = time.time()    
        
        wandb.init(
        project="mental-health-llm",
        name=f"session_{session_id}_{int(time.time())}",
        config={
            "session_id": session_id,
            "user_name": user_name}
        )

        # Load memory if session exists
        if not newly_created_session:
            await self.memory_service.load_memory(db, session_id)

        # Transcribe user audio
        raw_transcript = await self._transcode_and_transcribe(raw_audio)
        if not raw_transcript or len(raw_transcript) > 8000:
            raise HTTPException(400, "Invalid input length.")

        processed_query = raw_transcript

        # Check vagueness & expand if needed
        _, vague_decision = await self.rag_pipeline.classifier.classify(processed_query)
        if vague_decision == VagueDecision.VAGUE:
            try:
                processed_query = await self.rag_pipeline.query_expander.expand_query(
                    db=db, query=processed_query, session_id=session_id, user_id=user_name
                )
                logger.info(f"Expanded query: {processed_query}", extra={"session_id": session_id})
            except Exception:
                logger.warning("Query expansion failed, using original text.", extra={"session_id": session_id})

        agent = await get_mental_health_graph()
        try:
            tool_output = await asyncio.wait_for(agent.ainvoke({
                "user_input": processed_query,
                "intent_category": None,
                "response_style": None,
                "illness_prediction": None
            }), timeout=25)
            logger.info(f"Agent tool output: {tool_output}", extra={"session_id": session_id})
        except asyncio.TimeoutError:
            logger.error("Agent tool usage timeout", extra={"session_id": session_id})
        except Exception as e:
            logger.error(f"Agent tool error: {e}", extra={"session_id": session_id})

        metadata = {
            "agent_tool_output": tool_output,
            "intent_category": tool_output.get("intent_category", {}).get("intent_category")
                if isinstance(tool_output.get("intent_category"), dict)
                else tool_output.get("intent_category"),
            "illness_prediction": tool_output.get("illness_prediction"),
            "response_style": tool_output.get("response_style"),
        }

        await update_session_metadata(
            db, session_id,
            illness=metadata.get('illness_prediction'),
            intent=metadata.get('intent_category'),
            response_style=metadata.get('response_style'),
            illness_detected=bool(metadata)
        )

        # RAG pipeline
        rag_out = await self.rag_pipeline.run(
            db=db,
            query=processed_query,
            session_id=session_id,
            user_id=user_name,
            illness=metadata.get('illness_prediction'),
            skip_processing=True
        )

        # Build system prompt
        system_prompt = self._build_system_prompt(metadata, rag_out, user_name)
        final_query = rag_out.get('final_query', processed_query)

        # Build messages with history
        messages = self._build_messages(system_prompt, final_query)
        logger.info(f"[STT] Transcribed text: '{messages}'")

        # Generate response from main LLM
        try:
            out = await asyncio.wait_for(
                self.llm.agenerate([final_query], messages=messages), timeout=60
            )
            reply = out.generations[0][0].text.strip()
        except asyncio.TimeoutError:
            logger.error("LLM generation timeout", extra={"session_id": session_id})
            raise HTTPException(504, "Response generation timed out.")
        except Exception:
            logger.exception("LLM generation failed", extra={"session_id": session_id})
            raise HTTPException(502, "Response generation failed.")

        # Save conversation in memory
        self.memory_service.memory.chat_memory.add_user_message(final_query)
        self.memory_service.memory.chat_memory.add_ai_message(reply)

        # Save updated session metadata & memory
        await self.memory_service.save_memory(
            db=db,
            session_id=session_id,
            illness=metadata.get('illness_prediction'),
            intent=metadata.get('intent_category'),
            response_style=metadata.get('response_style')
        )

        # Convert reply to speech
        audio_base64 = await tts_to_base64(reply)
        logger.info("Completed chat pipeline", extra={"session_id": session_id})
        
        
        wandb.log({
        "timestamp": datetime.now().isoformat(),
        "transcript": raw_transcript,
        "expanded_query": processed_query,
        "response": reply,
        "response_length": len(reply),
        "transcript_length": len(raw_transcript),
        "use_rag": rag_out.get("use_rag", False),
        "intent": metadata.get("intent_category"),
        "illness": metadata.get("illness_prediction"),
        "response_style": metadata.get("response_style"),
        "latency_seconds": round(time.time() - start_time, 2),
        })

        wandb.finish()
        
        return {
            "transcript": raw_transcript,
            "response": reply,
            "audio_base64": audio_base64,
            "use_rag": rag_out.get('use_rag', False)
        }

    async def stream_chat(
        self,
        db: AsyncSession,
        session_id: str,
        user_name: str,
        newly_created_session: bool,
        raw_audio: bytes,
    ):
        """
        Handles real-time streaming chat responses from the LLM.

        This method:
        - Transcribes user's audio input.
        - Expands vague input if detected using classifier.
        - Invokes mental health agent to detect metadata (intent, illness, response style).
        - Updates session metadata in the database.
        - Runs the RAG pipeline to fetch relevant information.
        - Builds messages with system prompt and memory.
        - Streams the generated response token-by-token using the LLM.
        - Streams final output with transcript followed by the generated tokens.

        Args:
            db (AsyncSession): Async DB session.
            session_id (str): ID of the user session.
            user_name (str): User’s name or identifier.
            newly_created_session (bool): Whether this is a new session.
            raw_audio (bytes): Raw audio input in WebM format.

        Returns:
            StreamingResponse: A streaming plain-text response consisting of the transcript
            followed by token-wise output from the language model.
        """
        logger.info("starting streaming chat pipeline", extra={"session_id": session_id})
        
        start_time = time.time()
        
        
        wandb.init(
        project="mental-health-llm",
        name=f"session_{session_id}_{int(time.time())}",
        config={
            "session_id": session_id,
            "user_name": user_name
        })

        if not newly_created_session:
            await self.memory_service.load_memory(db, session_id)

        raw_transcript = await self._transcode_and_transcribe(raw_audio)
        if not raw_transcript or len(raw_transcript) > 8000:
            raise HTTPException(400, "Invalid input length.")

        processed_query = raw_transcript

        _, vague_decision = await self.rag_pipeline.classifier.classify(processed_query)
        if vague_decision.name == "VAGUE":
            try:
                processed_query = await self.rag_pipeline.query_expander.expand_query(
                    db=db, query=processed_query, session_id=session_id, user_id=user_name
                )
                logger.info(f"Expanded query: {processed_query}", extra={"session_id": session_id})
            except Exception:
                logger.warning("Query expansion failed, using original text.", extra={"session_id": session_id})

        try:
            agent = await get_mental_health_graph()
            tool_output = await asyncio.wait_for(agent.ainvoke({
                "user_input": processed_query,
                "intent_category": None,
                "response_style": None,
                "illness_prediction": None
            }), timeout=25)
            logger.info(f"Agent tool output: {tool_output}", extra={"session_id": session_id})
        except Exception as e:
            logger.warning(f"Agent tool usage failed: {e}", extra={"session_id": session_id})
            tool_output = {}

        metadata = {
            "intent_category": tool_output.get("intent_category", {}).get("intent_category")
                if isinstance(tool_output.get("intent_category"), dict)
                else tool_output.get("intent_category"),
            "illness_prediction": tool_output.get("illness_prediction"),
            "response_style": tool_output.get("response_style"),
        }

        await update_session_metadata(
            db, session_id,
            illness=metadata.get('illness_prediction'),
            intent=metadata.get('intent_category'),
            response_style=metadata.get('response_style'),
            illness_detected=bool(tool_output)
        )

        rag_out = await self.rag_pipeline.run(
            db=db,
            query=processed_query,
            session_id=session_id,
            user_id=user_name,
            illness=metadata.get("illness_prediction"),
            skip_processing=True
        )

        system_prompt = self._build_system_prompt(metadata, rag_out, user_name)
        final_query = rag_out.get("final_query", processed_query)
        messages = self._build_messages(system_prompt, final_query)
        logger.info(f" The final prompt is {messages}")

        async def token_stream():
            """
            Generator to yield the transcript followed by streamed tokens.

            Accumulates the full response for memory saving at the end.
            """
            response_accumulator = ""

            try:
                yield f"{raw_transcript}<<END_OF_TRANSCRIPT>>".encode("utf-8")

                async for token in self.llm.stream(input=final_query, messages=messages):
                    response_accumulator += token
                    yield token.encode("utf-8")

                self.memory_service.memory.chat_memory.add_user_message(final_query)
                self.memory_service.memory.chat_memory.add_ai_message(response_accumulator)

                await self.memory_service.save_memory(
                    db=db,
                    session_id=session_id,
                    illness=metadata.get('illness_prediction'),
                    intent=metadata.get('intent_category'),
                    response_style=metadata.get('response_style')
                )

            except Exception as e:
                logger.exception("Streaming LLM error", extra={"session_id": session_id})
                yield "[ERROR: Streaming failed]".encode("utf-8")
                
        
            wandb.log({
            "timestamp": datetime.now().isoformat(),
            "transcript": raw_transcript,
            "expanded_query": processed_query,
            "response": response_accumulator,
            "response_length": len(response_accumulator),
            "transcript_length": len(raw_transcript),
            "use_rag": rag_out.get("use_rag", False),
            "intent": metadata.get("intent_category"),
            "illness": metadata.get("illness_prediction"),
            "response_style": metadata.get("response_style"),
            "latency_seconds": round(time.time() - start_time, 2),
            })

        return StreamingResponse(
            token_stream(),
            media_type="text/plain"
        )
        

    def _build_system_prompt(self, metadata: dict, rag_out: dict, user_name: str) -> str:
        """
        Constructs the system prompt for the LLM based on user metadata and retrieved context.

        Args:
            metadata (dict): Metadata from the agent tool (intent, illness, response style).
            rag_out (dict): Output from the RAG pipeline including documents and flags.
            user_name (str): User's name.

        Returns:
            str: A formatted system prompt string to guide LLM behavior.
        """
        context = []
        if "agent_tool_output" in metadata:
            context.append(f"Agent tool output:\n{metadata['agent_tool_output']}")

        ip = metadata.get('illness_prediction')
        ic = metadata.get('intent_category')
        rs = metadata.get('response_style')

        if ip: context.append(f"Detected mental health condition: {ip}.")
        if ic: context.append(f"User intent: {ic}.")
        if rs: context.append(f"Preferred response style: {rs}.")

        if rag_out.get('use_rag') and rag_out.get('documents'):
            docs = "\n".join(f"• {d.page_content}" for d in rag_out['documents'])
            context.append(f"Relevant info:\n{docs}")

        additional_context = "\n".join(context)

        neutral_intents = {
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
            "Looking for mindfulness or relaxation techniques",
            "Need motivation or productivity tips",
            "Want therapy or self-care recommendations",
            "Have questions about medication or psychiatry",
            "Looking for professional help",
            "Other concerns that don’t fit above"
        }

        if ic in neutral_intents:
            logger.info("Using LIGHT prompt")
            return LIGHT_PROMPT_TEMPLATE.format(user_name=user_name, additional_context=additional_context)
        else:
            logger.info("Using SYSTEM (empathic) prompt")
            return SYSTEM_PROMPT_TEMPLATE.format(user_name=user_name, additional_context=additional_context)

    def _build_messages(self, system_prompt: str, final_query: str) -> List:
        """
        Builds the message history for the LLM input including memory and user input.

        Args:
            system_prompt (str): Prompt generated from system metadata.
            final_query (str): User's current query after processing.

        Returns:
            List: A list of SystemMessage, HumanMessage, and AIMessage objects for LLM input.
        """
        memory_vars = self.memory_service.memory.load_memory_variables({})
        history = memory_vars.get('chat_history', [])
        summary = memory_vars.get('summary', "")

        messages = [SystemMessage(content=system_prompt)]

        if summary:
            messages.append(SystemMessage(content=f"**Conversation Summary:**\n{summary}"))

        for entry in history:
            role = entry.get("role") if isinstance(entry, dict) else getattr(entry, "type", "user")
            content = entry.get("content") if isinstance(entry, dict) else getattr(entry, "content", "")
            if role in ["user", "human"]:
                messages.append(HumanMessage(content=content))
            elif role in ["assistant", "ai"]:
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=final_query))
        return messages

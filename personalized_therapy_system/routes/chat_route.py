"""
chat_route.py

This module defines the FastAPI routes for:
- Serving the chat UI page
- Handling user-to-AI interactions through audio (non-streaming and streaming)
- Converting text to speech via TTS

Each route handles session validation, interacts with the LLM service, and returns structured responses.
"""

from fastapi import (
    APIRouter, Depends, HTTPException, Request, UploadFile, File
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from database.db_config import get_async_db
from database.schemas import ChatResponse
from services.llm_service import LLMService
from services.speech_service import tts_to_base64

import logging

# ───────────────────────────
# Router & Logger Setup
# ───────────────────────────

router = APIRouter(prefix="/chat", tags=["Chat"])
templates = Jinja2Templates(directory="templates")
log = logging.getLogger(__name__)


# ───────────────────────────
# GET: Chat UI Page
# ───────────────────────────

@router.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """
    GET /chat

    Renders the chat UI page.

    Validates session ID and user ID from session data.
    Redirects to login page if session is missing.
    """
    session_id = request.session.get("session_id")
    user_id = request.session.get("user_id")
    user_name = request.session.get("user_name")
    newly_created_session = request.session.get("newly_created_session", False)
    session_created_at = request.session.get("session_created_during_login")

    if not session_id or not user_id:
        return RedirectResponse(url="/auth/login", status_code=302)

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "session_id": session_id,
            "user_id": user_id,
            "user_name": user_name,
            "newly_created_session": newly_created_session,
            "session_created_at": session_created_at,
        }
    )


# ───────────────────────────
# POST: Non-Streaming Chat
# ───────────────────────────

@router.post("/message", response_model=ChatResponse)
async def chat_with_bot(
    request: Request,
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_db),
):
    """
    POST /chat/message

    Handles user audio input and returns a full AI response (non-streaming).

    Parameters:
        - request: FastAPI Request object with session
        - audio: UploadFile (user's voice message in audio format)
        - db: AsyncSession for DB operations

    Returns:
        - ChatResponse model containing AI's reply, illness, intent, etc.
    """
    session_id = request.session.get("session_id")
    user_name = request.session.get("user_name")
    newly_created_session = request.session.get("newly_created_session", False)

    if not session_id or not user_name:
        raise HTTPException(status_code=401, detail="Unauthorized: session expired or missing.")

    raw_audio = await audio.read()
    llm_service = LLMService(request)

    response = await llm_service.chat(
        db=db,
        session_id=session_id,
        user_name=user_name,
        newly_created_session=newly_created_session,
        raw_audio=raw_audio,
    )

    if newly_created_session:
        request.session["newly_created_session"] = False

    log.debug("chat_with_bot response: %r", response)
    return ChatResponse(**response)


# ───────────────────────────
# POST: Streaming Chat
# ───────────────────────────

@router.post("/stream-message")
async def stream_chat_with_bot(
    request: Request,
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_db),
):
    """
    POST /chat/stream-message

    Handles audio-based chat and returns a streamed AI response token-by-token.

    Parameters:
        - request: FastAPI Request object with session
        - audio: UploadFile (user's voice message in audio format)
        - db: AsyncSession for DB operations

    Returns:
        - Streaming response object (StreamingResponse or equivalent)
    """
    session_id = request.session.get("session_id")
    user_name = request.session.get("user_name")
    newly_created_session = request.session.get("newly_created_session", False)

    if not session_id or not user_name:
        raise HTTPException(status_code=401, detail="Unauthorized: session expired or missing.")

    raw_audio = await audio.read()
    llm_service = LLMService(request)

    response = await llm_service.stream_chat(
        db=db,
        session_id=session_id,
        user_name=user_name,
        newly_created_session=newly_created_session,
        raw_audio=raw_audio,
    )

    if newly_created_session:
        request.session["newly_created_session"] = False

    return response


# ───────────────────────────
# POST: Text-to-Speech (TTS)
# ───────────────────────────

class TTSRequest(BaseModel):
    text: str


@router.post("/tts")
async def generate_tts_audio(payload: TTSRequest):
    """
    POST /chat/tts

    Converts given text to speech using a TTS engine (e.g., Bark or ElevenLabs).

    Parameters:
        - payload: JSON body with `text` field

    Returns:
        - JSON object with base64-encoded audio string
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        audio_base64 = await tts_to_base64(text)
        if not audio_base64:
            raise HTTPException(status_code=500, detail="TTS generation failed.")

        return {"audio_base64": audio_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

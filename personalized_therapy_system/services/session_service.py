"""
Module: session_service.py
Description:
    This module manages user session lifecycle including:
    - Starting and ending sessions
    - Fetching previous sessions
    - Handling session expiry
    - Updating session metadata

    This is central to managing contextual conversations in applications such as
    mental health chatbots or any persistent dialog system.

Created: 2025-06-12
Last Modified: 2025-07-08
"""

import os
from dotenv import load_dotenv
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError

from database.models import UserSession

load_dotenv() 

# -------------------- Logging Setup -------------------- #
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Session expiry duration (in minutes)
SESSION_EXPIRY_MINUTES = int(os.getenv("SESSION_EXPIRY_MINUTES", 60))
# -------------------- Session Management Utilities -------------------- #

async def start_new_session(
    db: AsyncSession,
    user_id: str,
    illness: Optional[str] = None,
    intent: Optional[str] = None,
    response_style: Optional[str] = None,
    illness_detected: bool = False,
    messages: Optional[str] = None,
    summary: Optional[str] = None
) -> str:
    """
    Starts a new session for a user, deactivating all currently active sessions.

    Args:
        db (AsyncSession): SQLAlchemy async database session.
        user_id (str): ID of the user.
        illness (Optional[str]): Illness name, if detected.
        intent (Optional[str]): Detected intent for the session.
        response_style (Optional[str]): Preferred response style.
        illness_detected (bool): Flag if illness was detected.
        messages (Optional[str]): Initial messages to attach.
        summary (Optional[str]): Summary of the session.

    Returns:
        str: The UUID of the newly created session.
    """
    try:
        result = await db.execute(
            select(UserSession).where(UserSession.user_id == user_id, UserSession.is_active == True)
        )
        active_sessions = result.scalars().all()
        for session in active_sessions:
            session.is_active = False
        await db.commit()

        new_session = UserSession(
            user_id=user_id,
            illness=illness,
            intent=intent,
            response_style=response_style,
            illness_detected=illness_detected,
            messages=messages,
            summary=summary,
            is_active=True
        )

        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)

        logger.info(f"New session started for user {user_id}: {new_session.session_id}")
        return new_session.session_id

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Failed to start session for user {user_id}: {e}")
        raise


async def get_last_active_session_by_user(db: AsyncSession, user_id: str) -> Optional[UserSession]:
    """
    Retrieves the most recent active session for the given user.

    Args:
        db (AsyncSession): Database session.
        user_id (str): User's ID.

    Returns:
        Optional[UserSession]: Most recent active session or None.
    """
    try:
        result = await db.execute(
            select(UserSession)
            .where(UserSession.user_id == user_id, UserSession.is_active == True)
            .order_by(UserSession.timestamp.desc())
        )
        return result.scalars().first()
    except SQLAlchemyError as e:
        logger.error(f"Error fetching last active session for user {user_id}: {e}")
        return None


async def get_last_session_by_user(db: AsyncSession, user_id: str) -> Optional[UserSession]:
    """
    Fetches the most recent session (active or not) for a user.

    Args:
        db (AsyncSession): Database session.
        user_id (str): User's ID.

    Returns:
        Optional[UserSession]: Most recent session or None.
    """
    try:
        result = await db.execute(
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .order_by(UserSession.timestamp.desc())
        )
        return result.scalars().first()
    except SQLAlchemyError as e:
        logger.error(f"Error fetching last session for user {user_id}: {e}")
        return None


async def get_nth_last_session_by_user(
    db: AsyncSession,
    user_id: str,
    n: int = 0
) -> Optional[UserSession]:
    """
    Retrieves the N-th most recent session for a user.

    Args:
        db (AsyncSession): Database session.
        user_id (str): User's ID.
        n (int): Index (0 = latest, 1 = second latest, ...)

    Returns:
        Optional[UserSession]: N-th recent session or None.
    """
    try:
        result = await db.execute(
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .order_by(desc(UserSession.timestamp))
            .offset(n)
            .limit(1)
        )
        return result.scalars().first()
    except SQLAlchemyError as e:
        logger.error(f"Error fetching {n}-th session for user {user_id}: {e}")
        return None


def is_session_expired(session: UserSession) -> bool:
    """
    Determines if a session has expired based on a predefined duration.

    Args:
        session (UserSession): The session instance.

    Returns:
        bool: True if the session is expired, otherwise False.
    """
    return datetime.utcnow() - session.timestamp > timedelta(minutes=SESSION_EXPIRY_MINUTES)


async def get_or_create_active_session(db: AsyncSession, user_id: str) -> Tuple[str, bool]:
    """
    Fetches an active session or creates a new one if none is found or it's expired.

    Args:
        db (AsyncSession): Database session.
        user_id (str): User's ID.

    Returns:
        Tuple[str, bool]: (Session ID, True if new session was created)
    """
    last_session = await get_last_active_session_by_user(db, user_id)

    if not last_session or is_session_expired(last_session):
        if last_session:
            last_session.is_active = False
            try:
                await db.commit()
            except SQLAlchemyError:
                await db.rollback()
                logger.warning(f"Failed to deactivate expired session {last_session.session_id}")
        new_session_id = await start_new_session(db, user_id)
        return new_session_id, True

    return last_session.session_id, False


async def update_session_metadata(
    db: AsyncSession,
    session_id: str,
    illness: Optional[str] = None,
    intent: Optional[str] = None,
    response_style: Optional[str] = None,
    illness_detected: Optional[bool] = None,
    messages: Optional[str] = None,
    summary: Optional[str] = None,
    deactivate: bool = False
) -> None:
    """
    Updates metadata of an existing session.

    Args:
        db (AsyncSession): Database session.
        session_id (str): ID of the session to update.
        illness (Optional[str]): Updated illness info.
        intent (Optional[str]): Updated intent.
        response_style (Optional[str]): Response style to be saved.
        illness_detected (Optional[bool]): Illness detection flag.
        messages (Optional[str]): Updated conversation messages.
        summary (Optional[str]): Summary of the session.
        deactivate (bool): Whether to deactivate the session.

    Returns:
        None
    """
    try:
        result = await db.execute(
            select(UserSession).where(UserSession.session_id == session_id)
        )
        session = result.scalars().first()

        if not session:
            logger.warning(f"No session found with session_id {session_id}")
            return

        if illness is not None:
            session.illness = illness
        if intent is not None:
            session.intent = intent
        if response_style is not None:
            session.response_style = response_style
        if illness_detected is not None:
            session.illness_detected = illness_detected
        if messages is not None:
            session.messages = messages
        if summary is not None:
            session.summary = summary
        if deactivate:
            session.is_active = False

        await db.commit()
        logger.info(f"Session {session_id} updated successfully")

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Failed to update session {session_id}: {e}")
        raise

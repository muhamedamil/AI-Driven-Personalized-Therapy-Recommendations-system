# -*- coding: utf-8 -*-
"""
Module: db.py
Description: Asynchronous SQLAlchemy session management and engine setup for database access.
Created: 2025-06-14
Last Modified: 2025-07-08
"""

import os
import logging
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# Load .env from parent directory
dotenv_path = Path(__file__).parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    logger.info("Loaded environment variables from .env")
else:
    logger.warning(".env file not found at expected path.")

# Retrieve DB URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL is not set in the environment.")

logger.info(f"Using database URL: {DATABASE_URL}")

# Create async SQLAlchemy engine
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Create session factory for async sessions
async_session_maker = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Declarative base for ORM models
Base = declarative_base()

# -------------------------------
# Async DB Session Dependency
# -------------------------------
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session for FastAPI routes.

    :yield: Async SQLAlchemy session object.
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

__all__ = ["engine", "async_session_maker", "Base", "get_async_db"]

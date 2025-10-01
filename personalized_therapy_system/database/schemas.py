# -*- coding: utf-8 -*-
"""
Module: schemas.py
Description: Pydantic schemas for request validation and API responses.
Author: TechTeam AI Labs
Created: 2024-06-05
Last Modified: 2025-07-08
"""

from uuid import UUID
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# -------------------------------
# Request Schemas
# -------------------------------

class RegisterRequest(BaseModel):
    """
    Schema for user registration input.
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=4)


class LoginRequest(BaseModel):
    """
    Schema for user login input.
    """
    email: EmailStr
    password: str


# -------------------------------
# Response Schemas
# -------------------------------

class UserResponse(BaseModel):
    """
    Schema returned after user registration or login.
    """
    user_id: UUID
    username: str
    email: EmailStr
    created_at: datetime
    session_id: Optional[UUID] = None
    session_created: bool

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """
    Schema containing JWT token response.
    """
    access_token: str
    token_type: str = "bearer"


class SessionResponse(BaseModel):
    """
    Schema returned after a new session is created.
    """
    session_id: str
    session_created: datetime
    user_name: str


class ChatResponse(BaseModel):
    """
    Schema for returning chat system response.
    """
    transcript: str
    response: str
    use_rag: bool
    audio_base64: Optional[str] = None


__all__ = [
    "RegisterRequest",
    "LoginRequest",
    "UserResponse",
    "TokenResponse",
    "SessionResponse",
    "ChatResponse"
]

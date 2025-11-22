"""
Module: auth_services.py
Description:
    This module handles user authentication and registration logic:
    - Registering new users (with email uniqueness check and password hashing)
    - Authenticating users and retrieving or creating their session
Created: 2025-06-25
Last Modified: 2025-07-08
"""

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from sqlalchemy import select
from uuid import UUID
from typing import Optional

from database.models import User
from database.schemas import RegisterRequest, UserResponse
from services.session_service import get_or_create_active_session
from auth.password_service import hash_password, verify_password


async def register_user(register_data: RegisterRequest, db: AsyncSession) -> UserResponse:
    """
    Registers a new user.

    Steps:
    - Checks if the email is already registered.
    - Hashes the password securely.
    - Creates and commits a new user to the database.

    Args:
        register_data (RegisterRequest): User registration data.
        db (AsyncSession): Async SQLAlchemy database session.

    Returns:
        UserResponse: Sanitized response with user details (excluding password).

    Raises:
        HTTPException: 400 if user already exists.
        HTTPException: 500 if password hashing fails.
    """
    result = await db.execute(
        select(User).filter(User.email == register_data.email)
    )
    existing_user = result.scalars().first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    hashed_pw = hash_password(register_data.password)
    if not hashed_pw:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password hashing failed"
        )

    new_user = User(
        username=register_data.username,
        email=register_data.email,
        password_hash=hashed_pw
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return UserResponse(
        user_id=new_user.user_id,
        username=new_user.username,
        email=new_user.email,
        created_at=new_user.created_at
    )


async def authenticate_user(email: str, password: str, db: AsyncSession) -> Optional[UserResponse]:
    """
    Authenticates a user and returns session info.

    Steps:
    - Verifies email and password match.
    - Fetches or creates an active session.

    Args:
        email (str): User email.
        password (str): Plain-text password.
        db (AsyncSession): Async database session.

    Returns:
        Optional[UserResponse]: Authenticated user data with session info.

    Raises:
        HTTPException: 404 if user not found.
        HTTPException: 401 if password is incorrect.
    """
    result = await db.execute(
        select(User).filter(User.email == email)
    )
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")

    session_id, session_created = await get_or_create_active_session(db, user.user_id)

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        session_id=session_id,
        session_created=session_created
    )

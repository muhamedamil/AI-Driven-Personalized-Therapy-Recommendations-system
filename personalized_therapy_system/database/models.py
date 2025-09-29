# -*- coding: utf-8 -*-
"""
Module: models.py
Description: Contains SQLAlchemy ORM models for User and UserSession entities.
Author: TechTeam AI Labs
Created: 2024-06-05
Last Modified: 2025-07-08
"""

import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, func, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from database.db_config import Base


class User(Base):
    """
    ORM model representing application users.
    """
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class UserSession(Base):
    """
    ORM model representing a single session of user interaction.
    """
    __tablename__ = "user_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)

    messages = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    illness = Column(String, nullable=True)
    intent = Column(String, nullable=True)
    response_style = Column(String, nullable=True)
    illness_detected = Column(Boolean, default=False)

    timestamp = Column(DateTime, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, session_id={self.session_id})>"


__all__ = ["User", "UserSession"]

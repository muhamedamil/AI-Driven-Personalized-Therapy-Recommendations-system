"""
Module: home_routes.py
Description:
    Provides routes for home page rendering and user session handling in a mental health chatbot platform.
    Supports continuing a previous session or starting a new one based on user input.
Created: 2025-06-30
Last Modified: 2025-07-08
"""

from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from database.db_config import get_async_db
from database.models import UserSession
from services.session_service import (
    get_nth_last_session_by_user,
    start_new_session
)

router = APIRouter(
    prefix="/home",
    tags=["Home"]
)

templates = Jinja2Templates(directory="templates")


def serialize_datetime(obj):
    """
    Convert datetime object to ISO format string for JSON serialization.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


@router.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """
    GET /home

    Serves the home page for authenticated users.
    Redirects unauthenticated users to the login page.
    """
    user_id = request.session.get("user_id")
    user_name = request.session.get("user_name")

    if not user_id:
        return RedirectResponse(url="/auth/login", status_code=302)

    return templates.TemplateResponse("home.html", {
        "request": request,
        "user_name": user_name
    })


@router.post("/session-handler")
async def handle_session(
    request: Request,
    choice: str = Form(...),
    db: AsyncSession = Depends(get_async_db)
):
    """
    POST /home/session-handler

    Handles user input to either:
    - Continue the most recent session (if exists), or
    - Start a new session

    Updates the session cookie and returns a JSON response with session info.

    Request Form Fields:
    - choice: either "continue" or "new"

    Raises:
    - 401 if user is not authenticated
    - 404 if continuing and no previous session found
    - 400 for invalid choice
    """
    user_id = request.session.get("user_id")
    user_name = request.session.get("user_name")

    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    newly_created_session = False

    if choice.lower() == "continue":
        last_session = await get_nth_last_session_by_user(db, user_id, n=1)
        if not last_session:
            raise HTTPException(status_code=404, detail="No previous session found")
        session_id = str(last_session.session_id)
        session_created = last_session.timestamp

    elif choice.lower() == "new":
        session_id = await start_new_session(db, user_id)
        newly_created_session = True

        # Retrieve timestamp of newly created session
        result = await db.execute(
            select(UserSession).where(UserSession.session_id == session_id)
        )
        session_obj = result.scalars().first()
        session_created = session_obj.timestamp if session_obj else datetime.utcnow()

    else:
        raise HTTPException(status_code=400, detail="Invalid choice")

    # Update session state
    request.session["session_id"] = str(session_id)
    request.session["session_created_during_login"] = session_created.isoformat()
    request.session["user_id"] = user_id
    request.session["user_name"] = user_name
    request.session["newly_created_session"] = newly_created_session

    return JSONResponse(content={
        "session_id": str(session_id),
        "user_name": user_name,
        "session_created": session_created.isoformat(),
        "newly_created_session": newly_created_session,
        "redirect_url": "/chat"
    })

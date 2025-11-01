"""
Authentication Routes

Provides routes for user registration and login:
- GET /login and /register pages (HTML)
- POST /register handles form-based signup
- POST /login supports both form-based and JSON login
- Sets session and cookie for web sessions
"""

from fastapi import (
    APIRouter, Depends, HTTPException, status,
    Request, Form, Body, Response
)
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from services.auth_services import register_user, authenticate_user
from database.schemas import RegisterRequest, LoginRequest
from database.db_config import get_async_db

# --- Router Setup ---
router = APIRouter(prefix="/auth", tags=["Authentication"])
templates = Jinja2Templates(directory="templates")


# === Page Routes ===

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login HTML page."""
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render the registration HTML page."""
    return templates.TemplateResponse("register.html", {"request": request})


# === Form + JSON Handlers ===

@router.post("/register", response_class=HTMLResponse)
async def register_user_route(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Handle user registration form submission.

    - Validates form data
    - Calls `register_user` service
    - Redirects to login on success
    """
    try:
        data = RegisterRequest(username=username, email=email, password=password)
        await register_user(data, db)
        return RedirectResponse(url="/auth/login", status_code=303)

    except HTTPException as e:
        return templates.TemplateResponse(
            "register.html", {"request": request, "error": e.detail},
            status_code=e.status_code,
        )
    except Exception:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Registration failed. Please try again."},
            status_code=500,
        )


@router.post("/login")
async def login_user_route(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
    email: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    login_data: Optional[LoginRequest] = Body(None),
):
    """
    Handles both form and JSON login:

    - Validates credentials
    - If JSON: returns user JSON response
    - If form: sets session + cookie and redirects
    """
    is_json = request.headers.get("content-type", "").startswith("application/json")

    try:
        # Handle JSON login
        if is_json:
            if not login_data:
                raise HTTPException(status_code=400, detail="Missing JSON body.")
            email = login_data.email
            password = login_data.password

        # Validate required credentials
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required.")

        # Authenticate user
        user = await authenticate_user(email, password, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        # If JSON login, return user info
        if is_json:
            return JSONResponse(content=user.model_dump(), status_code=200)

        # Form login: set session cookie and redirect
        redirect = RedirectResponse(url="/home", status_code=303)
        redirect.set_cookie(
            key="session_id",
            value=str(user.session_id),
            httponly=True,
            secure=False,       # Set to True in production (HTTPS)
            samesite="lax",
            max_age=86400       # 1 day
        )

        request.session["session_id"] = str(user.session_id)
        request.session["user_id"] = str(user.user_id)
        request.session["user_name"] = user.username

        return redirect

    except HTTPException as e:
        if is_json:
            raise e
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": e.detail},
            status_code=e.status_code,
        )

    except Exception:
        if is_json:
            raise HTTPException(status_code=500, detail="Login failed. Please try again.")
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Server error during login."},
            status_code=500,
        )

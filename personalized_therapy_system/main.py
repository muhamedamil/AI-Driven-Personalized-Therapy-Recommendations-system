"""
Module: main.py

Description:
    Entry point for the AI-Powered Psychologist FastAPI application.

    Responsibilities:
    - Initializes all global services using FastAPI lifespan
    - Loads machine learning models and vector store
    - Registers API routers for authentication, home, and chat functionalities
    - Manages session cookies, CORS policy, and custom error pages

Created: 2025-07-01
Last Modified: 2025-07-08
"""

import os
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer

# Import application services and routers
from services.memory_service import MemoryService
from rag.vector_store import VectorStoreService
from rag.utils.similarity_filter import SimilarityFilter
from rag.rag_pipeline import RAGPipeline
from services.openrouter_llm import OpenRouterLLM
from routes import home_route, auth_route, chat_route

# -------------------------------
# LIFESPAN CONTEXT MANAGER
# -------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes all core services at application startup
    and releases resources on shutdown.
    """
    print("Initializing global services (lifespan)...")

    # Core services
    app.state.memory_service = MemoryService()
    app.state.similarity_filter = SimilarityFilter()

    # Embedding model (for vector store & semantic search)
    app.state.embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    app.state.embedding_model = SentenceTransformer(app.state.embedding_model_name)

    # Vector store
    app.state.vectorstore = await VectorStoreService.create(app.state.embedding_model_name)

    # RAG pipeline for context-aware response generation
    app.state.rag_pipeline = RAGPipeline(
        memory_service=app.state.memory_service,
        vectorstore=app.state.vectorstore,
        similarity_filter=app.state.similarity_filter
    )

    # LLM client (via OpenRouter)
    app.state.llm_model = OpenRouterLLM(
                                        api_key=os.getenv("OPENROUTER_API_KEY"),
                                        api_url="https://openrouter.ai/api/v1/chat/completions")


    print("Global services and tools ready.")
    yield  # Application is now running
    print("Lifespan shutdown: cleanup done.")

# -------------------------------
# FastAPI App with Lifespan
# -------------------------------

app = FastAPI(
    title="AI Powered Psychologist",
    lifespan=lifespan
)

# -------------------------------
# Middleware: Session & CORS
# -------------------------------

SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "fallback-dev-secret")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Template Engine Setup
# -------------------------------

templates = Jinja2Templates(directory="templates")

# -------------------------------
# Router Registration
# -------------------------------

print("Registering routers...")
app.include_router(auth_route.router)
print("auth_route registered")

app.include_router(home_route.router)
print("home_route registered")

app.include_router(chat_route.router)
print("chat_route registered")

# -------------------------------
# Root Redirect
# -------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """
    Redirects root URL to the login page.
    """
    return HTMLResponse('<script>window.location.href="/auth/login";</script>')

# -------------------------------
# Custom 404 Page
# -------------------------------

@app.exception_handler(StarletteHTTPException)
async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    """
    Handles 404 errors with a custom HTML page.
    """
    if exc.status_code == 404:
        return templates.TemplateResponse("404.html", {"request": request}, status_code=404)
    return JSONResponse(content={"detail": exc.detail}, status_code=exc.status_code)

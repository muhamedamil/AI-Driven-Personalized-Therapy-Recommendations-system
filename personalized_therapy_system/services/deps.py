"""
Module: dependencies.py
Description:
    Provides reusable and singleton-based dependencies for FastAPI routes,
    including memory service, vector store, LLM integration, and RAG pipeline.

    These dependencies are injected across routes and services to ensure consistency,
    performance, and single instantiation where appropriate.
Created: 2025-06-25
Last Modified: 2025-07-08
"""

import os
from typing import Optional
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project-specific imports
from services.memory_service import MemoryService
from services.openrouter_llm import OpenRouterLLM
from rag.vector_store import VectorStoreService
from rag.utils.similarity_filter import SimilarityFilter
from rag.rag_pipeline import RAGPipeline
from services.llm_service import LLMService
from database.db_config import get_async_db
from rag.utils.config import settings


# -------------------------
# Singleton Instances
# -------------------------
_vectorstore_instance: Optional[VectorStoreService] = None
memory_service = MemoryService()
similarity_filter = SimilarityFilter()


# -------------------------
# Dependency Providers
# -------------------------

def get_memory_service() -> MemoryService:
    """
    Returns the shared memory service instance.
    """
    return memory_service


def get_similarity_filter() -> SimilarityFilter:
    """
    Returns the shared similarity filter instance.
    """
    return similarity_filter


async def get_vectorstore() -> VectorStoreService:
    """
    Initializes and returns a singleton vector store instance.
    """
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = await VectorStoreService.create()
    return _vectorstore_instance


def get_llm_model() -> OpenRouterLLM:
    """
    Initializes the OpenRouter LLM using the API key from environment.
    Raises:
        ValueError: if OPENROUTER_API_KEY is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENROUTER_API_KEY not set in environment.")
    return OpenRouterLLM(api_key=api_key)


async def get_rag_pipeline(
    memory_service: MemoryService        = Depends(get_memory_service),
    similarity_filter: SimilarityFilter  = Depends(get_similarity_filter),
    vectorstore: VectorStoreService      = Depends(get_vectorstore),
) -> RAGPipeline:
    """
    Constructs and returns a configured RAGPipeline.
    """
    return RAGPipeline(
        memory_service=memory_service,
        vectorstore=vectorstore,
        similarity_filter=similarity_filter,
    )


async def get_llm_service(
    db: AsyncSession                     = Depends(get_async_db),
    memory_service: MemoryService       = Depends(get_memory_service),
    rag_pipeline: RAGPipeline           = Depends(get_rag_pipeline),
    llm_model: OpenRouterLLM            = Depends(get_llm_model),
) -> LLMService:
    """
    Constructs and returns the LLMService.

    Args:
        db (AsyncSession): Active DB session.
        memory_service (MemoryService): Memory buffer instance.
        rag_pipeline (RAGPipeline): Retrieval-augmented pipeline.
        llm_model (OpenRouterLLM): Configured LLM client.

    Returns:
        LLMService: Ready-to-use language model service.
    """
    return LLMService(
        memory_service=memory_service,
        rag_pipeline=rag_pipeline,
        llm=llm_model,
    )

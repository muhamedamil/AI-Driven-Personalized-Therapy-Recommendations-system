"""
Module: rag_pipeline.py
Description: Executes a full Retrieval-Augmented Generation (RAG) pipeline with optional
             query expansion, classification, relevance filtering, and fallback handling.
Created: 2025-06-13
Last Modified: 2025-07-08
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict, Tuple
from langchain.schema import Document
from logging.handlers import RotatingFileHandler

from rag.vector_store import VectorStoreService
from rag.query_expansion import QueryExpander
from rag.utils.similarity_filter import SimilarityFilter
from services.memory_service import MemoryService
from rag.rag_use_classifier import RAGUseClassifier, RAGDecision, VagueDecision
from rag.utils.fallback_handler import FallbackHandler, FallbackType
from rag.utils.config import settings

# ---------------------------
# Logging Configuration
# ---------------------------
LOG_DIR = "logs"
LOG_FILE = "rag_pipeline.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)
os.makedirs(LOG_DIR, exist_ok=True)

file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# ---------------------------
# RAG Pipeline Class
# ---------------------------

class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation workflow with optional query expansion.

    Responsibilities:
        - Classify whether RAG is needed and whether the query is vague.
        - Expand vague queries using LLMs.
        - Retrieve and filter relevant documents using a vector store.
        - Provide final decision and output for downstream processing.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        vectorstore: VectorStoreService,
        similarity_filter: SimilarityFilter,
    ):
        self.memory_service = memory_service
        self.query_expander = QueryExpander(memory_service=memory_service)
        self.vectorstore = vectorstore
        self.similarity_filter = similarity_filter
        self.classifier = RAGUseClassifier()

    @classmethod
    async def create(
        cls,
        memory_service: MemoryService,
        similarity_filter: SimilarityFilter,
    ) -> "RAGPipeline":
        """
        Factory method to initialize the pipeline with vectorstore dependency.

        :param memory_service: Instance of MemoryService
        :param similarity_filter: Similarity filter for document relevance
        :return: Initialized RAGPipeline object
        """
        vectorstore = await VectorStoreService.create(settings.VECTOR_COLLECTION_NAME)
        return cls(memory_service, vectorstore, similarity_filter)

    async def run(
        self,
        db,
        query: str,
        session_id: Optional[str],
        user_id: str,
        illness: Optional[str] = None,
        *,
        skip_processing: bool = False,
    ) -> Dict[str, object]:
        """
        Execute the full RAG pipeline including classification, expansion, and retrieval.

        :param db: DB session/connection
        :param query: Raw user query
        :param session_id: User session ID
        :param user_id: User ID
        :param illness: Optional illness tag
        :param skip_processing: If True, skip classification and expansion
        :return: Dict containing final_query, use_rag flag, and documents
        """
        try:
            final_query = query
            documents: List[Document] = []
            use_rag = False

            if not skip_processing:
                try:
                    rag_decision, vague_decision = await self.classifier.classify(query)
                    logger.info(f"[RAGPipeline] RAG decision: {rag_decision}, Vague decision: {vague_decision}")
                except Exception as e:
                    return FallbackHandler.handle(
                        fallback_type=FallbackType.CLASSIFIER_FAILURE,
                        default_value={"final_query": query, "use_rag": False, "documents": []},
                        context={"query": query},
                        error=e,
                        user_message="Failed to classify the query."
                    )

                if vague_decision == VagueDecision.VAGUE:
                    try:
                        final_query = await self.query_expander.expand_query(
                            db=db,
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                            illness=illness
                        )
                        logger.info(f"[RAGPipeline] Expanded query: {final_query}")
                    except Exception as e:
                        return FallbackHandler.handle(
                            fallback_type=FallbackType.QUERY_EXPANSION_FAILURE,
                            default_value={"final_query": query, "use_rag": False, "documents": []},
                            context={"query": query, "session_id": session_id},
                            error=e,
                            user_message="Query expansion failed."
                        )

                if rag_decision == RAGDecision.NO_RAG and vague_decision == VagueDecision.CLEAR:
                    logger.info("[RAGPipeline] Clear query, no RAG needed. Skipping retrieval.")
                    return {"final_query": final_query, "use_rag": False, "documents": []}

                retrieval_condition = (rag_decision == RAGDecision.RAG)
            else:
                retrieval_condition = True

            if retrieval_condition:
                try:
                    results = await self.vectorstore.similarity_search_with_score(final_query)
                    logger.info(f"[RAGPipeline] Retrieved {len(results)} documents for query: '{final_query}'")

                    if await self.similarity_filter.is_query_relevant(results):
                        documents = await self.similarity_filter.filter_documents(results)
                        use_rag = True
                        logger.info("[RAGPipeline] Using RAG with relevant documents.")
                    else:
                        logger.info("[RAGPipeline] Retrieved documents not relevant. Skipping RAG.")
                except Exception as e:
                    return FallbackHandler.handle(
                        fallback_type=FallbackType.VECTOR_SEARCH_FAILURE,
                        default_value={"final_query": final_query, "use_rag": False, "documents": []},
                        context={"query": final_query},
                        error=e,
                        user_message="Vector search failed."
                    )

            return {"final_query": final_query, "use_rag": use_rag, "documents": documents}

        except Exception as e:
            return FallbackHandler.handle(
                fallback_type=FallbackType.GENERAL_FALLBACK,
                default_value={"final_query": query, "use_rag": False, "documents": []},
                context={"query": query},
                error=e,
                user_message="Something went wrong while processing the query."
            )

    async def retrieve_only(self, final_query: str) -> Tuple[bool, List[Document]]:
        """
        Retrieve relevant documents without classification or expansion.

        :param final_query: Expanded or raw query to search for
        :return: Tuple of use_rag flag and list of filtered documents
        """
        try:
            results = await self.vectorstore.similarity_search_with_score(final_query)
            if await self.similarity_filter.is_query_relevant(results):
                docs = await self.similarity_filter.filter_documents(results)
                return True, docs
        except Exception as e:
            logger.warning(f"[RAGPipeline] retrieve_only failed for query '{final_query}': {e}")
        return False, []

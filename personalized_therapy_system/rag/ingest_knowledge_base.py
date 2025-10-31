"""
Script: ingest_knowledge_base.py
Description: Asynchronous script to ingest documents into the RAG knowledge base.
Created: 2025-06-10
"""

import logging
import asyncio
from rag.knowledge_base import KnowledgeBaseService

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Ingest Documents into Vector Store
# -------------------------------
async def run_knowledge_base_pipeline():
    """
    Loads and processes knowledge base documents for ingestion into a vector store.
    """
    collection_name = "psychology_knowledge_base"
    documents_path = "knowledge_base"

    try:
        kb_service = KnowledgeBaseService(collection_name=collection_name)
        await kb_service.process_documents(documents_path)
        logger.info(f"Successfully processed and stored documents from: {documents_path}")
    except Exception as e:
        logger.exception("Failed to process knowledge base documents.")


if __name__ == "__main__":
    asyncio.run(run_knowledge_base_pipeline())

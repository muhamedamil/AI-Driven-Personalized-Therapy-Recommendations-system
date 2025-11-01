import asyncio
import logging
from typing import List, Optional

from sqlalchemy import text
from database.db_config import engine

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_postgres.vectorstores import PGVector, DistanceStrategy

from rag.utils.config import settings
from rag.utils.exceptions import VectorStoreInitializationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VectorStoreService:
    """
    Manages PGVector-based vector storage and retrieval using LangChain.

    Responsibilities:
    - Ensures pgvector extension is present.
    - Initializes PGVector with HuggingFace embeddings.
    - Adds documents to the vector store.
    - Performs similarity search.
    - Verifies if a document (based on metadata source) exists.
    """

    def __init__(self, embedding_model_name: str, vectorstore: PGVector):
        self.collection_name = settings.VECTOR_COLLECTION_NAME
        self.embedding_model_name = embedding_model_name
        self.vectorstore = vectorstore

    @classmethod
    async def create(cls, embedding_model: str) -> "VectorStoreService":
        """
        Initializes the PGVector collection and returns a service instance.

        Args:
            embedding_model (str): Name of HuggingFace embedding model.

        Returns:
            VectorStoreService: The initialized service.
        """
        try:
            # Ensure the pgvector extension exists
            async with engine.begin() as conn:
                await conn.run_sync(lambda sync_conn:
                    sync_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                )
            logger.info("[VectorStore] Ensured 'vector' extension exists.")

            huggingface_embeddings = HuggingFaceEmbeddings(model=embedding_model)

            sync_url = settings.PGVECTOR_URL
            logger.debug(f"Converted sync URL for PGVector: {sync_url}")

            # Instantiate PGVector (run in thread)
            def _make_store():
                return PGVector(
                    connection=str(sync_url),
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    embeddings=huggingface_embeddings,
                    distance_strategy=DistanceStrategy.COSINE,
                    use_jsonb=True
                )

            vectorstore = await asyncio.to_thread(_make_store)
            logger.info(f"[VectorStore] PGVector initialized for collection: {settings.VECTOR_COLLECTION_NAME}")

            return cls(embedding_model, vectorstore)

        except Exception as e:
            logger.error("[VectorStore] Initialization failed", exc_info=True)
            raise VectorStoreInitializationError(f"Failed to initialize VectorStore: {e}")

    async def add_documents(self, docs: List[Document]) -> None:
        """
        Adds a list of documents to the vector store.

        Args:
            docs (List[Document]): Documents to add.
        """
        if not docs:
            logger.warning("[VectorStore] No documents to add.")
            return
        try:
            await asyncio.to_thread(self.vectorstore.add_documents, docs)
            logger.info(f"[VectorStore] Added {len(docs)} docs to '{self.collection_name}'")
        except Exception as e:
            logger.error(f"[VectorStore] Error adding docs: {e}")
            raise

    async def similarity_search_with_score(
        self, query: str, k: int = 3, filters: Optional[dict] = None
    ) -> List[tuple]:
        """
        Performs similarity search using PGVector.

        Args:
            query (str): The search query.
            k (int): Number of top results to retrieve.
            filters (Optional[dict]): Optional metadata filters.

        Returns:
            List[tuple]: List of (Document, score) tuples.
        """
        try:
            return await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score, query, k, filters
            )
        except Exception as e:
            logger.error(f"[VectorStore] Similarity search failed: {e}")
            return []

    async def delete_collection(self) -> None:
        """
        Deletes the entire vector store collection.
        """
        try:
            await asyncio.to_thread(self.vectorstore.delete_collection)
            logger.info(f"[VectorStore] Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"[VectorStore] Error deleting collection: {e}")
            raise

    async def get_vectorstore(self) -> PGVector:
        """
        Returns the PGVector instance.
        """
        return self.vectorstore

    async def has_document(self, source: str) -> bool:
        """
        Checks if any document in the vector store has a given source.

        Args:
            source (str): The source file path (from metadata).

        Returns:
            bool: True if at least one matching vector exists.
        """
        query = text("""
            SELECT COUNT(*) FROM langchain_pg_embedding
            WHERE collection_name = :collection_name
            AND metadata->>'source' = :source
        """)
        async with engine.connect() as conn:
            result = await conn.execute(query, {
                "collection_name": self.collection_name,
                "source": source
            })
            count = result.scalar_one()
            return count > 0

import logging
from typing import List, Tuple
from rag.utils.fallback_handler import FallbackHandler, FallbackType
from langchain.schema import Document  # Update if custom Document schema

logger = logging.getLogger(__name__)


class SimilarityFilter:
    def __init__(self, threshold: float = 0.8):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold

    async def is_query_relevant(self, results: List[Tuple[Document, float]]) -> bool:
        """
        Determines if any of the provided documents have a similarity score
        above the relevance threshold.

        Parameters:
        - results: List of tuples (Document, similarity_score)

        Returns:
        - True if relevant, False otherwise
        """
        if not results:
            logger.warning("[SimilarityFilter] No similar documents provided.")
            return await FallbackHandler.handle(
                fallback_type=FallbackType.EMPTY_SIMILARITY_RESULT,
                default_value=False,
                context={},
                user_message="No similar documents found."
            )

        top_score = max(score for _, score in results)
        logger.debug(f"[SimilarityFilter] Top similarity score: {top_score}")

        if top_score >= self.threshold:
            logger.info(f"[SimilarityFilter] Query is relevant with score: {top_score}")
            return True
        else:
            logger.info(f"[SimilarityFilter] Query irrelevant (score: {top_score} < threshold: {self.threshold})")
            return await FallbackHandler.handle(
                fallback_type=FallbackType.LOW_SIMILARITY_SCORE,
                default_value=False,
                context={"top_score": top_score},
                user_message="The documents found weren’t relevant enough."
            )

    async def filter_documents(self, results: List[Tuple[Document, float]]) -> List[Document]:
        """
        Filters out documents below the threshold.

        Parameters:
        - results: List of tuples (Document, similarity_score)

        Returns:
        - List of Documents that meet or exceed the threshold
        """
        filtered = [doc for doc, score in results if score >= self.threshold]
        logger.info(f"[SimilarityFilter] {len(filtered)} documents passed the similarity threshold of {self.threshold}.")
        return filtered

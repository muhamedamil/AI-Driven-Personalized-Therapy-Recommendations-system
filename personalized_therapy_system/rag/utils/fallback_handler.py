import logging
import os
from enum import Enum
from typing import Optional, Any
import asyncio

# Set up logging
dir_path = os.getcwd()
log_dir = os.path.join(dir_path, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "fallback.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers during imports
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

class FallbackType(str, Enum):
    CLASSIFIER_FAILURE = "Classifier Failure"
    QUERY_EXPANSION_FAILURE = "Query Expansion Failure"
    VECTOR_SEARCH_FAILURE = "Vector Search Failure"
    EMPTY_SIMILARITY_RESULT = "Empty Similarity Result"
    LOW_SIMILARITY_SCORE = "Low Similarity Score"
    GENERAL_FALLBACK = "General Fallback"

class FallbackHandler:
    @staticmethod
    async def handle(
        fallback_type: FallbackType,
        default_value: Any,
        context: Optional[dict] = None,
        error: Optional[Exception] = None,
        user_message: Optional[str] = None
    ) -> Any:
        """
        Handles different fallback scenarios and logs necessary details.
        Returns a safe default value.
        """

        message = f"[FallbackHandler] {fallback_type.value} triggered."
        if context:
            message += f" | Context: {context}"

        # Synchronous logging function
        def log_sync():
            if error:
                message_with_error = message + f" | Error: {error!s}"
                logger.exception(message_with_error)
            else:
                logger.warning(message)

            if user_message:
                logger.info(f"User Fallback Message: {user_message}")

        # Run logging in a thread to avoid blocking event loop
        await asyncio.to_thread(log_sync)

        return default_value

"""
Module: analysis_tools.py
Description: Asynchronous analysis utilities for illness detection, intent recognition, and response style classification.
Created: 2025-06-28
Last Modified: 2025-07-08
"""

import asyncio
import logging
from typing import Any, Dict

from services.illness_service import IllnessService
from services.intent_service import intent_recognition
from services.ai_response_style_service import ResponseStyleService

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# Initialize model instances (singleton pattern could be applied if needed)
_illness_service = IllnessService()
_style_service = ResponseStyleService()

__all__ = [
    'illness_detector',
    'intent_detector',
    'response_style_classifier',
    'full_analysis',
]


def _validate_input(text: str, task_name: str) -> None:
    """
    Ensure the provided text input is a non-empty string.

    :param text: The input text to validate.
    :param task_name: Contextual name of the task for error messages.
    :raises ValueError: If text is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"'{task_name}' input must be a non-empty string.")


def _handle_exception(result: Any) -> Any:
    """
    Convert exceptions returned from asyncio.gather into error dicts.

    :param result: The result which might be an exception.
    :returns: Original result or an error dictionary.
    """
    if isinstance(result, Exception):
        return {'error': str(result)}
    return result


async def illness_detector(text: str) -> Dict[str, Any]:
    """
    Predict illness from the input text.

    :param text: User input text describing symptoms.
    :returns: Dictionary containing illness prediction or error.
    """
    task = 'Illness detection'
    try:
        _validate_input(text, task)
        logger.info(f"{task} started.")
        prediction = _illness_service.predict_illness(text)
        return {'prediction': prediction}
    except Exception as exc:
        logger.error(f"{task} failed: {exc}")
        return {'error': f"{task} failed: {exc}"}


async def intent_detector(text: str) -> Dict[str, Any]:
    """
    Recognize intent category from the input text.

    :param text: User input text.
    :returns: Dictionary with intent category or error message.
    """
    task = 'Intent detection'
    try:
        _validate_input(text, task)
        logger.info(f"{task} started.")
        intent = await intent_recognition(text)
        return {'intent_category': intent}
    except Exception as exc:
        logger.error(f"{task} failed: {exc}")
        return {'error': f"{task} failed: {exc}"}


async def response_style_classifier(text: str) -> Dict[str, Any]:
    """
    Classify the style of response from the input text.

    :param text: User input text.
    :returns: Dictionary with style classification or error.
    """
    task = 'Response style classification'
    try:
        _validate_input(text, task)
        logger.info(f"{task} started.")
        style = _style_service.classify_response_style(text)
        return {'style': style}
    except Exception as exc:
        logger.error(f"{task} failed: {exc}")
        return {'error': f"{task} failed: {exc}"}


async def full_analysis(user_input: str) -> Dict[str, Any]:
    """
    Perform combined analysis: illness detection, intent recognition, and response style.

    :param user_input: The input text to analyze.
    :returns: Dictionary containing all analysis results or errors.
    """
    task = 'Full analysis'
    try:
        _validate_input(user_input, task)
        logger.info(f"{task} started.")

        results = await asyncio.gather(
            _illness_service.predict_illness(user_input),
            intent_recognition(user_input),
            _style_service.classify_response_style(user_input),
            return_exceptions=True
        )

        return {
            'illness_prediction': _handle_exception(results[0]),
            'intent_category': _handle_exception(results[1]),
            'response_style': _handle_exception(results[2])
        }
    except Exception as exc:
        logger.error(f"{task} failed: {exc}")
        return {'error': f"{task} failed: {exc}"}



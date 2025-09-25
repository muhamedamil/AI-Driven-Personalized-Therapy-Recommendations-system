# -*- coding: utf-8 -*-
"""
Module: mental_health_nodes.py
Description: Defines LangGraph processing nodes for intent detection, response style classification,
             and illness prediction in a mental health assistant pipeline.
Created: 2025-07-1
Last Modified: 2025-07-08
History:
    - 2025-07-1: Initial implementation of analysis nodes.
    - 2025-07-08: Refactored to include structured logging, typing, and error handling.
"""

import logging
from typing import TypedDict, Optional, Dict

from agents.analysis_tools import (
    intent_detector,
    response_style_classifier,
    illness_detector
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# ---------------------------
# 1. LangGraph State Schema
# ---------------------------
class MentalHealthState(TypedDict):
    user_input: str
    intent_category: Optional[str]
    response_style: Optional[str]
    illness_prediction: Optional[str]


# ---------------------------
# 2. Tool Nodes with Logging
# ---------------------------
async def detect_intent_node(state: MentalHealthState) -> MentalHealthState:
    """
    Detects the user's intent from input.

    :param state: Current state dictionary with user_input.
    :return: Updated state with intent_category.
    """
    user_input = state["user_input"]
    logger.info("Detecting intent...")
    try:
        result = await intent_detector(user_input)
        intent = result.get("intent_category", "unknown")
        logger.info(f"Intent detected: {intent}")
        return {**state, "intent_category": intent}
    except Exception as e:
        logger.warning(f"Intent detection failed: {e}")
        return {**state, "intent_category": "unknown"}


async def detect_response_style_node(state: MentalHealthState) -> MentalHealthState:
    """
    Detects the user's response style from input.

    :param state: Current state dictionary with user_input.
    :return: Updated state with response_style.
    """
    user_input = state["user_input"]
    logger.info("Detecting response style...")
    try:
        style_result = await response_style_classifier(user_input)
        style = style_result.get("style") or style_result.get("response_style", "supportive")
        logger.info(f"Response style: {style}")
        return {**state, "response_style": style}
    except Exception as e:
        logger.warning(f"Style detection failed: {e}")
        return {**state, "response_style": "supportive"}


async def detect_illness_node(state: MentalHealthState) -> MentalHealthState:
    """
    Predicts signs of mental illness from user input.

    :param state: Current state dictionary with user_input.
    :return: Updated state with illness_prediction.
    """
    user_input = state["user_input"]
    logger.info("Checking for illness signs (conditionally)...")
    try:
        illness_result = await illness_detector(user_input)
        illness = illness_result.get("prediction", "none")
        logger.info(f"Illness detected: {illness}")
        return {**state, "illness_prediction": illness}
    except Exception as e:
        logger.warning(f"Illness detection failed: {e}")
        return {**state, "illness_prediction": "none"}


# ---------------------------
# 3. Final Output Node
# ---------------------------
def format_json_output(state: MentalHealthState) -> Dict[str, str]:
    """
    Formats the final output into a clean dictionary.

    :param state: The final state after processing.
    :return: Dictionary with keys: intent_category, response_style, illness_prediction.
    """
    output = {
        "intent_category": state.get("intent_category", "unknown"),
        "response_style": state.get("response_style", "supportive"),
        "illness_prediction": state.get("illness_prediction", "none")
    }
    logger.info(f"Final output: {output}")
    return output




"""
Module: mental_health_graph.py
Description: Constructs a LangGraph pipeline for analyzing user mental health input through intent, response style, and optional illness classification.
Author: TechTeam AI Labs
Created: 2025-06-30
Last Modified: 2025-07-08
History:
    - 2025-06-30: Initial implementation with LangGraph steps.
    - 2025-07-08: Integrated Groq-based LLM routing and modularized steps.
"""

import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, Runnable

from agents.mental_health_nodes import (
    detect_intent_node,
    detect_response_style_node,
    detect_illness_node,
    format_json_output,
    MentalHealthState
)

from agents.llm_illness_classifier import LLMIntentBasedIllnessClassifier

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# Initialize illness classifier
llm_classifier = LLMIntentBasedIllnessClassifier()

# ---------------------------
# Router for Illness Decision
# ---------------------------
async def route_illness_check(state: MentalHealthState) -> str:
    """
    Determines the next step using LLM-based illness classification.

    :param state: Current MentalHealthState containing user_input
    :return: 'detect_illness' if analysis is needed, otherwise 'output'
    """
    user_input = state["user_input"]
    try:
        should_check = await llm_classifier.needs_illness_check(user_input)
        if should_check:
            logger.info("LLM decided to perform illness detection.")
            return "detect_illness"
        else:
            logger.info("LLM skipped illness detection.")
            return "output"
    except Exception as e:
        logger.warning(f" Illness routing failed: {e}. Defaulting to skip.")
        return "output"

# ---------------------------
# LangGraph Graph Definition
# ---------------------------
def build_mental_health_graph() -> RunnableLambda:
    """
    Constructs the mental health assistant pipeline using LangGraph.

    Steps:
    - detect_intent → detect_style → route to detect_illness or output
    - Optional: perform illness detection if LLM deems it necessary
    - Final output formatting

    :return: Compiled LangGraph Runnable
    """
    workflow = StateGraph(MentalHealthState)

    # Step-by-step pipeline
    workflow.add_node("detect_intent", detect_intent_node)
    workflow.add_node("detect_style", detect_response_style_node)
    workflow.add_node("detect_illness", detect_illness_node)
    workflow.add_node("output", format_json_output)

    # LLM-based illness routing
    workflow.add_conditional_edges(
        "detect_style",
        route_illness_check,
        {
            "detect_illness": "detect_illness",
            "output": "output"
        }
    )

    # Transitions
    workflow.set_entry_point("detect_intent")
    workflow.add_edge("detect_intent", "detect_style")
    workflow.add_edge("detect_illness", "output")
    workflow.set_finish_point("output")

    logger.info("LangGraph mental health pipeline built.")
    return workflow.compile()


async def get_mental_health_graph() -> Runnable:
    """
    Asynchronously returns the compiled LangGraph Runnable.

    :return: Runnable instance representing the mental health assistant
    """
    return build_mental_health_graph()

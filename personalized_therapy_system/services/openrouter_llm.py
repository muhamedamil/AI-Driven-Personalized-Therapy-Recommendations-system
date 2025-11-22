"""
Module: openrouter_llm.py
Description:
    Custom implementation of a LangChain-compatible LLM wrapper using OpenRouter's
    chat completions API. Supports both standard and streaming modes for interacting
    with LLMs like Qwen, Mistral, etc.

Author: Muhammed Amil
Created: 2025-07-08
"""

import json
from typing import Optional, List, Dict, Any, Union, AsyncGenerator

import httpx
import requests as _requests
from pydantic import Field

from langchain.chat_models.base import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain.schema import AIMessage
from langchain.schema.output import ChatGeneration, ChatResult

import logging

logger = logging.getLogger(__name__)

_VALID_OPENAI_ROLES = ("system", "user", "assistant")


class OpenRouterLLM(BaseChatModel):
    """
    A custom LangChain-compatible LLM class that integrates with OpenRouter's
    chat API. It supports both standard and streaming chat completion modes.

    Attributes:
        api_key (str): OpenRouter API key for authorization.
        api_url (str): API endpoint for chat completions.
        model_name (str): Model to use (e.g., "qwen/qwen3-32b:free").
        streaming (bool): If True, enables streaming response.
    """

    api_key: str = Field(..., description="OpenRouter API key")
    api_url: str = Field(
        "https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter API endpoint"
    )
    model_name: str = Field(
        "meta-llama/llama-3.2-3b-instruct:free",
        description="Model name registered on OpenRouter"
    )
    streaming: bool = True  # Enable/disable streaming completions

    # --- Helpers -------------------------------------------------------------

    def _serialize_message(self, msg: Union[BaseMessage, Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """
        Converts a LangChain message or dictionary into OpenRouter-compatible format.
        Normalizes roles and drops empty/invalid messages.
        """
        if isinstance(msg, BaseMessage):
            # Map LangChain types -> OpenAI roles
            role_map = {
                "system": "system",
                "human": "user",
                "ai": "assistant",
                "chat": "assistant",  # fallback
            }
            role = role_map.get(getattr(msg, "type", ""), "user")
            content = (getattr(msg, "content", "") or "").strip()
            if not content:
                return None
            return {"role": role, "content": content}

        elif isinstance(msg, dict):
            # Normalize dict payloads as well
            content = (msg.get("content") or "").strip()
            if not content:
                return None
            role = msg.get("role", "user")
            # fix non-OpenAI roles if they sneak in
            if role not in _VALID_OPENAI_ROLES:
                role = {
                    "human": "user",
                    "ai": "assistant",
                    "assistant": "assistant",
                    "user": "user",
                    "system": "system",
                }.get(role, "user")
            return {"role": role, "content": content}

        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

    def _normalize_messages(self, messages: List[Union[BaseMessage, Dict[str, Any], str]]) -> List[Dict[str, str]]:
        """
        Ensures all messages are valid OpenRouter/OpenAI Chat API messages and removes empties.
        Strings are wrapped as user messages.
        """
        normalized: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, str):
                m = {"role": "user", "content": m}
            nm = self._serialize_message(m)
            if nm is not None:
                normalized.append(nm)
        return normalized

    # --- Core generation -----------------------------------------------------

    def _generate(
        self,
        input: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        """
        Generate a response using OpenRouter (non-async, optionally streaming).

        Args:
            input (str): The user prompt.
            stop (List[str], optional): Stop sequences.
            run_manager (CallbackManagerForLLMRun, optional): For streaming callbacks.

        Returns:
            ChatResult: The result object with generations.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Default starting messages
        messages: List[Union[BaseMessage, Dict[str, Any], str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ]

        # User-provided messages override/augment defaults
        if "messages" in kwargs:
            raw_messages = kwargs.pop("messages")

            if isinstance(raw_messages, list) and all(isinstance(m, dict) for m in raw_messages):
                messages = raw_messages
            elif isinstance(raw_messages, list) and all(isinstance(m, BaseMessage) for m in raw_messages):
                messages = raw_messages
            elif isinstance(raw_messages, dict):
                messages = [raw_messages]
            elif isinstance(raw_messages, list) and all(isinstance(m, str) for m in raw_messages):
                messages = [{"role": "user", "content": m} for m in raw_messages]
            elif isinstance(raw_messages, str):
                messages = [{"role": "user", "content": raw_messages}]
            else:
                raise ValueError(
                    "messages must be a list of {role, content} dicts, BaseMessage objects, or strings"
                )

        # NEW: normalize + drop empties / wrong roles
        messages = self._normalize_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.streaming,
            **kwargs,
        }
        if stop:
            payload["stop"] = stop

        logger.debug("==== DEBUG START ====")
        logger.debug("API URL: %r", self.api_url)
        logger.debug("Model: %s", self.model_name)
        logger.debug("Payload:\n%s", json.dumps(payload, indent=2))
        logger.debug("==== DEBUG END ====")

        if self.streaming:
            return self._streaming_call(headers, payload, run_manager)
        else:
            response = _requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code != 200:
                # surface the API error text for easier debugging
                raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _streaming_call(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        run_manager: Optional[CallbackManagerForLLMRun]
    ) -> ChatResult:
        """
        Handles synchronous streaming responses via requests.

        Returns:
            ChatResult: The final response content.
        """
        buffer = ""
        full_content = ""

        with _requests.post(self.api_url, headers=headers, json=payload, stream=True) as r:
            # NEW: handle non-200 up front (otherwise iter_content hangs or hides errors)
            if r.status_code != 200:
                try:
                    err_text = r.text
                except Exception:
                    err_text = "<no body>"
                raise RuntimeError(f"OpenRouter API error {r.status_code}: {err_text}")

            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                while True:
                    line_end = buffer.find('\n')
                    if line_end == -1:
                        break

                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]

                    if not line:
                        continue

                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            data_obj = json.loads(data)
                            delta = data_obj["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                if run_manager:
                                    run_manager.on_llm_new_token(content)
                        except json.JSONDecodeError:
                            # ignore keepalives / partials
                            continue

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=full_content))])

    async def stream(
        self,
        input: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming interface for LangChain.

        Args:
            input (str): The user prompt.
            stop (List[str], optional): Stop sequences.
            run_manager (CallbackManagerForLLMRun, optional): Streaming handler.

        Yields:
            str: Token content as it streams in.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        messages: List[Union[BaseMessage, Dict[str, Any], str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ]

        if "messages" in kwargs:
            raw_messages = kwargs.pop("messages")

            if isinstance(raw_messages, list) and all(isinstance(m, dict) for m in raw_messages):
                messages = raw_messages
            elif isinstance(raw_messages, list) and all(isinstance(m, BaseMessage) for m in raw_messages):
                messages = raw_messages
            elif isinstance(raw_messages, dict):
                messages = [raw_messages]
            elif isinstance(raw_messages, list) and all(isinstance(m, str) for m in raw_messages):
                messages = [{"role": "user", "content": m} for m in raw_messages]
            elif isinstance(raw_messages, str):
                messages = [{"role": "user", "content": raw_messages}]
            else:
                raise ValueError(
                    "messages must be a list of {role, content} dicts, BaseMessage objects, or strings"
                )

        # NEW: normalize + drop empties / wrong roles
        messages = self._normalize_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if stop:
            payload["stop"] = stop

        logger.debug("==== DEBUG START ====")
        logger.debug("API URL: %r", self.api_url)
        logger.debug("Model: %s", self.model_name)
        logger.debug("Payload:\n%s", json.dumps(payload, indent=2))
        logger.debug("==== DEBUG END ====")

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.api_url, headers=headers, json=payload) as response:
                # NEW: fail fast on non-200 and dump body
                if response.status_code != 200:
                    body = await response.aread()
                    try:
                        body_text = body.decode("utf-8", errors="replace")
                    except Exception:
                        body_text = "<unreadable body>"
                    raise RuntimeError(f"OpenRouter API error {response.status_code}: {body_text}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            data_obj = json.loads(data)
                            delta = data_obj["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                if run_manager:
                                    run_manager.on_llm_new_token(content)
                                yield content
                        except json.JSONDecodeError:
                            continue

    @property
    def _identifying_params(self) -> Dict:
        """
        Required override for LangChain's internal caching/config handling.
        """
        return {
            "api_url": self.api_url,
            "model_name": self.model_name,
            "streaming": self.streaming
        }

    @property
    def _llm_type(self) -> str:
        """
        Required override to identify LLM class type.
        """
        return "openrouter"

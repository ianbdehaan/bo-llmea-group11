"""LLM manager to connect to the model.
"""
# Group 11 change: the original LLaMEA-BO file had clients for many api providers (OpenAI,
# Groq, Gemini, OpenRouter, ...). We only use a local model served by Ollama, so this file
# keeps the LLaMEA-BO interface (LLMClientResponse, LLMClient, LLMmanager.chat) with a single
# Ollama client.
import time
import logging
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable
import requests


class LLMClientResponse:
    def __init__(self, response):
        self.response = response
        self.text = None
        self._prompt_token_count = 0
        self._response_token_count = 0
        self.error = None
        self.finish_reason = None
        # reasoning text of thinking models (e.g. qwen3 with think=True), only kept for the logs
        self.thinking = None

    @property
    def prompt_token_count(self):
        return self._prompt_token_count

    @prompt_token_count.setter
    def prompt_token_count(self, value):
        if value is not None:
            self._prompt_token_count = value

    @property
    def response_token_count(self):
        return self._response_token_count

    @response_token_count.setter
    def response_token_count(self, value):
        if value is not None:
            self._response_token_count = value

    def __str__(self):
        if self.error is not None:
            return str(self.error)
        return str(self.response)


class LLMClient(ABC):
    """Abstract base class for LLM backends."""
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

    @abstractmethod
    def raw_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMClientResponse:
        """Generate a raw completion from the LLM."""

    @property
    def name(self) -> str:
        return f"{self.model_name}"


class OllamaClient(LLMClient):
    # Uses the native Ollama api (/api/chat). The OpenAI compatible endpoint of Ollama ignores
    # the context length, so long prompts (population summary + code) would be cut silently
    # to the default context. Here num_ctx is passed explicitly.
    def __init__(self, model_name, base_url, num_ctx, think=None):
        super().__init__(model_name, base_url)
        self.num_ctx = num_ctx
        # True/False switches the thinking mode of models such as qwen3 on/off, None = model default
        self.think = think

    def raw_completion(self, messages, **kwargs):
        options = {"num_ctx": self.num_ctx}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]
        payload = {"model": self.model_name, "messages": messages, "stream": False, "options": options}
        if self.think is not None:
            payload["think"] = self.think
        try:
            response = requests.post(self.base_url + "/api/chat", json=payload, timeout=kwargs.get("timeout", 600))
            response.raise_for_status()
            json_res = response.json()
            res = LLMClientResponse(json_res)
            res.text = json_res["message"]["content"]
            res.thinking = json_res["message"].get("thinking")
            res.prompt_token_count = json_res.get("prompt_eval_count")
            res.response_token_count = json_res.get("eval_count")
            res.finish_reason = json_res.get("done_reason")
            # warn when the prompt fills the whole context (then ollama drops the start of the prompt)
            if res.prompt_token_count is not None and res.prompt_token_count >= self.num_ctx - 16:
                logging.warning("Prompt (%s tokens) reached the context size %s", res.prompt_token_count, self.num_ctx)
            return res
        except Exception as e:
            res = LLMClientResponse(None)
            res.error = e
            return res


class LLMmanager:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", num_ctx: int = 32768, think=None):
        self.client = OllamaClient(model_name, base_url, num_ctx, think)
        self.mock_res_provider: Callable[..., str] = None
        # how often and how long we wait when the request fails (server busy, timeout, ...)
        self.max_api_retries = 30
        self.max_wait_seconds = 3600

    def model_name(self) -> str:
        return self.client.name

    def chat(self, session_messages, **kwargs):
        if self.mock_res_provider is not None:
            _content = self.mock_res_provider(session_messages, **kwargs)
            res = LLMClientResponse(None)
            res.text = _content
            res.response_token_count = len(_content.split())
            return res

        logging.info("LLM: %s, %s", self.model_name(), kwargs)

        # Group 11 change: the original code returned a failed request directly, so it became a
        # failed candidate. We retry with exponential backoff, so only real answers count as candidates.
        wait_time = 10
        for i_api in range(self.max_api_retries):
            response = self.client.raw_completion(session_messages, **kwargs)
            if response.error is None:
                break
            logging.error("LLM: %s, attempt %s/%s, %s", self.model_name(), i_api + 1, self.max_api_retries, response.error)
            logging.info("Waiting %.0f seconds before calling the LLM again", wait_time)
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, self.max_wait_seconds)
        return response

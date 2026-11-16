"""LLM manager to connect to different types of models.
"""
import time
import os
import re
from datetime import datetime
from abc import ABC, abstractmethod
import logging
from typing import Optional, Any
from collections.abc import Callable
import requests
import openai
import aisuite as ai
from aisuite.providers.groq_provider import GroqMessageConverter
from google import genai
from google.genai import types


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL_FOR_OPENAI_CLIENT = os.getenv("GROQ_BASE_URL_FOR_OPENAI_CLIENT", "https://api.groq.com/openai/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
ONEHUB_API_KEY = os.getenv("ONEHUB_API_KEY")
ONEHUB_BASE_URL = os.getenv("ONEHUB_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
# a model served on our own machine (ollama, llama.cpp or vllm all offer an OpenAI compatible api)
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "local")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

LLMS = {
    # Group 11 change: gpt-oss-120b through Cerebras (OpenAI compatible api, larger free daily limit)
    "cerebras/gpt-oss-120b": ("gpt-oss-120b", CEREBRAS_API_KEY, CEREBRAS_BASE_URL, 2, "openai"),
    # Group 11 change: the same open-weight models served locally (ollama model names)
    "local/gpt-oss-120b": ("gpt-oss:120b", LOCAL_LLM_API_KEY, LOCAL_LLM_BASE_URL, 0, "openai"),
    "local/gpt-oss-20b": ("gpt-oss:20b", LOCAL_LLM_API_KEY, LOCAL_LLM_BASE_URL, 0, "openai"),
    # Group 11 change: Qwen3-Coder-30B-A3B (Q4_K_M) on the Ollama server of the machine (native api)
    "ollama/qwen3-coder": ("qwen3-coder:latest", None, OLLAMA_URL, 0, "ollama"),
    "openai/gpt-oss-120b": ("openai/gpt-oss-120b", GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 2, "groq"),
    "openai/gpt-oss-20b": ("openai/gpt-oss-20b", GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 2, "groq"),
    "qwen/qwen3.8-27b": ("qwen/qwen3.8-27b", GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 2, "groq"),

    'llama3-70b-8192': ('llama3-70b-8192', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 2, 'groq'),
    "llama-3.3-70b-versatile": ('llama-3.3-70b-versatile', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 15, 'groq'),
    'llama-4-maverick': ('meta-llama/llama-4-maverick-17b-128e-instruct', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 5, 'groq'),

    "gemini-2.0-flash-exp": ('gemini-2.0-flash-exp', GEMINI_API_KEY, GEMINI_BASE_URL, 10, 'google'),
    "gemini-2.0-flash": ('gemini-2.0-flash', GEMINI_API_KEY, GEMINI_BASE_URL, 10, 'google'),
    'gemini-2.5-flash': ('gemini-2.5-flash', GEMINI_API_KEY, GEMINI_BASE_URL, 10, 'google'),

    'onehub-gemini-2.0-flash': ('gemini-2.0-flash-exp', ONEHUB_API_KEY, ONEHUB_BASE_URL, 5, None),
    'onehub-gemma2-9b-it': ('gemma2-9b-it', ONEHUB_API_KEY, ONEHUB_BASE_URL, 5, None),

    'o_deepseek-r1-free': ('deepseek/deepseek-r1-0528:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
    'o_deepseek-r1': ('deepseek/deepseek-r1-0528', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),

    'o_qwen3-coder-free': ('qwen/qwen3-coder:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
    'o_qwen3-coder': ('qwen/qwen3-coder', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),

    'o_gpt4o': ('openai/gpt-4o-2024-11-20', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),

    'o_gemini-2.0-flash': ('google/gemini-2.0-flash-001', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),

    'gpt-4o': ('gpt-4o-2024-11-20', OPENAI_API_KEY, None, 5, 'openai'),
}

class LLMClientResponse:
    def __init__(self, response):
        self.response = response
        self.text = None
        self._prompt_token_count = 0
        self._response_token_count = 0
        self.error = None
        self.finish_reason = None

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
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    @abstractmethod
    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMClientResponse:
        """Generate a raw completion from the LLM.
        """

    @property
    def name(self) -> str:
        """Return the name of the LLM backend."""
        return f"{self.model_name}"

class GoogleGenAIClient(LLMClient):
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.client = genai.Client(api_key=api_key)

    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMClientResponse:
        """Generate a raw completion using Google's GenAI API."""
        try:
            sys_prompt = None
            user_contents = []
            for _msg in messages:
                if _msg['role'] == 'system':
                    sys_prompt = _msg['content']
                else:
                    user_contents.append(types.Part.from_text(text=_msg['content']))
                    user_contents.append(_msg['content'])

            response = self.client.models.generate_content(
                model=self.model_name,
                contents= user_contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    **kwargs
                ),
            )   
            res = LLMClientResponse(response)
            res.text = response.text
            res.prompt_token_count = response.usage_metadata.prompt_token_count
            res.response_token_count = response.usage_metadata.candidates_token_count
            return res
        except Exception as e:
            res = LLMClientResponse(None)
            res.error = e
            return res

class CustomGroqMessageConverter(GroqMessageConverter):
    @staticmethod
    def convert_response(response_data):

        res = GroqMessageConverter.convert_response(response_data)

        current_timestamp = time.time()
        setattr(res, "_response_timestamp", current_timestamp)

        AISuiteClient.raw_response_map[current_timestamp] = response_data

        return res
    
class AISuiteClient(LLMClient):
    # content should be deleted after use
    raw_response_map = {}
    
    def __init__(self, model_key: str):
        if model_key not in LLMS:
            raise ValueError(f"Invalid model key: {model_key}")
        _model = LLMS[model_key]
        api_key = _model[1]
        model_name = _model[0]
        base_url = _model[2]

        self.provider = _model[4]
        
        super().__init__(api_key, model_name, base_url)
        _config = {
            "groq": {
                "api_key": GROQ_API_KEY,
            },
        }
        
        self.client = ai.Client(provider_configs=_config)

        if self.provider == "groq":
            provider = self.client.providers.get(self.provider)
            provider.transformer = CustomGroqMessageConverter()

    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMClientResponse:
        """Generate a raw completion using AISuite's API."""
        res = None
        try:
            _aisuite_model = f"{self.provider}:{self.model_name}"
            response = self.client.chat.completions.create(
                model=_aisuite_model,
                messages=messages,
                **kwargs
            )

            res = LLMClientResponse(response)
            res.text = response.choices[0].message.content

            raw_response = None
            if getattr(response, "_response_timestamp", None) is not None:
                _response_timestamp = getattr(response, "_response_timestamp")
                raw_response = AISuiteClient.raw_response_map.pop(_response_timestamp, None)
                res.response = raw_response
                if self.provider == "groq":
                    res.prompt_token_count = raw_response["usage"]["prompt_tokens"]
                    res.response_token_count = raw_response["usage"]["completion_tokens"]

            return res
        except Exception as e:
            res = LLMClientResponse(None)
            res.error = e
            return res

class OpenAIClient(LLMClient):
    """OpenAI GPT backend implementation."""

    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMClientResponse:
        """Generate a raw completion using OpenAI's API."""
        res = None
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            res = LLMClientResponse(response)
            res.text = response.choices[0].message.content
            res.prompt_token_count = response.usage.prompt_tokens
            res.response_token_count = response.usage.completion_tokens
            # "length" means max_tokens was used up (for reasoning models often by the reasoning)
            res.finish_reason = response.choices[0].finish_reason
            return res
        except Exception as e:
            res = LLMClientResponse(None)
            res.error = e
            return res

class OllamaClient(LLMClient):
    # Group 11 change: talks to the native Ollama api (/api/chat). The OpenAI compatible endpoint of
    # Ollama ignores the context length, so long prompts (population summary + code) would be cut
    # silently to the default context. Here we pass num_ctx explicitly.
    def __init__(self, api_key, model_name, base_url=None):
        super().__init__(api_key, model_name, base_url)
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "32768"))

    def raw_completion(self, messages, **kwargs):
        options = {"num_ctx": self.num_ctx}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]
        if "seed" in kwargs:
            options["seed"] = kwargs["seed"]
        payload = {"model": self.model_name, "messages": messages, "stream": False, "options": options}
        try:
            response = requests.post(self.base_url + "/api/chat", json=payload, timeout=kwargs.get("timeout", 600))
            response.raise_for_status()
            json_res = response.json()
            res = LLMClientResponse(json_res)
            res.text = json_res["message"]["content"]
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

class RequestClient(LLMClient):
    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMClientResponse:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        res = None
        try:
            json_res = response.json()
            res = LLMClientResponse(json_res)
            res.text = json_res["choices"][0]["message"]["content"]
            if "usage" in json_res:
                res.prompt_token_count = json_res["usage"]["prompt_tokens"]
                res.response_token_count = json_res["usage"]["completion_tokens"]
            return res
        except Exception as e:
            res = LLMClientResponse(None)
            res.error = e
            return res

class LLMmanager:
    def __init__(self, model_key: str = None, model_name: str = None, api_key: str = None, base_url: str = None, client_str: str = None):
        if model_key is None:
            if model_name is None or api_key is None:
                raise ValueError("model_name and api_key must be provided.")

            _model = (model_name, api_key, base_url, 5, client_str)
        else:
            if model_key not in LLMS:
                raise ValueError(f"Invalid model key: {model_key}")

            _model = LLMS[model_key]

        api_key = _model[1]
        model_name = _model[0]
        base_url = _model[2]
        client_str = _model[4]

        if client_str == "openai":
            self.client = OpenAIClient(api_key, model_name, base_url)
        elif client_str == "openrouter":
            self.client = OpenAIClient(api_key, model_name, base_url)
        elif client_str == "groq":
            groq_url = base_url or "https://api.groq.com/openai/v1"
            self.client = OpenAIClient(api_key, model_name, groq_url)
        elif client_str == "request":
            self.client = RequestClient(api_key, model_name, base_url)
        elif client_str == "ollama":
            self.client = OllamaClient(api_key, model_name, base_url)
        elif client_str == "google":
            self.client = GoogleGenAIClient(api_key, model_name, base_url)
        else:
            self.client = AISuiteClient(model_key)

        self.max_interval = _model[3]
        self.mock_res_provider:Callable[..., str] = None
        # how often and how long we wait when the api returns an error (rate limit, timeout, ...)
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

        # Group 11 change: the original code returned API errors (e.g. rate limits) directly,
        # so a failed request immediately became a failed candidate. We now wait and retry
        # API errors, so only real model answers are counted as candidates.
        wait_time = 10
        for i_api in range(self.max_api_retries):
            response = self.client.raw_completion(
                session_messages,
                **kwargs
            )
            if response.error is None:
                break
            logging.error("LLM: %s, attempt %s/%s, %s", self.model_name(), i_api + 1, self.max_api_retries, response.error)
            # use the waiting time suggested by the server if it gives one
            suggested = _retry_after_seconds(response.error)
            sleep_time = suggested if suggested is not None else wait_time
            sleep_time = min(sleep_time + 1, self.max_wait_seconds)
            logging.info("Waiting %.0f seconds before calling the LLM again", sleep_time)
            time.sleep(sleep_time)
            # exponential backoff for errors without a suggested waiting time
            wait_time = min(wait_time * 2, self.max_wait_seconds)
        return response


def _retry_after_seconds(error):
    # Groq/Cerebras (OpenAI compatible) put the waiting time in the retry-after header
    # or in the error message, e.g. "Please try again in 7m12.5s"
    resp = getattr(error, "response", None)
    if resp is not None:
        header = resp.headers.get("retry-after")
        if header is not None:
            try:
                return float(header)
            except ValueError:
                pass
    msg = str(error)
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", msg)
    if match and any(match.groups()):
        hours = float(match.group(1) or 0)
        minutes = float(match.group(2) or 0)
        seconds = float(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    return None

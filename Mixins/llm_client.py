import os
import json
import time
import logging
import httpx
from typing import List, Dict, Optional, Any
# from Constants.Mixins.llm_client_constants import LLM_API_KEY as OPENROUTER_API_KEY

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


MODEL = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterMixin:
    """
    Async Mixin for OpenRouter Chat Completions API using httpx.
    Integrates with centralized logging config.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        timeout: int = 60
    ):
        self.api_key = os.getenv("LLM_API_KEY")
        self.site_url = site_url
        self.site_name = site_name
        self.timeout = timeout

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Async chat completion request to OpenRouter API.
        Includes execution time logging.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if extra_params:
            payload.update(extra_params)

        headers = self._build_headers()
        logger.info(f"Sending request to OpenRouter model={model}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        start_time = time.perf_counter()  # ⏱️ Start timer
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url=self.OPENROUTER_BASE_URL,
                    headers=headers,
                    data=json.dumps(payload),
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                elapsed_time = time.perf_counter() - start_time  # ⏱️ End timer
                logger.info(f"Received response from {model} in {elapsed_time:.2f} seconds")
                logger.debug(f"Response JSON: {json.dumps(data, indent=2)}")
                return content

            except httpx.TimeoutException:
                elapsed_time = time.perf_counter() - start_time
                logger.error(f"Request timed out after {self.timeout}s (elapsed: {elapsed_time:.2f}s)")
                return "⚠️ Error: Request timed out."

            except httpx.RequestError as e:
                elapsed_time = time.perf_counter() - start_time
                logger.error(f"Request failed after {elapsed_time:.2f}s: {e}")
                return "⚠️ Error: Unable to reach OpenRouter API."

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                elapsed_time = time.perf_counter() - start_time
                logger.error(f"Unexpected API response after {elapsed_time:.2f}s: {e}")
                logger.debug(f"Raw response: {response.text}")
                return "⚠️ Error: Unexpected response from OpenRouter."


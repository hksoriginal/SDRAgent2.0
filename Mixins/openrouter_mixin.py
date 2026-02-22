from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult, ChatMessage, HumanMessage
from typing import List, Any
import asyncio
from llm_client import OpenRouterMixin
from types import SimpleNamespace


class OpenRouterLLM(BaseChatModel):
    """LangChain wrapper for OpenRouterMixin async client"""

    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-3.3-70b-instruct", **kwargs):

        self.client = OpenRouterMixin()
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs

    async def _agenerate(self, messages: List[ChatMessage], stop: List[str] = None) -> ChatResult:
        # Convert LangChain ChatMessages to OpenRouter format
        api_messages = []
        for msg in messages:
            if msg.type == "human":
                api_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                api_messages.append(
                    {"role": "assistant", "content": msg.content})
            elif msg.type == "system":
                api_messages.append({"role": "system", "content": msg.content})

        # Call OpenRouter API
        content = await self.client.chat_completion(
            messages=api_messages,
            model=self.model_name,
            **self.kwargs
        )

        # Return in LangChain format
        return ChatResult(generations=[[SimpleNamespace(text=content)]])

    def generate(self, messages: List[ChatMessage], stop: List[str] = None) -> ChatResult:
        # Synchronous wrapper for LangChain
        return asyncio.run(self._agenerate(messages, stop=stop))

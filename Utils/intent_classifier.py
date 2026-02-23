from Mixins.llm_client import OpenRouterMixin
from Mixins.text_processor import TextProcessor
import time
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

INTENT_CLASSIFICATION_PROMPT = """
You are an AI assistant that classifies the user's intent based on their message.

## Task
Classify the user request into one of these categories:

1. **query_data** — when user asks to fetch, filter, or analyze data (e.g., "Get leads from universities").
2. **send_email** — when user wants to send or draft emails (e.g., "Email all leads from Delhi").
3. **refresh_data** — when user wants to refresh or update the lead data (e.g., "Refresh the lead data").
4. **fallback** — when the intent doesn’t match any of the above clearly.

## Output Format
Return a JSON object with two keys:
{{
  "intent": "<intent_name>",
  "confidence": <float between 0 and 1>
}}

User message:
{user_input}
"""


class IntentClassifier(OpenRouterMixin, TextProcessor):
    """
    Class that classifies user intent using LLM via OpenRouter.
    """

    def __init__(self):
        super().__init__()

    async def classify_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Classifies user intent with timing and structured output.
        """
        if not user_input.strip():
            return {"intent": "fallback", "confidence": 0.0}

        try:
            start_time = time.perf_counter()
            logger.info(f"🧭 Classifying intent for: '{user_input}'")

            prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=user_input)
            response = await self.chat_completion(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=100
            )

            elapsed_time = time.perf_counter() - start_time

            try:
                intent_data = self.extract_and_validate(
                    response=response, required_keys=["intent", "confidence"])
                logger.info(
                    f"✅ Intent classified in {elapsed_time:.2f}s | Intent: {intent_data.get('intent', 'fallback')}")
            except json.JSONDecodeError:
                intent_data = {"intent": "fallback", "confidence": 0.0}

            return {
                "intent": intent_data.get("intent", "fallback"),
                "confidence": float(intent_data.get("confidence", 0.0)),
                "execution_time": round(elapsed_time, 2)
            }

        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {"intent": "fallback", "confidence": 0.0}

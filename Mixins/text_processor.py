import re
import json
import logging
from typing import Dict, Any, Optional, List
from toon import encode

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Utility class for parsing and processing text responses,
    especially those returned by LLMs or APIs that include JSON content.
    """

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """
        Extracts the first JSON object found in a text response.

        Args:
            response (str): The raw text that may contain a JSON-like structure.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary.

        Raises:
            ValueError: If no valid JSON object is found or parsing fails.
        """
        if not response or not isinstance(response, str):
            raise ValueError("Response must be a non-empty string")

        # Try to find JSON enclosed in { ... }
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                extracted_json = json.loads(json_str)
                logger.info("✅ JSON extracted successfully")
                return extracted_json
            except json.JSONDecodeError as e:
                logger.warning(
                    f"⚠️ JSON decoding failed: {e}. Attempting cleanup.")
                cleaned = self._cleanup_json(json_str)
                try:
                    extracted_json = json.loads(cleaned)
                    logger.info("✅ JSON extracted successfully after cleanup")
                    return extracted_json
                except json.JSONDecodeError as e2:
                    logger.error(f"❌ Cleanup also failed: {e2}")
                    raise ValueError(f"Invalid JSON content: {e2}") from e2
        else:
            raise ValueError("No JSON object found in response")

    def _cleanup_json(self, json_str: str) -> str:
        """
        Attempts to sanitize malformed JSON by:
          - Removing trailing commas
          - Replacing single quotes with double quotes
          - Trimming surrounding text artifacts
        """
        cleaned = json_str.strip()
        cleaned = re.sub(r"(\w)':", r'"\1":', cleaned)  # Fix key quotes
        # Replace single with double quotes
        cleaned = re.sub(r"'", '"', cleaned)
        # Remove trailing commas before }
        cleaned = re.sub(r",\s*}", "}", cleaned)
        # Remove trailing commas before ]
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return cleaned

    def extract_and_validate(self, response: str, required_keys: Optional[list] = None) -> Dict[str, Any]:
        """
        Extracts JSON and validates presence of required keys (if provided).
        """
        data = self._extract_json(response)
        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")
        return data

    def filter_emails(self, strings: List[str]) -> List[str]:
        """
        Filters a list of strings and returns only valid email addresses.

        Args:
            strings (List[str]): List of strings to filter.

        Returns:
            List[str]: List containing only valid email addresses.
        """
        EMAIL_REGEX = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )

        if not strings:
            return []

        return [s for s in strings if isinstance(s, str) and EMAIL_REGEX.match(s)]

    def get_toon(self, json: Dict[str, Any]) -> str:
        return encode(json)

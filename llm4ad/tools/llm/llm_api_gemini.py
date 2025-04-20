from __future__ import annotations

from typing import Any
from google import genai

from llm4ad.base import LLM


class HttpsApiGemini(LLM):
    """Https API
    Args:
        api_key: API key.
        model  : LLM model name.
        timeout: API timeout.
    """
    def __init__(self, api_key: str, model: str, timeout=30, **kwargs):
        super().__init__()
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )

        return response.text
    

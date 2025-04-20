from __future__ import annotations
import sys
import os

# Add the project root directory to system path

import openai
from typing import Any
from google import genai

# from ....base import LLM

# Add the project root directory to system path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
# sys.path.append(project_root)

from ...base import LLM


class HttpsApiOpenAI(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, timeout=30, **kwargs):
        super().__init__()
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout, **kwargs)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=prompt,
            stream=False,
        )
        return response.choices[0].message.content

if __name__ == '__main__':
    model = HttpsApiOpenAI(base_url='https://api.openai.com',
                           api_key='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',
                           model='gpt-3.5-turbo',
                           timeout=30)
    print(model.draw_sample('Hello, how are you?'))

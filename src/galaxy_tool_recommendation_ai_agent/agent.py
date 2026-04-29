from __future__ import annotations

from collections.abc import Callable

from .config import OllamaConfig
from .ollama_client import OllamaClient


SYSTEM_PROMPT = """You are a concise Q&A assistant running locally through Ollama.
Answer the user's question directly.
If you are unsure, say so clearly instead of inventing details.
"""


class TaskAgent:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self.client = OllamaClient(config)

    def run(self, instruction: str, on_token: Callable[[str], None] | None = None) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        return self.client.chat(
            model=self.config.model,
            messages=messages,
            stream=on_token is not None,
            on_token=on_token,
        )

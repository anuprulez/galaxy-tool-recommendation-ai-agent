from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib import error, request

from .config import OllamaConfig


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.config.temperature},
        }

        req = request.Request(
            url=f"{self.config.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                if stream:
                    return self._read_stream(response, on_token)
                body = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Start it with `ollama serve` and confirm the base URL."
            ) from exc

        data = json.loads(body)
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

    def _read_stream(self, response: Any, on_token: Callable[[str], None] | None) -> str:
        parts: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            data = json.loads(line)
            message = data.get("message", {})
            token = message.get("content", "")
            if token:
                parts.append(token)
                if on_token is not None:
                    on_token(token)
            if data.get("done"):
                break

        content = "".join(parts)
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

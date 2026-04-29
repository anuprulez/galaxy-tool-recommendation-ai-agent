from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(slots=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama2:7b"
    temperature: float = 0.2
    workspace: Path = Path.cwd()

    @classmethod
    def from_env(cls, workspace: Path | None = None) -> "OllamaConfig":
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama2:7b"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
            workspace=workspace or Path.cwd(),
        )


OLLAMA_MODEL_PRESETS: dict[str, str] = {
    "balanced": "llama2:7b",
    "fast": "qwen2.5:3b",
    "reasoning": "qwen2.5:7b",
    "code": "deepseek-coder:6.7b",
}

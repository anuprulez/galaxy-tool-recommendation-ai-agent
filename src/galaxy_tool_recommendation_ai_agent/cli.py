from __future__ import annotations

import argparse

from .agent import TaskAgent
from .config import OLLAMA_MODEL_PRESETS, OllamaConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple Q&A agent backed by a local Ollama model."
    )
    parser.add_argument("--instruction", nargs="*", help="Question or prompt to send to the agent")
    parser.add_argument("--model", help="Ollama model to use for the response.")
    parser.add_argument(
        "--preset",
        choices=sorted(OLLAMA_MODEL_PRESETS),
        help="Apply a recommended Ollama model.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    instruction = " ".join(args.instruction).strip()
    if not instruction:
        parser.error("Provide a natural-language instruction.")

    config = OllamaConfig.from_env()

    if args.preset:
        config.model = OLLAMA_MODEL_PRESETS[args.preset]
    if args.model:
        config.model = args.model

    agent = TaskAgent(config)
    response = agent.run(instruction, on_token=lambda token: print(token, end="", flush=True))
    if response and not response.endswith("\n"):
        print()


if __name__ == "__main__":
    main()

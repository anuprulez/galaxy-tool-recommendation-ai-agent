from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = "data/workflows_published"
DEFAULT_OUTPUT_FILE = "data/workflow_summaries.json"
DEFAULT_PROMPT_FILE = "prompts/workflow_summary.yml"
DEFAULT_OLLAMA_MODEL = "llama2:7b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://openwebui.uni-freiburg.de/api"

#"https://api.openai.com/v1"


def parse_args() -> argparse.Namespace:#
    parser = argparse.ArgumentParser(
        description="Summarise published Galaxy workflow JSON files with LangChain."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing published workflow .ga.json files.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="JSON file where workflow summaries are written as a list.",
    )
    parser.add_argument(
        "--prompt",
        help="Instruction prompt used for each workflow summary. Overrides --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=(
            "Read the instruction prompt from a YAML file. The YAML must contain "
            "a 'summary_prompt' or 'prompt' string."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "LLM model name. Defaults to OPENAI_MODEL for OpenAI mode or "
            "OLLAMA_MODEL for Ollama mode."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "ollama", "openai"),
        default="auto",
        help=(
            "LLM provider. In auto mode, OpenAI-compatible chat is used when "
            "OPENAI_API_KEY or OPENWEBUI_API_KEY is available; otherwise Ollama is used."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama base URL. Ignored in OpenAI mode.",
    )
    parser.add_argument(
        "--openai-base-url",
        help=(
            "OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL, "
            "or the public OpenAI API URL."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        help="Model temperature.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="Maximum compact workflow context characters sent to the model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Summarise only the first N workflows. Useful for testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip workflows already present in the output JSON file.",
    )
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()
    if not args.prompt_file:
        raise ValueError("Provide --prompt or --prompt-file.")

    prompt_file = Path(args.prompt_file)
    data = load_yaml_prompt_file(prompt_file)
    for key in ("summary_prompt", "prompt"):
        prompt = data.get(key)
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

    raise ValueError(
        f"Prompt YAML must contain a non-empty 'summary_prompt' or 'prompt' string: {prompt_file}"
    )


def load_yaml_prompt_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read prompt YAML files. Install project dependencies "
            "with `pip install -e .`."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must contain a mapping: {path}")
    return data


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_openai_api_key() -> str:
    return (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("OPENWEBUI_API_KEY", "").strip()
    )


def resolve_provider(provider: str) -> str:
    has_openai_key = bool(resolve_openai_api_key())
    if provider == "auto":
        return "openai" if has_openai_key else "ollama"
    if provider == "openai" and not has_openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY or OPENWEBUI_API_KEY is required for OpenAI mode. "
            "Add one of them to .env or the environment."
        )
    return provider


def resolve_model(model: str | None, provider: str) -> str:
    if model:
        return model
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def resolve_openai_base_url(base_url: str | None) -> str:
    return base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def iter_workflow_files(input_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(input_dir.glob("*.ga.json"), key=lambda path: path.name.lower())
    if limit is not None:
        return files[:limit]
    return files


def workflow_id_from_filename(path: Path) -> str:
    stem = path.name.removesuffix(".ga.json")
    if "__" not in stem:
        return stem
    return stem.rsplit("__", maxsplit=1)[-1]


def compact_workflow_context(workflow: dict[str, Any], max_chars: int) -> str:
    metadata = {
        "name": workflow.get("name"),
        "annotation": workflow.get("annotation"),
        "tags": workflow.get("tags"),
        "license": workflow.get("license"),
        "creator": workflow.get("creator"),
        "version": workflow.get("version"),
        "uuid": workflow.get("uuid"),
    }
    lines = [
        "Workflow metadata:",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "",
        "Ordered workflow steps:",
    ]

    for step in ordered_steps(workflow.get("steps", {})):
        step_text = format_step(step)
        if step_text:
            lines.append(step_text)

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def ordered_steps(raw_steps: Any) -> list[dict[str, Any]]:
    if isinstance(raw_steps, dict):
        steps = list(raw_steps.values())
    elif isinstance(raw_steps, list):
        steps = raw_steps
    else:
        return []

    return sorted(
        [step for step in steps if isinstance(step, dict)],
        key=lambda step: int(step.get("id", 0) or 0),
    )


def format_step(step: dict[str, Any]) -> str:
    step_id = step.get("id")
    step_type = step.get("type")
    name = step.get("name") or step.get("label") or f"Step {step_id}"
    tool_id = step.get("tool_id")
    tool_version = step.get("tool_version")
    annotation = normalize_text(step.get("annotation"))
    inputs = format_names(step.get("inputs"))
    outputs = format_outputs(step.get("outputs"))
    connections = format_connections(step.get("input_connections"))

    parts = [
        f"- Step {step_id}: {name}",
        f"type={step_type}" if step_type else "",
        f"tool_id={tool_id}" if tool_id else "",
        f"tool_version={tool_version}" if tool_version else "",
        f"annotation={annotation}" if annotation else "",
        f"inputs={inputs}" if inputs else "",
        f"outputs={outputs}" if outputs else "",
        f"connections={connections}" if connections else "",
    ]
    return " | ".join(part for part in parts if part)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def format_names(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    names = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("label")
            if name:
                names.append(str(name))
    return ", ".join(names)


def format_outputs(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    outputs = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name")
            output_type = item.get("type")
            if name and output_type:
                outputs.append(f"{name} ({output_type})")
            elif name:
                outputs.append(str(name))
    return ", ".join(outputs)


def format_connections(connections: Any) -> str:
    if not isinstance(connections, dict):
        return ""
    formatted = []
    for input_name, connection in connections.items():
        if isinstance(connection, dict):
            source_step = connection.get("id")
            source_output = connection.get("output_name")
            formatted.append(f"{input_name}<-step{source_step}:{source_output}")
        elif isinstance(connection, list):
            sources = []
            for item in connection:
                if isinstance(item, dict):
                    sources.append(f"step{item.get('id')}:{item.get('output_name')}")
            if sources:
                formatted.append(f"{input_name}<-[{', '.join(sources)}]")
    return "; ".join(formatted)


def build_summary_chain(
    provider: str,
    model: str,
    base_url: str,
    openai_base_url: str,
    openai_api_key: str,
    temperature: float,
    prompt: str,
):
    try:
        from langchain_core.messages import SystemMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        raise RuntimeError(
            "LangChain dependencies are missing. Install the project dependencies first, "
            "for example with `pip install -e .`."
        ) from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            (
                "human",
                "Instruction:\n{instruction}\n\nWorkflow context:\n{workflow_context}",
            ),
        ]
    )
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI LangChain dependency is missing. Install project dependencies "
                "with `pip install -e .`."
            ) from exc

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )
    else:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Ollama LangChain dependency is missing. Install project dependencies "
                "with `pip install -e .`."
            ) from exc

        llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    return prompt | llm | StrOutputParser()


def existing_workflow_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Existing output is not a JSON list: {output_file}")
    return {
        str(item.get("workflow_id"))
        for item in data
        if isinstance(item, dict) and item.get("status") == "summarised"
    }


def load_existing_results(output_file: Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []
    data = json.loads(output_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Existing output is not a JSON list: {output_file}")
    return [item for item in data if isinstance(item, dict)]


def write_results(output_file: Path, results: list[dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_summary_json(summary: str) -> dict[str, Any]:
    cleaned = strip_markdown_json_fence(summary.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM summary is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM summary JSON must be an object.")
    return parsed


def strip_markdown_json_fence(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def summarise_workflows(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    instruction = load_prompt(args)
    provider = resolve_provider(args.provider)
    model = resolve_model(args.model, provider)
    openai_api_key = resolve_openai_api_key()
    openai_base_url = resolve_openai_base_url(args.openai_base_url)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = iter_workflow_files(input_dir, args.limit)
    if not files:
        raise RuntimeError(f"No .ga.json workflow files found in {input_dir}")

    existing_results = load_existing_results(output_file) if args.resume else []
    results = [
        item for item in existing_results if item.get("status") == "summarised"
    ]
    completed_ids = existing_workflow_ids(output_file) if args.resume else set()
    chain = build_summary_chain(
        provider,
        model,
        args.base_url,
        openai_base_url,
        openai_api_key,
        args.temperature,
        instruction,
    )

    for index, path in enumerate(files, start=1):
        workflow_id = workflow_id_from_filename(path)
        if workflow_id in completed_ids:
            print(f"[{index}/{len(files)}] skipped existing: {path.name}")
            continue

        workflow = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict):
            print(f"[{index}/{len(files)}] skipped non-object JSON: {path.name}")
            continue

        context = compact_workflow_context(workflow, max_chars=args.max_context_chars)
        record = {
            "workflow_id": workflow_id,
            "source_file": str(path),
            "name": workflow.get("name"),
            "annotation": workflow.get("annotation"),
            "tags": workflow.get("tags") or [],
            "summary_prompt": instruction,
            "summary_provider": provider,
            "summary_model": model,
            "summary_base_url": openai_base_url if provider == "openai" else args.base_url,
        }

        try:
            summary = chain.invoke(
                {
                    "instruction": instruction,
                    "workflow_context": context,
                }
            ).strip()
            record["status"] = "summarised"
            record["summary"] = parse_summary_json(summary)
            print(f"[{index}/{len(files)}] summarised: {path.name}")
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"[{index}/{len(files)}] failed: {path.name}: {exc}")

        results.append(record)
        write_results(output_file, results)

    write_results(output_file, results)
    print(f"Wrote {len(results)} workflow summaries to {output_file}")


def main() -> None:
    load_env_file()
    args = parse_args()
    summarise_workflows(args)


if __name__ == "__main__":
    main()

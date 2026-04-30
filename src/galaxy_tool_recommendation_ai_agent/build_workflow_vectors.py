from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SUMMARIES_FILE = "data/workflow_summaries.json"
DEFAULT_INDEX_DIR = "data/workflow_summary_index"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LlamaIndex vector index from workflow summary records."
    )
    parser.add_argument(
        "--summaries-file",
        default=DEFAULT_SUMMARIES_FILE,
        help="JSON file containing workflow summary records.",
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Directory where the persisted LlamaIndex vector index is written.",
    )
    parser.add_argument(
        "--embed-model",
        help=(
            "Embedding model to use. Defaults to OPENAI_EMBED_MODEL for OpenAI mode "
            "or OLLAMA_EMBED_MODEL for Ollama mode."
        ),
    )
    parser.add_argument(
        "--embed-provider",
        choices=("auto", "ollama", "openai"),
        default="ollama",
        help=(
            "Embedding provider. In auto mode, OpenAI-compatible embeddings are used "
            "when OPENAI_API_KEY or OPENWEBUI_API_KEY is available; otherwise Ollama is used."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
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
        "--include-failed",
        action="store_true",
        help="Include records whose status is not 'summarised' if they have a summary field.",
    )
    return parser.parse_args()


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


def resolve_embed_model(model: str | None, provider: str) -> str:
    if model:
        return model
    if provider == "openai":
        return os.getenv("OPENAI_EMBED_MODEL", DEFAULT_OPENAI_EMBED_MODEL)
    return os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)


def resolve_openai_base_url(base_url: str | None) -> str:
    return base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def load_summary_records(path: Path, include_failed: bool) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON list in {path}")

    records: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not is_usable_summary(item.get("summary")):
            continue
        if not include_failed and item.get("status") != "summarised":
            continue
        records.append(item)
    return records


def is_usable_summary(summary: Any) -> bool:
    if isinstance(summary, str):
        return bool(summary.strip())
    if isinstance(summary, dict):
        return bool(summary)
    if isinstance(summary, list):
        return bool(summary)
    return False


def summary_to_text(summary: Any) -> str:
    if isinstance(summary, str):
        return summary.strip()
    if isinstance(summary, dict):
        return json_summary_to_text(summary)
    if isinstance(summary, list):
        return "\n".join(
            json_summary_to_text(item) for item in summary if isinstance(item, dict)
        )
    return ""


def json_summary_to_text(summary: dict[str, Any]) -> str:
    parts = [
        text_field("Workflow summary name", summary.get("workflow_name")),
        text_field("Workflow purpose", summary.get("workflow_purpose")),
        text_field("Scientific domain", summary.get("scientific_domain")),
        text_field("Main analysis task", summary.get("main_analysis_task")),
    ]

    ordered_steps = summary.get("ordered_steps")
    if isinstance(ordered_steps, list):
        step_lines = [
            format_summary_step(step)
            for step in ordered_steps
            if isinstance(step, dict)
        ]
        if step_lines:
            parts.extend(["Ordered steps:", *step_lines])

    nested_summary = summary.get("summary")
    if isinstance(nested_summary, dict) and nested_summary:
        parts.append("Summary details:")
        parts.extend(format_key_value(key, value) for key, value in nested_summary.items())
    elif isinstance(nested_summary, str) and nested_summary.strip():
        parts.append(f"Summary details: {nested_summary.strip()}")

    extra_keys = [
        key
        for key in summary
        if key
        not in {
            "workflow_name",
            "workflow_purpose",
            "scientific_domain",
            "main_analysis_task",
            "ordered_steps",
            "summary",
        }
    ]
    if extra_keys:
        parts.append("Additional summary fields:")
        parts.extend(format_key_value(key, summary[key]) for key in extra_keys)

    return "\n".join(part for part in parts if part)


def text_field(label: str, value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return f"{label}: {text}"


def format_summary_step(step: dict[str, Any]) -> str:
    fields = [
        f"step_id={step.get('step_id')}" if step.get("step_id") is not None else "",
        f"tool_name={step.get('tool_name')}" if step.get("tool_name") else "",
        f"step_purpose={step.get('step_purpose')}" if step.get("step_purpose") else "",
        format_key_value("input_datatypes", step.get("input_datatypes")),
        format_key_value("output_datatypes", step.get("output_datatypes")),
    ]
    return "- " + " | ".join(field for field in fields if field)


def format_key_value(key: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        return f"{key}: {', '.join(str(item) for item in value)}"
    if isinstance(value, dict):
        if not value:
            return ""
        return f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
    text = str(value).strip()
    if not text:
        return ""
    return f"{key}: {text}"


def record_to_document(record: dict[str, Any]):
    try:
        from llama_index.core import Document
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex dependencies are missing. Install them with `pip install -e .`."
        ) from exc

    workflow_id = str(record.get("workflow_id") or "")
    name = str(record.get("name") or workflow_id or "Unnamed workflow")
    annotation = str(record.get("annotation") or "")
    tags = record.get("tags") if isinstance(record.get("tags"), list) else []
    summary = summary_to_text(record.get("summary"))

    text = "\n".join(
        part
        for part in [
            f"Workflow name: {name}",
            f"Workflow id: {workflow_id}" if workflow_id else "",
            f"Annotation: {annotation}" if annotation else "",
            f"Tags: {', '.join(str(tag) for tag in tags)}" if tags else "",
            "Summary:",
            summary,
        ]
        if part
    )

    return Document(
        text=text,
        doc_id=workflow_id or str(record.get("source_file") or name),
        metadata={
            "workflow_id": workflow_id,
            "name": name,
            "source_file": str(record.get("source_file") or ""),
            "tags": tags,
            "status": str(record.get("status") or ""),
        },
    )


def build_index(
    summaries_file: Path,
    index_dir: Path,
    embed_provider: str,
    embed_model: str,
    base_url: str,
    openai_base_url: str,
    openai_api_key: str,
    include_failed: bool,
) -> None:
    try:
        from llama_index.core import Settings, VectorStoreIndex
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex dependencies are missing. Install them with `pip install -e .`."
        ) from exc

    records = load_summary_records(summaries_file, include_failed=include_failed)
    if not records:
        raise RuntimeError(f"No usable workflow summaries found in {summaries_file}")

    if embed_provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "LlamaIndex OpenAI embedding dependency is missing. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        Settings.embed_model = OpenAIEmbedding(
            model=embed_model,
            api_key=openai_api_key,
            api_base=openai_base_url,
        )
    else:
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "LlamaIndex Ollama embedding dependency is missing. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        Settings.embed_model = OllamaEmbedding(model_name=embed_model, base_url=base_url)

    documents = [record_to_document(record) for record in records]
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(index_dir))

    manifest = {
        "summaries_file": str(summaries_file),
        "index_dir": str(index_dir),
        "records_indexed": len(records),
        "embed_provider": embed_provider,
        "embed_model": embed_model,
        "base_url": openai_base_url if embed_provider == "openai" else base_url,
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "workflow_index_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Indexed {len(records)} workflow summaries into {index_dir}")


def main() -> None:
    load_env_file()
    args = parse_args()
    embed_provider = resolve_provider(args.embed_provider)
    embed_model = resolve_embed_model(args.embed_model, embed_provider)
    openai_api_key = resolve_openai_api_key()
    openai_base_url = resolve_openai_base_url(args.openai_base_url)
    build_index(
        summaries_file=Path(args.summaries_file),
        index_dir=Path(args.index_dir),
        embed_provider=embed_provider,
        embed_model=embed_model,
        base_url=args.base_url,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        include_failed=args.include_failed,
    )


if __name__ == "__main__":
    main()

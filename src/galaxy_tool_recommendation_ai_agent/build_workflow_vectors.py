from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SUMMARIES_FILE = "data/workflow_summaries.json"
DEFAULT_INDEX_DIR = "data/workflow_summary_index"
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
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
        default=DEFAULT_EMBED_MODEL,
        help="Ollama embedding model to use.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include records whose status is not 'summarised' if they have a summary field.",
    )
    return parser.parse_args()


def load_summary_records(path: Path, include_failed: bool) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON list in {path}")

    records: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        if not include_failed and item.get("status") != "summarised":
            continue
        records.append(item)
    return records


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
    summary = str(record.get("summary") or "")

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
    embed_model: str,
    base_url: str,
    include_failed: bool,
) -> None:
    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.embeddings.ollama import OllamaEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex Ollama dependencies are missing. Install them with `pip install -e .`."
        ) from exc

    records = load_summary_records(summaries_file, include_failed=include_failed)
    if not records:
        raise RuntimeError(f"No usable workflow summaries found in {summaries_file}")

    Settings.embed_model = OllamaEmbedding(model_name=embed_model, base_url=base_url)
    documents = [record_to_document(record) for record in records]
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(index_dir))

    manifest = {
        "summaries_file": str(summaries_file),
        "index_dir": str(index_dir),
        "records_indexed": len(records),
        "embed_model": embed_model,
        "base_url": base_url,
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "workflow_index_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Indexed {len(records)} workflow summaries into {index_dir}")


def main() -> None:
    args = parse_args()
    build_index(
        summaries_file=Path(args.summaries_file),
        index_dir=Path(args.index_dir),
        embed_model=args.embed_model,
        base_url=args.base_url,
        include_failed=args.include_failed,
    )


if __name__ == "__main__":
    main()

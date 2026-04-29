from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_INDEX_DIR = "data/workflow_summary_index"
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve workflow summary context with LlamaIndex and answer a user query."
    )
    parser.add_argument("query", nargs="*", help="User question to answer.")
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the persisted workflow summary vector index.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Ollama embedding model used when querying the index.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help="Ollama LLM model used to answer the query.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of workflow summaries to retrieve.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved workflow context before the answer.",
    )
    return parser.parse_args()


def load_index(index_dir: Path, embed_model: str, base_url: str):
    try:
        from llama_index.core import Settings, StorageContext, load_index_from_storage
        from llama_index.embeddings.ollama import OllamaEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex dependencies are missing. Install them with `pip install -e .`."
        ) from exc

    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory does not exist: {index_dir}. "
            "Run `python build_workflow_vectors.py` first."
        )

    Settings.embed_model = OllamaEmbedding(model_name=embed_model, base_url=base_url)
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    return load_index_from_storage(storage_context)


def retrieve_context(index, query: str, top_k: int) -> list[dict[str, object]]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    contexts: list[dict[str, object]] = []
    for node in nodes:
        metadata = dict(node.node.metadata or {})
        contexts.append(
            {
                "score": node.score,
                "workflow_id": metadata.get("workflow_id"),
                "name": metadata.get("name"),
                "source_file": metadata.get("source_file"),
                "text": node.node.get_content(),
            }
        )
    return contexts


def answer_query(query: str, contexts: list[dict[str, object]], model: str, base_url: str) -> str:
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex Ollama LLM dependency is missing. Install it with `pip install -e .`."
        ) from exc

    context_text = "\n\n".join(
        (
            f"Context {index}\n"
            f"Workflow: {context.get('name')}\n"
            f"Workflow id: {context.get('workflow_id')}\n"
            f"Similarity score: {context.get('score')}\n"
            f"{context.get('text')}"
        )
        for index, context in enumerate(contexts, start=1)
    )

    prompt = f"""You are a Galaxy workflow recommendation assistant.
Use only the retrieved workflow summaries below to answer the user query.
Recommend the most relevant workflows and explain briefly why they match.
If the retrieved context is insufficient, say so.

Retrieved workflow summaries:
{context_text}

User query:
{query}

Answer:"""

    llm = Ollama(model=model, base_url=base_url, request_timeout=180.0)
    response = llm.complete(prompt)
    return str(response).strip()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("Provide a user query.")

    index = load_index(Path(args.index_dir), embed_model=args.embed_model, base_url=args.base_url)
    contexts = retrieve_context(index, query=query, top_k=args.top_k)
    if args.show_context:
        print(json.dumps(contexts, indent=2, ensure_ascii=False))
        print()

    answer = answer_query(query, contexts, model=args.model, base_url=args.base_url)
    print(answer)


if __name__ == "__main__":
    main()

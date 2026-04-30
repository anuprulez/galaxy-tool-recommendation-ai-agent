from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INDEX_DIR = "data/workflow_summary_index"
DEFAULT_PROMPT_FILE = "prompts/workflow_rag.yml"
DEFAULT_RESPONSE_OUTPUT_FILE = "data/output/wf_rag_llm_responses.json"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_LLM_MODEL = "llama2:7b"
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://openwebui.uni-freiburg.de/api"
#"https://api.openai.com/v1"
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_LLM_MODEL)
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
        help=(
            "Embedding model used when querying the index. Defaults to "
            "OPENAI_EMBED_MODEL for OpenAI mode or OLLAMA_EMBED_MODEL for Ollama mode."
        ),
    )
    parser.add_argument(
        "--embed-provider",
        choices=("auto", "ollama", "openai"),
        default="ollama",
        help=(
            "Embedding provider. Use the same provider/model used to build the index. "
            "In auto mode, OpenAI-compatible embeddings are used when OPENAI_API_KEY "
            "or OPENWEBUI_API_KEY is available; otherwise Ollama is used."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "LLM model used to answer the query. Defaults to OPENAI_MODEL for OpenAI "
            "mode or OLLAMA_MODEL for Ollama mode."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "ollama", "openai"),
        default="ollama",
        help=(
            "Answer LLM provider. In auto mode, OpenAI-compatible chat is used when "
            "OPENAI_API_KEY or OPENWEBUI_API_KEY is available; otherwise Ollama is used."
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
        "--top-k",
        type=int,
        default=1,
        help="Number of workflow summaries to retrieve.",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=(
            "Read the answer prompt template from a YAML file. The YAML must contain "
            "a 'rag_prompt' or 'prompt' string."
        ),
    )
    parser.add_argument(
        "--response-output-file",
        default=DEFAULT_RESPONSE_OUTPUT_FILE,
        help="JSON file where generated LLM responses are appended.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved workflow context before the answer.",
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


def resolve_llm_model(model: str | None, provider: str) -> str:
    if model:
        return model
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_LLM_MODEL)
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_LLM_MODEL)


def resolve_openai_base_url(base_url: str | None) -> str:
    return base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def load_prompt_template(prompt_file: Path) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read prompt YAML files. Install project dependencies "
            "with `pip install -e .`."
        ) from exc

    data = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must contain a mapping: {prompt_file}")

    for key in ("rag_prompt", "prompt"):
        prompt = data.get(key)
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

    raise ValueError(
        f"Prompt YAML must contain a non-empty 'rag_prompt' or 'prompt' string: {prompt_file}"
    )


def load_index(
    index_dir: Path,
    embed_provider: str,
    embed_model: str,
    base_url: str,
    openai_base_url: str,
    openai_api_key: str,
):
    try:
        from llama_index.core import Settings, StorageContext, load_index_from_storage
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex dependencies are missing. Install them with `pip install -e .`."
        ) from exc

    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory does not exist: {index_dir}. "
            "Run `python build_workflow_vectors.py` first."
        )

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
                "tags": metadata.get("tags", []),
            }
        )
    return contexts


def build_answer_prompt(
    query: str,
    contexts: list[dict[str, object]],
    prompt_template: str,
) -> str:
    context_text = "\n\n".join(
        (
            f"Context {index}\n"
            f"Workflow: {context.get('name')}\n"
            f"Workflow id: {context.get('workflow_id')}\n"
            f"Similarity score: {context.get('score')}\n"
            f"Tags: {', '.join(context.get('tags', []))}\n"
            f"{context.get('text')}"
        )
        for index, context in enumerate(contexts, start=1)
    )

    return (
        prompt_template.replace("{context_text}", context_text)
        .replace("{query}", query)
        .strip()
    )


def answer_query(
    query: str,
    contexts: list[dict[str, object]],
    provider: str,
    model: str,
    base_url: str,
    openai_base_url: str,
    openai_api_key: str,
    prompt_template: str,
) -> str:
    prompt = build_answer_prompt(query, contexts, prompt_template)
    if provider == "openai":
        try:
            from llama_index.llms.openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LlamaIndex OpenAI LLM dependency is missing. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        llm = OpenAI(
            model=model,
            api_key=openai_api_key,
            api_base=openai_base_url,
            request_timeout=600.0,
        )
    else:
        try:
            from llama_index.llms.ollama import Ollama
        except ImportError as exc:
            raise RuntimeError(
                "LlamaIndex Ollama LLM dependency is missing. Install it with `pip install -e .`."
            ) from exc

        llm = Ollama(model=model, base_url=base_url, request_timeout=600.0)
    response = llm.complete(prompt)
    return str(response).strip()


def save_generated_response(
    output_file: Path,
    query: str,
    response: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "query": query,
        "response": response,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        record.update(metadata)

    records = load_saved_responses(output_file)
    records.append(record)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def load_saved_responses(output_file: Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []

    content = output_file.read_text(encoding="utf-8").strip()
    if not content:
        return []

    data = json.loads(content)
    if not isinstance(data, list):
        raise RuntimeError(f"Existing response output is not a JSON list: {output_file}")
    if not all(isinstance(item, dict) for item in data):
        raise RuntimeError(f"Existing response output must be a list of JSON objects: {output_file}")
    return data


def main() -> None:
    load_env_file()
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("Provide a user query.")

    embed_provider = resolve_provider(args.embed_provider)
    llm_provider = resolve_provider(args.provider)
    embed_model = resolve_embed_model(args.embed_model, embed_provider)
    llm_model = resolve_llm_model(args.model, llm_provider)
    openai_api_key = resolve_openai_api_key()
    openai_base_url = resolve_openai_base_url(args.openai_base_url)
    prompt_template = load_prompt_template(Path(args.prompt_file))
    index = load_index(
        Path(args.index_dir),
        embed_provider=embed_provider,
        embed_model=embed_model,
        base_url=args.base_url,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )
    contexts = retrieve_context(index, query=query, top_k=args.top_k)
    if args.show_context:
        print(json.dumps(contexts, indent=2, ensure_ascii=False))
        print()

    answer = answer_query(
        query,
        contexts,
        provider=llm_provider,
        model=llm_model,
        base_url=args.base_url,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        prompt_template=prompt_template,
    )
    save_generated_response(
        Path(args.response_output_file),
        query=query,
        response=answer,
        metadata={
            "provider": llm_provider,
            "model": llm_model,
            "embed_provider": embed_provider,
            "embed_model": embed_model,
            "base_url": openai_base_url if llm_provider == "openai" else args.base_url,
            "embed_base_url": openai_base_url if embed_provider == "openai" else args.base_url,
            "top_k": args.top_k,
            "prompt_file": args.prompt_file,
            "retrieved_workflows": [
                {
                    "workflow_id": context.get("workflow_id"),
                    "name": context.get("name"),
                    "score": context.get("score"),
                    "source_file": context.get("source_file"),
                }
                for context in contexts
            ],
        },
    )
    print(answer)


if __name__ == "__main__":
    main()

# galaxy-tool-recommendation-ai-agent

Local Q&A agent that accepts a natural-language prompt and answers it with an Ollama model running on your machine.

## Features

- Natural-language question input from the CLI.
- Single Ollama chat request with no planner/executor loop.
- Streaming output so tokens appear immediately instead of waiting for the full response.
- Simple model selection with either `--model` or `--preset`.
- Knowledge-base builder that normalizes GTN tutorial Markdown and Galaxy workflows into retrieval-ready JSONL.

## Run It

1. Start Ollama:

```bash
ollama serve
```

2. Pull at least one model:

```bash
ollama pull llama3.1:8b
```

3. Ask a question:

```bash
python main.py "What is retrieval augmented generation?"
```

Use a specific model if needed:

```bash
python main.py --model qwen2.5:7b "Explain vector databases in simple terms"
```

## Ollama Model Options

The agent is set up specifically for Ollama models.

```bash
python main.py --model llama3.1:8b "Summarize how transformers work"
```

Or use a preset:

```bash
python main.py --preset reasoning "Compare REST and GraphQL"
```

Available presets:

- `fast` -> `qwen2.5:3b`
- `balanced` -> `llama3.1:8b`
- `reasoning` -> `qwen2.5:7b`
- `code` -> `deepseek-coder:6.7b`

Environment variables also work:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_TEMPERATURE=0.2
```

## Build A Knowledge Base

The repository already contains GTN-derived source files under `data/`. You can normalize them into retrieval-ready documents with:

```bash
python build_kb.py --input-dir data --output-dir kb
```

Or through the module / script entry point after installation:

```bash
python -m galaxy_tool_recommendation_ai_agent.build_kb --input-dir data --output-dir kb
build-kb --input-dir data --output-dir kb
```

This writes:

- `kb/documents.jsonl`: one JSON document per tutorial section, workflow step, or workflow summary
- `kb/manifest.json`: counts by source type and build metadata

Current document types:

- `tutorial_section`: Markdown front matter + heading-based section chunks
- `workflow_step`: one document per Galaxy tool step with tool ID and datatypes
- `workflow_summary`: one document per workflow with ordered tool sequence

Representative fields in `documents.jsonl`:

```json
{
  "doc_id": "topics__single-cell__tutorials__scrna-scanpy-pbmc3k__workflows__Clustering-3k-PBMC-with-Scanpy__step_5",
  "source_type": "workflow_step",
  "topic": "single-cell",
  "tutorial": "scrna-scanpy-pbmc3k",
  "tool_id": "toolshed.g2.bx.psu.edu/repos/iuc/scanpy_filter/scanpy_filter/1.10.2+galaxy0",
  "input_types": ["h5ad"],
  "output_types": ["h5ad"],
  "text": "Workflow tool step 5: Scanpy Filter..."
}
```

This is the intended starting point for retrieval and reranking before prompting the LLM for Galaxy tool recommendations.

## Collect GTN Topic Files

To download Markdown and workflow files from the Galaxy Training Network topics tree while preserving the original `topics/...` layout:

```bash
python src/galaxy_tool_recommendation_ai_agent/collect_gtn_material.py
```

After installation, the same collector is available as:

```bash
collect-gtn-material
```

By default, files are written under `data/gtn`. The collector writes an ordered `manifest.json` alongside the downloaded tree. Within each topic/tutorial group, tutorial Markdown files are listed before workflow files.

## Collect Published Histories

To download published history metadata from `usegalaxy.eu`:

```bash
python src/galaxy_tool_recommendation_ai_agent/collect_published_histories.py
```

After installation, the same collector is available as:

```bash
collect-published-histories
```

By default, the JSON response is written to `data/usegalaxy_eu_published_histories.json`.

## Collect Published Workflows

To download published workflow JSON files from `usegalaxy.eu`:

```bash
python src/galaxy_tool_recommendation_ai_agent/collect_published_workflows.py
```

After installation, the same collector is available as:

```bash
collect-published-workflows
```

By default, workflow files are written under `data/workflows_published` with one `.ga.json` file per workflow and a `manifest.json` summary.

## Summarise Published Workflows

To generate LangChain/Ollama summaries for workflows in `data/workflows_published`:

```bash
python summarise.py --output-file data/workflow_summaries.json --resume
```

By default, the summary instruction is read from `prompts/workflow_summary.yml`. Use `--prompt-file` to point at another YAML file containing `summary_prompt` or `prompt`, or use `--prompt` to override it directly. The output is a JSON list with one record per workflow, including the source file, workflow metadata, prompt, model, status, and summary text.

## Workflow Summary RAG

Build a LlamaIndex vector index from `data/workflow_summaries.json`:

```bash
python build_workflow_vectors.py
```

Ask a question against the indexed workflow summaries:

```bash
python query_workflow_rag.py "Which workflows can analyse bacterial genome assemblies?"
```

The answer prompt is read from `prompts/workflow_rag.yml` by default. Use `--prompt-file` to point at another YAML file containing `rag_prompt` or `prompt`; include `{context_text}` and `{query}` where the retrieved summaries and user query should be inserted.

Generated answers are appended to `data/output/wf_rag_llm_responses.json` with the original query and retrieval metadata. Use `--response-output-file` to write to a different JSON file.

Both scripts use Ollama by default. `OLLAMA_EMBED_MODEL` controls the embedding model, while `OLLAMA_MODEL` controls the answer model.

## Collect IWC Workflows

To download files from the Intergalactic Workflow Commission `workflows/` tree:

```bash
python src/galaxy_tool_recommendation_ai_agent/collect_iwc_workflows.py
```

After installation, the same collector is available as:

```bash
collect-iwc-workflows
```

By default, files are written under `data/workflows` using the same layout as the GitHub `workflows/` directory.

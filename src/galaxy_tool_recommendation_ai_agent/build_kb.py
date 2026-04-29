from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


MARKDOWN_SUFFIXES = {".md"}
WORKFLOW_SUFFIXES = {".ga"}
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a retrieval-ready knowledge base from GTN tutorial markdown and Galaxy workflows."
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory containing flattened GTN source files.",
    )
    parser.add_argument(
        "--output-dir",
        default="kb",
        help="Directory where JSONL documents and metadata are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    builder = KnowledgeBaseBuilder(input_dir=input_dir, output_dir=output_dir)
    result = builder.build()
    print(
        "Built knowledge base with "
        f"{result['documents_written']} documents from {result['files_processed']} files "
        f"into {output_dir}"
    )


class KnowledgeBaseBuilder:
    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def build(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        files_processed = 0
        skipped_files = 0
        source_counter: Counter[str] = Counter()

        for path in sorted(self.input_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name == "manifest.json":
                continue

            suffix = path.suffix.lower()
            if suffix in MARKDOWN_SUFFIXES:
                file_records = self._parse_markdown_file(path)
            elif suffix in WORKFLOW_SUFFIXES:
                file_records = self._parse_workflow_file(path)
            else:
                skipped_files += 1
                continue

            files_processed += 1
            records.extend(file_records)
            for record in file_records:
                source_counter.update([record["source_type"]])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        documents_path = self.output_dir / "documents.jsonl"
        documents_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

        manifest = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "files_processed": files_processed,
            "files_skipped": skipped_files,
            "documents_written": len(records),
            "document_types": dict(sorted(source_counter.items())),
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def _parse_markdown_file(self, path: Path) -> list[dict[str, Any]]:
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = parse_markdown_front_matter(raw_text)
        source_meta = infer_source_metadata(path)

        sections = split_markdown_sections(body)
        records: list[dict[str, Any]] = []
        for index, section in enumerate(sections, start=1):
            text = normalize_whitespace(section["content"])
            if not text:
                continue

            title = section["title"] or metadata.get("title") or source_meta["tutorial"]
            section_tags = dedupe_preserve_order(
                stringify_list(metadata.get("tags"))
                + source_meta["path_parts"]
            )

            records.append(
                {
                    "doc_id": f"{source_meta['source_stem']}__section_{index}",
                    "source_type": "tutorial_section",
                    "source_path": path.name,
                    "title": title,
                    "topic": source_meta["topic"],
                    "tutorial": source_meta["tutorial"],
                    "section_title": section["title"],
                    "section_level": section["level"],
                    "tags": section_tags,
                    "questions": stringify_list(metadata.get("questions")),
                    "objectives": stringify_list(metadata.get("objectives")),
                    "level": metadata.get("level"),
                    "text": text,
                }
            )

        return records

    def _parse_workflow_file(self, path: Path) -> list[dict[str, Any]]:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        source_meta = infer_source_metadata(path)
        records: list[dict[str, Any]] = []

        ordered_steps = sorted(
            workflow.get("steps", {}).values(),
            key=lambda item: int(item.get("id", 0)),
        )
        tool_names: list[str] = []
        for step in ordered_steps:
            if step.get("type") != "tool":
                continue

            step_id = int(step.get("id", 0))
            tool_name = step.get("name") or step.get("label") or f"Step {step_id}"
            tool_names.append(tool_name)
            input_types = extract_step_input_types(step)
            output_types = extract_step_output_types(step)
            tool_id = step.get("tool_id")
            annotation = normalize_whitespace(step.get("annotation", ""))

            text_parts = [
                f"Workflow tool step {step_id}: {tool_name}.",
                f"Tool id: {tool_id}." if tool_id else "",
                f"Annotation: {annotation}." if annotation else "",
                format_datatypes("Input datatypes", input_types),
                format_datatypes("Output datatypes", output_types),
            ]
            records.append(
                {
                    "doc_id": f"{source_meta['source_stem']}__step_{step_id}",
                    "source_type": "workflow_step",
                    "source_path": path.name,
                    "title": tool_name,
                    "topic": source_meta["topic"],
                    "tutorial": source_meta["tutorial"],
                    "workflow_name": workflow.get("name"),
                    "workflow_annotation": normalize_whitespace(workflow.get("annotation", "")),
                    "step_id": step_id,
                    "tool_id": tool_id,
                    "tool_version": step.get("tool_version"),
                    "input_types": input_types,
                    "output_types": output_types,
                    "tags": dedupe_preserve_order(
                        source_meta["path_parts"] + [tool_name] + ([tool_id] if tool_id else [])
                    ),
                    "text": normalize_whitespace(" ".join(part for part in text_parts if part)),
                }
            )

        summary_text = normalize_whitespace(
            " ".join(
                part
                for part in [
                    f"Workflow: {workflow.get('name', source_meta['tutorial'])}.",
                    f"Annotation: {workflow.get('annotation', '')}."
                    if workflow.get("annotation")
                    else "",
                    f"Tool sequence: {', '.join(tool_names)}." if tool_names else "",
                ]
                if part
            )
        )
        records.append(
            {
                "doc_id": f"{source_meta['source_stem']}__workflow_summary",
                "source_type": "workflow_summary",
                "source_path": path.name,
                "title": workflow.get("name") or source_meta["tutorial"],
                "topic": source_meta["topic"],
                "tutorial": source_meta["tutorial"],
                "workflow_name": workflow.get("name"),
                "workflow_annotation": normalize_whitespace(workflow.get("annotation", "")),
                "tool_sequence": tool_names,
                "tags": dedupe_preserve_order(source_meta["path_parts"] + tool_names),
                "text": summary_text,
            }
        )
        return records


def infer_source_metadata(path: Path) -> dict[str, Any]:
    parts = path.stem.split("__")
    topic = value_after(parts, "topics")
    tutorial = value_after(parts, "tutorials")
    return {
        "source_stem": path.stem,
        "topic": topic,
        "tutorial": tutorial,
        "path_parts": parts,
    }


def value_after(parts: list[str], marker: str) -> str | None:
    try:
        index = parts.index(marker)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def parse_markdown_front_matter(raw_text: str) -> tuple[dict[str, Any], str]:
    if not raw_text.startswith("---\n"):
        return {}, raw_text

    end_marker = "\n---\n"
    end_index = raw_text.find(end_marker, 4)
    if end_index == -1:
        return {}, raw_text

    front_matter = raw_text[4:end_index]
    body = raw_text[end_index + len(end_marker) :]
    return parse_simple_yaml(front_matter), body


def parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith(" ") and not line.lstrip().startswith("- "):
            continue

        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and current_key and current_list is not None:
            current_list.append(strip_yaml_value(stripped[2:]))
            continue

        if line.startswith(" "):
            continue

        if ":" not in line:
            current_key = None
            current_list = None
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key

        if not value:
            current_list = []
            data[key] = current_list
            continue

        current_list = None
        data[key] = parse_yaml_value(value)

    return data


def parse_yaml_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [strip_yaml_value(item) for item in inner.split(",")]
    return strip_yaml_value(value)


def strip_yaml_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def split_markdown_sections(body: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_level: int | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            if current_lines:
                sections.append(
                    {
                        "title": current_title,
                        "level": current_level,
                        "content": "\n".join(current_lines).strip(),
                    }
                )
            current_title = match.group(2).strip()
            current_level = len(match.group(1))
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "title": current_title,
                "level": current_level,
                "content": "\n".join(current_lines).strip(),
            }
        )

    return sections


def extract_step_input_types(step: dict[str, Any]) -> list[str]:
    input_types: list[str] = []
    for value in step.get("inputs", []):
        datatype = value.get("type")
        if datatype:
            input_types.append(datatype)
    return dedupe_preserve_order(input_types)


def extract_step_output_types(step: dict[str, Any]) -> list[str]:
    output_types: list[str] = []
    for output in step.get("outputs", []):
        datatype = output.get("type")
        if datatype:
            output_types.append(datatype)
    return dedupe_preserve_order(output_types)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_datatypes(label: str, values: list[str]) -> str:
    if not values:
        return ""
    return f"{label}: {', '.join(values)}."


def stringify_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def dedupe_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


if __name__ == "__main__":
    main()

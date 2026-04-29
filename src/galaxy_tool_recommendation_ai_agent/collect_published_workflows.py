from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

import requests


DEFAULT_GALAXY_URL = "https://usegalaxy.eu"
DEFAULT_OUTPUT_DIR = "data/workflows_published"
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download published Galaxy workflows as workflow JSON files."
    )
    parser.add_argument(
        "--galaxy-url",
        default=DEFAULT_GALAXY_URL,
        help="Galaxy instance URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where published workflow JSON files are written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between workflow downloads in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download workflows even when a local copy already exists.",
    )
    return parser.parse_args()


def safe_filename(value: str) -> str:
    safe = SAFE_NAME_PATTERN.sub("_", value).strip("._")
    return safe or "workflow"


def list_published_workflows(
    session: requests.Session,
    galaxy_url: str,
    timeout: int,
) -> list[dict[str, Any]]:
    response = session.get(
        f"{galaxy_url.rstrip('/')}/api/workflows",
        params={"published": "true"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("Published workflows endpoint did not return a list.")
    return data


def download_workflow(
    session: requests.Session,
    galaxy_url: str,
    workflow_id: str,
    timeout: int,
) -> object:
    response = session.get(
        f"{galaxy_url.rstrip('/')}/api/workflows/{workflow_id}/download",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def workflow_output_path(output_dir: Path, workflow: dict[str, Any]) -> Path:
    workflow_id = str(workflow["id"])
    name = str(workflow.get("name") or workflow_id)
    return output_dir / f"{safe_filename(name)}__{safe_filename(workflow_id)}.ga.json"


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


def collect_published_workflows(
    galaxy_url: str,
    output_dir: Path,
    timeout: int,
    delay: float,
    overwrite: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "galaxy-tool-recommendation-ai-agent",
        }
    )

    workflows = list_published_workflows(session, galaxy_url, timeout=timeout)
    print(f"Found {len(workflows)} workflows")

    manifest: list[dict[str, Any]] = []
    saved = 0
    skipped = 0
    failed = 0

    for index, workflow in enumerate(workflows, start=1):
        workflow_id = str(workflow["id"])
        name = str(workflow.get("name") or workflow_id)
        outfile = workflow_output_path(output_dir, workflow)

        record: dict[str, Any] = {
            "order": index,
            "workflow_id": workflow_id,
            "name": name,
            "local_path": str(outfile.relative_to(output_dir)),
            "download_url": f"{galaxy_url.rstrip('/')}/api/workflows/{workflow_id}/download",
        }

        if outfile.exists() and not overwrite:
            skipped += 1
            record["status"] = "skipped"
            manifest.append(record)
            print(f"[{index}/{len(workflows)}] skipped: {outfile}")
            continue

        try:
            data = download_workflow(session, galaxy_url, workflow_id, timeout=timeout)
            outfile.write_text(json.dumps(data, indent=2), encoding="utf-8")
            saved += 1
            record["status"] = "saved"
            print(f"[{index}/{len(workflows)}] saved: {outfile}")
            if delay > 0:
                time.sleep(delay)
        except Exception as exc:
            failed += 1
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"[{index}/{len(workflows)}] failed: {workflow_id} {name} {exc}")

        manifest.append(record)

    write_manifest(output_dir, manifest)
    return {
        "found": len(workflows),
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
    }


def main() -> None:
    args = parse_args()
    result = collect_published_workflows(
        galaxy_url=args.galaxy_url,
        output_dir=Path(args.output_dir).resolve(),
        timeout=args.timeout,
        delay=args.delay,
        overwrite=args.overwrite,
    )
    print(
        "Published workflow collection complete: "
        f"{result['saved']} saved, {result['skipped']} skipped, {result['failed']} failed"
    )


if __name__ == "__main__":
    main()

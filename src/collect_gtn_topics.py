from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from urllib import error, request
from urllib.parse import quote


API_TREE_URL = "https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
RAW_FILE_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

WORKFLOW_SUFFIXES = (
    ".ga",
    ".cwl",
    ".wdl",
    ".gxwf.yml",
    ".gxwf.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GTN topic Markdown and workflow files into a local data directory."
    )
    parser.add_argument("--owner", default="galaxyproject", help="GitHub owner.")
    parser.add_argument("--repo", default="training-material", help="GitHub repository name.")
    parser.add_argument("--ref", default="main", help="Git ref, branch, or tag.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where downloaded files and manifest are stored.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of download retries for transient network failures.",
    )
    return parser.parse_args()


def github_get_json(url: str, timeout: int, retries: int) -> dict:
    req = request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "galaxy-tool-recommendation-ai-agent",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = min(2**attempt, 10)
            print(
                f"Retrying metadata request ({attempt}/{retries}) after error: {exc}. "
                f"Waiting {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to fetch GitHub metadata: {url}") from last_error


def download_text(url: str, timeout: int, retries: int) -> str:
    req = request.Request(url, headers={"User-Agent": "galaxy-tool-recommendation-ai-agent"})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = min(2**attempt, 10)
            print(
                f"Retrying download ({attempt}/{retries}) after error: {exc}. "
                f"Waiting {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download file: {url}") from last_error


def should_collect(path: str) -> bool:
    if not path.startswith("topics/"):
        return False

    path_parts = path.split("/")
    filename = path_parts[-1]

    if (
        "tutorials" in path_parts
        and filename.startswith("tutorial")
        and filename.endswith(".md")
    ):
        return True

    if path.endswith(WORKFLOW_SUFFIXES):
        return True

    if "workflows" in path_parts and (
        path.endswith(".yml")
        or path.endswith(".yaml")
        or path.endswith(".json")
    ):
        return True

    return False


def flattened_name(path: str) -> str:
    return path.replace("/", "__")


def collect_files(owner: str, repo: str, ref: str, timeout: int, retries: int) -> list[str]:
    tree_url = API_TREE_URL.format(owner=owner, repo=repo, ref=ref)
    data = github_get_json(tree_url, timeout=timeout, retries=retries)
    tree = data.get("tree", [])
    if not tree:
        raise RuntimeError("GitHub tree response was empty.")

    files: list[str] = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if should_collect(path):
            files.append(path)
    return sorted(files)


def write_outputs(
    owner: str,
    repo: str,
    ref: str,
    output_dir: Path,
    files: list[str],
    timeout: int,
    retries: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    for index, path in enumerate(files, start=1):
        raw_url = RAW_FILE_URL.format(
            owner=owner,
            repo=repo,
            ref=ref,
            path=quote(path, safe="/"),
        )
        destination = output_dir / flattened_name(path)
        if destination.exists():
            manifest.append(
                {
                    "source_path": path,
                    "download_url": raw_url,
                    "local_path": destination.name,
                }
            )
            print(f"[{index}/{len(files)}] Skipping existing file: {path}")
            continue

        content = download_text(raw_url, timeout=timeout, retries=retries)
        destination.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "source_path": path,
                "download_url": raw_url,
                "local_path": destination.name,
            }
        )
        print(f"[{index}/{len(files)}] {path} -> {destination}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    files = collect_files(args.owner, args.repo, args.ref, timeout=args.timeout, retries=args.retries)
    print(f"Found {len(files)} topic markdown/workflow files in {args.owner}/{args.repo}@{args.ref}")
    write_outputs(
        args.owner,
        args.repo,
        args.ref,
        output_dir,
        files,
        timeout=args.timeout,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()

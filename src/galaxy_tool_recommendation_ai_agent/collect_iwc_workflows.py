from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from urllib import error, request
from urllib.parse import quote


DEFAULT_OWNER = "galaxyproject"
DEFAULT_REPO = "iwc"
DEFAULT_REF = "main"
DEFAULT_OUTPUT_DIR = "data/workflows"
WORKFLOWS_PREFIX = "workflows/"

API_TREE_URL = "https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
RAW_FILE_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download files from the IWC workflows tree, preserving its layout."
    )
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="GitHub owner.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository name.")
    parser.add_argument("--ref", default=DEFAULT_REF, help="Git ref, branch, or tag.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where workflow files and manifest are written.",
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
        help="Number of retries for transient network failures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download files even when a local copy already exists.",
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
    return json.loads(fetch_bytes(req, timeout=timeout, retries=retries).decode("utf-8"))


def fetch_bytes(req: request.Request, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = min(2**attempt, 10)
            print(
                f"Retrying request ({attempt}/{retries}) after error: {exc}. "
                f"Waiting {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed request: {req.full_url}") from last_error


def should_collect(path: str) -> bool:
    if not path.startswith(WORKFLOWS_PREFIX):
        return False
    filename = path.rsplit("/", maxsplit=1)[-1]
    return not filename.startswith(".")


def local_relative_path(source_path: str) -> Path:
    return Path(source_path.removeprefix(WORKFLOWS_PREFIX))


def collect_paths(owner: str, repo: str, ref: str, timeout: int, retries: int) -> list[str]:
    tree_url = API_TREE_URL.format(owner=owner, repo=repo, ref=ref)
    data = github_get_json(tree_url, timeout=timeout, retries=retries)
    tree = data.get("tree", [])
    if not tree:
        raise RuntimeError("GitHub tree response was empty.")

    paths = [
        item["path"]
        for item in tree
        if item.get("type") == "blob" and should_collect(item.get("path", ""))
    ]
    return sorted(paths, key=str.lower)


def write_outputs(
    owner: str,
    repo: str,
    ref: str,
    output_dir: Path,
    paths: list[str],
    timeout: int,
    retries: int,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str | int]] = []

    for index, source_path in enumerate(paths, start=1):
        raw_url = RAW_FILE_URL.format(
            owner=owner,
            repo=repo,
            ref=ref,
            path=quote(source_path, safe="/"),
        )
        destination = output_dir / local_relative_path(source_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        status = "downloaded"
        if destination.exists() and not overwrite:
            status = "skipped"
            print(f"[{index}/{len(paths)}] Skipping existing file: {source_path}")
        else:
            req = request.Request(
                raw_url,
                headers={"User-Agent": "galaxy-tool-recommendation-ai-agent"},
            )
            destination.write_bytes(fetch_bytes(req, timeout=timeout, retries=retries))
            print(f"[{index}/{len(paths)}] {source_path} -> {destination}")

        manifest.append(
            {
                "order": index,
                "source_path": source_path,
                "download_url": raw_url,
                "local_path": str(destination.relative_to(output_dir)),
                "status": status,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    paths = collect_paths(
        args.owner,
        args.repo,
        args.ref,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(f"Found {len(paths)} files in {args.owner}/{args.repo}@{args.ref}/{WORKFLOWS_PREFIX}")
    write_outputs(
        args.owner,
        args.repo,
        args.ref,
        output_dir,
        paths,
        timeout=args.timeout,
        retries=args.retries,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

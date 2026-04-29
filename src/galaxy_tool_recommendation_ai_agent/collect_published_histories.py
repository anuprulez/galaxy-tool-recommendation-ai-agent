from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from urllib import error, request


DEFAULT_URL = "https://usegalaxy.eu/api/histories/published"
DEFAULT_OUTPUT_FILE = "data/usegalaxy_eu_published_histories.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download published Galaxy histories from usegalaxy.eu as JSON."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Galaxy API endpoint for published histories.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="JSON file where published histories are written.",
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
    return parser.parse_args()


def fetch_json(url: str, timeout: int, retries: int) -> object:
    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "galaxy-tool-recommendation-ai-agent",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = min(2**attempt, 10)
            print(
                f"Retrying request ({attempt}/{retries}) after error: {exc}. "
                f"Waiting {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch JSON: {url}") from last_error


def write_json(output_file: Path, data: object) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_file = Path(args.output_file).resolve()
    data = fetch_json(args.url, timeout=args.timeout, retries=args.retries)
    write_json(output_file, data)

    item_count = len(data) if isinstance(data, list) else "unknown"
    print(f"Wrote published histories JSON to {output_file}")
    print(f"History records: {item_count}")


if __name__ == "__main__":
    main()

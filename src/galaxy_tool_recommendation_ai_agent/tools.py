from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess


BLOCKED_COMMANDS = {
    "rm",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd",
    "poweroff",
    "halt",
    "init",
}

SHELL_META_CHARS = {"|", ";", "&", ">", "<", "$", "`"}


class ToolRegistry:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def describe(self) -> str:
        return json.dumps(
            {
                "list_files": {"path": "Relative directory path. Defaults to '.'"},
                "read_file": {"path": "Relative file path"},
                "write_file": {"path": "Relative file path", "content": "Full file content"},
                "append_file": {"path": "Relative file path", "content": "Text to append"},
                "run_command": {"command": "Single safe command without shell pipes or redirection"},
                "finish": {"response": "Final answer for the user"},
            },
            indent=2,
        )

    def execute(self, action: str, arguments: dict[str, str]) -> str:
        try:
            if action == "list_files":
                return self.list_files(arguments.get("path", "."))
            if action == "read_file":
                return self.read_file(arguments["path"])
            if action == "write_file":
                return self.write_file(arguments["path"], arguments["content"])
            if action == "append_file":
                return self.append_file(arguments["path"], arguments["content"])
            if action == "run_command":
                return self.run_command(arguments["command"])
            raise ValueError(f"Unknown action: {action}")
        except Exception as exc:
            return json.dumps(
                {
                    "error": str(exc),
                    "action": action,
                    "arguments": arguments,
                },
                indent=2,
            )

    def _resolve_path(self, path_text: str) -> Path:
        normalized = (path_text or ".").strip()
        raw_path = Path(normalized).expanduser()
        candidate = (raw_path if raw_path.is_absolute() else self.workspace / raw_path).resolve()
        if candidate != self.workspace and not candidate.is_relative_to(self.workspace):
            raise ValueError("Path escapes the workspace.")
        return candidate

    def list_files(self, path_text: str) -> str:
        path = self._resolve_path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"{path_text} does not exist.")
        if path.is_file():
            return path.name
        entries = sorted(item.name + ("/" if item.is_dir() else "") for item in path.iterdir())
        return "\n".join(entries) if entries else "(empty)"

    def read_file(self, path_text: str) -> str:
        path = self._resolve_path(path_text)
        return path.read_text(encoding="utf-8")

    def write_file(self, path_text: str, content: str) -> str:
        path = self._resolve_path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {path.relative_to(self.workspace)}"

    def append_file(self, path_text: str, content: str) -> str:
        path = self._resolve_path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"Appended to {path.relative_to(self.workspace)}"

    def run_command(self, command: str) -> str:
        if any(char in command for char in SHELL_META_CHARS):
            raise ValueError("Shell metacharacters are blocked. Use a single plain command.")

        parts = shlex.split(command)
        if not parts:
            raise ValueError("Command cannot be empty.")
        if parts[0] in BLOCKED_COMMANDS:
            raise ValueError(f"Blocked command: {parts[0]}")

        completed = subprocess.run(
            parts,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        output = completed.stdout.strip()
        error = completed.stderr.strip()
        return json.dumps(
            {
                "exit_code": completed.returncode,
                "stdout": output,
                "stderr": error,
            },
            indent=2,
        )

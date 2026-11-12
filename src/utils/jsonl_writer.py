"""Đóng gói và xuất file submission.jsonl cùng các tiện ích JSONL."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from config.settings import DEFAULT_ROOT, PUBLIC_TEST_EXPECTED_COUNT


def compact(value) -> str:
    """Serialize JSON compactly without NaN."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def dump(path: str | Path, obj) -> None:
    """Write formatted JSON to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write a list of dictionary records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Read a JSONL file and return (rows, errors)."""
    rows, errors = [], []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                errors.append({"line": line_no, "error": "blank_line"})
                continue
            try:
                obj = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                if not isinstance(obj, dict):
                    raise ValueError("row must be a JSON object")
                rows.append(obj)
            except (ValueError, TypeError) as exc:
                errors.append({"line": line_no, "error": str(exc)})
    return rows, errors


def index_jsonl(path: str | Path) -> dict[str, dict]:
    """Read a JSONL file and index records by 'id'."""
    rows, errors = read_jsonl(path)
    mapped = {row["id"]: row for row in rows}
    if errors or len(rows) != len(mapped):
        raise ValueError(f"Invalid input {path}: {errors}")
    return mapped


def history_public(history: list[dict]) -> list[dict]:
    """Keep only public conversation turn fields."""
    fields = {"role", "text", "transcript", "tool_calls", "tool_results"}
    return [{k: v for k, v in turn.items() if k in fields} for turn in history]


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_submission(directory: str | Path, output: str | Path, root: str | Path = DEFAULT_ROOT) -> None:
    """Validate and export a candidate submission.jsonl to the final destination."""
    directory = Path(directory)
    output = Path(output)
    root = Path(root)
    candidate = directory / "submission.jsonl"
    rows, errors = read_jsonl(candidate)
    expected, source_errors = read_jsonl(root / "public_test" / "public_test.jsonl")
    if (
        errors
        or source_errors
        or len(rows) != PUBLIC_TEST_EXPECTED_COUNT
        or {r["id"] for r in rows} != {r["id"] for r in expected}
    ):
        raise ValueError(f"Submission must cover all {PUBLIC_TEST_EXPECTED_COUNT} public test IDs exactly once.")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "zaic_tools",
            "--split",
            "public_test",
            "--root",
            str(root),
            "--validate",
            str(candidate.resolve()),
        ],
        cwd=root / "public_test",
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if f"{PUBLIC_TEST_EXPECTED_COUNT} lines checked, 0 schema errors" not in result.stdout:
        raise ValueError(result.stdout)
    temporary = output.with_suffix(".pending")
    shutil.copyfile(candidate, temporary)
    temporary.replace(output)
    print(result.stdout.strip(), flush=True)
    print(f"Submission ready at {output}", flush=True)


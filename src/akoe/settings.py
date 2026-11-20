# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Local settings from a `.env` file (never committed).

Values already present in the environment win over the file. Values are never
logged or printed: only the names of the variables that were loaded.

Search order: the path given, then $AKOE_ENV_FILE, then `./.env`, then the
`.env` at the root of a source checkout.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_VAR = "AKOE_ENV_FILE"


def _candidates(path: Path | None) -> list[Path]:
    out: list[Path] = []
    if path is not None:
        out.append(Path(path))
    if os.environ.get(ENV_FILE_VAR):
        out.append(Path(os.environ[ENV_FILE_VAR]))
    out.append(Path.cwd() / ".env")
    out.append(Path(__file__).resolve().parents[2] / ".env")  # src/akoe/settings.py -> checkout root
    return out


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines; `#` comments, blank lines and `export ` prefixes are
    ignored; matching single or double quotes around the value are removed."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(path: Path | None = None) -> list[str]:
    """Load the first `.env` found into os.environ without overriding variables
    that are already set, and skipping empty values. Returns the loaded names."""
    for candidate in _candidates(path):
        if candidate.is_file():
            loaded = []
            for key, value in parse_env(candidate.read_text(encoding="utf-8")).items():
                if value and key not in os.environ:
                    os.environ[key] = value
                    loaded.append(key)
            return loaded
    return []

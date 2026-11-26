"""The JSON files next to the populators, which the database is seeded from."""

import json
from pathlib import Path
from typing import Any


def load_config(path: Path, populator: str) -> dict[str, Any]:
    """The JSON object in ``path``; ``populator`` names the caller in the error."""
    if not path.exists():
        raise FileNotFoundError(
            f"[{populator}] Missing JSON configuration file: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

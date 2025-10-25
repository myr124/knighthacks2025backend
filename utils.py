import os
import json
from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv

# Load .env from project root automatically
load_dotenv(find_dotenv())


def load_archetypes(path: str) -> Dict[str, str]:
    """
    Load personality archetypes from a JSON file.

    Args:
        path: Path to the JSON file containing archetypes

    Returns:
        Dictionary mapping archetype names to descriptions

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    base_dir = os.path.dirname(__file__)
    prefix = "multi-persona-agent/"
    if path.startswith(prefix):
        rel_path = path[len(prefix):]
    else:
        rel_path = path

    full_path = os.path.normpath(os.path.join(base_dir, rel_path))

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Archetypes file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_instructions(path: str) -> str:
    """
    Load instruction text for agents.

    Behavior:
    - If `path` starts with "multi-persona-agent/", the remainder is resolved
      relative to this package directory (where this utils.py lives).
    - Otherwise `path` is treated as relative to this package directory.
    - Returns the file contents as a string.

    Raises:
    - FileNotFoundError if the file does not exist.
    - OSError for other IO errors.
    """
    base_dir = os.path.dirname(__file__)
    prefix = "multi-persona-agent/"
    if path.startswith(prefix):
        rel_path = path[len(prefix) :]
    else:
        rel_path = path

    full_path = os.path.normpath(os.path.join(base_dir, rel_path))

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Instruction file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

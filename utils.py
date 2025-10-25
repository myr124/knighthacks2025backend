import os
import json
import re
from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv

# Load .env from project root automatically
load_dotenv(find_dotenv())


def _resolve_path(path: str) -> str:
    """
    Resolve a file path relative to the project root.

    Args:
        path: Path to the file (can be relative or include agent-specific prefixes)

    Returns:
        Absolute path to the file

    Behavior:
        - If path starts with a known agent directory (e.g., "multi-persona-agent/"),
          it's resolved relative to the project root
        - Otherwise, path is treated as relative to the project root
    """
    base_dir = os.path.dirname(__file__)

    # List of known agent directories (can be extended)
    agent_prefixes = ["multi-persona-agent/"]

    # Check if path starts with any known prefix
    rel_path = path
    for prefix in agent_prefixes:
        if path.startswith(prefix):
            rel_path = path
            break

    full_path = os.path.normpath(os.path.join(base_dir, rel_path))
    return full_path


def load_json(path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.

    Args:
        path: Path to the JSON file (relative to project root)

    Returns:
        Parsed JSON data as a dictionary

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
        OSError: For other IO errors

    Example:
        archetypes = load_json("multi-persona-agent/personality_archetypes.txt")
        config = load_json("config.json")
    """
    full_path = _resolve_path(path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"JSON file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str) -> str:
    """
    Load a text file and return its contents as a string.

    Args:
        path: Path to the text file (relative to project root)

    Returns:
        File contents as a string

    Raises:
        FileNotFoundError: If the file does not exist
        OSError: For other IO errors

    Example:
        prompt = load_text("multi-persona-agent/prompts/archetype_instructions_template.txt")
        readme = load_text("README.md")
    """
    full_path = _resolve_path(path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Text file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


# Backward compatibility aliases
def load_archetypes(path: str) -> Dict[str, str]:
    """Deprecated: Use load_json() instead."""
    return load_json(path)


def load_instructions(path: str) -> str:
    """Deprecated: Use load_text() instead."""
    return load_text(path)


def parse_emergency_plan_phases(plan_text: str) -> List[Dict[str, str]]:
    """
    Parse an emergency management plan into distinct phases.

    Args:
        plan_text: Full emergency plan text

    Returns:
        List of dictionaries with 'phase_name', 'timestamp', and 'content' keys

    Example:
        phases = parse_emergency_plan_phases(load_text("prompts/emergency_manager_input_example.txt"))
        # Returns: [
        #   {"phase_name": "72 HOURS BEFORE LANDFALL", "timestamp": "Day 1 - Monday 6:00 AM", "content": "..."},
        #   {"phase_name": "48 HOURS BEFORE LANDFALL", "timestamp": "Day 2 - Tuesday 7:00 AM", "content": "..."},
        #   ...
        # ]
    """
    phases = []

    # Split by major phase markers (=== TIMELINE or === DURING or === POST or === RECOVERY)
    phase_pattern = r'===\s*([^=]+?)\s*==='

    # Find all phase headers
    matches = list(re.finditer(phase_pattern, plan_text))

    for i, match in enumerate(matches):
        phase_header = match.group(1).strip()
        start_pos = match.end()

        # Find the content between this phase and the next
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            # For the last phase, take everything until the end or until "END OF"
            end_match = re.search(r'END OF EMERGENCY', plan_text[start_pos:])
            if end_match:
                end_pos = start_pos + end_match.start()
            else:
                end_pos = len(plan_text)

        phase_content = plan_text[start_pos:end_pos].strip()

        # Extract timestamp from the first line if it has "UPDATE" or phase indicator
        timestamp_match = re.search(r'(Day \d+[^)]*\):|[A-Z][a-z]+\s+\d+:\d+\s+[AP]M)', phase_content)
        timestamp = timestamp_match.group(1) if timestamp_match else "No specific time"

        phases.append({
            "phase_name": phase_header,
            "timestamp": timestamp,
            "content": phase_content
        })

    return phases

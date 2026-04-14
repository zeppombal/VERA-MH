"""Shared path/filename conventions for generation runs and transcripts."""

import re
from typing import Any, Dict

# Suffix after first `_` in `{tag}_{persona_token}_{model}_run{N}.txt`
TRANSCRIPT_RUN_SUFFIX_RE = re.compile(r"_run(?P<run>\d+)\.txt$")


def model_token_for_run_folder(model_name: str) -> str:
    """Normalize a model id for p_/a_ segments in generation run folder names."""
    return model_name.replace("-", "_").replace(".", "_")


def build_generation_run_folder_name(
    persona_model: str,
    agent_model: str,
    max_turns: int,
    runs_per_prompt: int,
    timestamp: str,
) -> str:
    """
    Basename only (no parent path):
    p_{persona}__a_{agent}__t{turns}__r{runs}__{timestamp}
    """
    p = model_token_for_run_folder(persona_model)
    a = model_token_for_run_folder(agent_model)
    return f"p_{p}__a_{a}__t{max_turns}__r{runs_per_prompt}__{timestamp}"


def parse_generation_run_folder_name(folder_name: str) -> Dict[str, Any]:
    """
    Parse a generation run folder basename:
      p_{persona}__a_{agent}__t{turns}__r{runs}__{timestamp}
    """
    pattern = (
        r"^p_(?P<persona>.+)__a_(?P<agent>.+)__t(?P<turns>\d+)__r(?P<runs>\d+)__"
        r"(?P<timestamp>\d{8}_\d{6})$"
    )
    match = re.match(pattern, folder_name)
    if not match:
        raise ValueError(
            "Resume mode requires --folder-name to be a run folder with format "
            "'p_{persona}__a_{agent}__t{turns}__r{runs}__{timestamp}'."
        )
    return {
        "persona": match.group("persona"),
        "agent": match.group("agent"),
        "turns": int(match.group("turns")),
        "runs": int(match.group("runs")),
        "timestamp": match.group("timestamp"),
    }


def persona_token_for_transcript_stem(persona_name: str) -> str:
    """Normalize persona names to match transcript filename stems when resuming."""
    return persona_name.replace(" ", "_").replace(".", "")


def is_generation_run_folder_basename(name: str) -> bool:
    """True if basename looks like a generation run folder (p_*__a_*...)."""
    return name.startswith("p_") and "__a_" in name

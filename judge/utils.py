"""Utility functions for the judge module."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_judge_models(model_arg):
    """Parse judge model specifications from command line argument into a dictionary."""
    judge_models = {}
    for model_spec in model_arg:
        if ":" in model_spec:
            # Format: "model:count"
            model, count = model_spec.rsplit(":", 1)
            try:
                n = int(count)
            except ValueError:
                raise ValueError(
                    f"Judge model count must be an integer, got {count!r}"
                ) from None
            if n < 1:
                raise ValueError(f"Judge model count must be positive, got {n}")
            judge_models[model] = n
        else:
            # Format: "model" (defaults to 1 instance)
            judge_models[model_spec] = 1

    return judge_models


def _safe_path_token(value: str) -> str:
    token = value.replace("-", "_").replace(".", "_")
    token = re.sub(r"[^A-Za-z0-9_]+", "_", token)
    return re.sub(r"_+", "_", token).strip("_")


def build_evaluation_run_folder_path(
    output_root: str,
    judge_info: str,
    timestamp: str,
    conversation_run_basename: str,
) -> str:
    """
    Build the folder path for a judge evaluation run.
    """
    safe_judge_info = _safe_path_token(judge_info)
    folder_name = f"j_{safe_judge_info}__{timestamp}"
    return str(Path(output_root) / folder_name)


def judge_evaluation_tsv_filename(
    conversation_filename: str,
    judge_model: str,
    judge_instance: Optional[int] = None,
) -> str:
    """
    Basename of the per-evaluation TSV written by LLMJudge._save_results.

    When judge_instance is set (batch runner), suffix is _i{instance}. When None
    (e.g. single-conversation CLI), there is no _i suffix — matches historical behavior.
    """
    conversation_name = Path(conversation_filename).stem
    judge_suffix = judge_model.replace("/", "_").replace(":", "_")
    if judge_instance is not None:
        judge_suffix += f"_i{judge_instance}"
    return f"{conversation_name}_{judge_suffix}.tsv"


def build_single_conversation_judge_run_key(
    conversation_path: str | Path,
    *,
    now: Optional[datetime] = None,
) -> str:
    """
    Human-readable folder basename for ``judge.py`` single-file (``-c``) mode.

    Format: ``single_<timestamp_ms>__<conversation_stem>``. Batch judging uses the
    evaluation folder basename instead; this gives a stable, unique folder name
    per CLI run. Per-task logs use :func:`build_judge_task_log_path` with
    ``output_folder`` set, so no ``run_key`` is involved.
    """
    dt = datetime.now() if now is None else now
    ts = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = Path(conversation_path).stem
    return f"single_{ts}__{stem}"


def get_judge_logs_root() -> str:
    """
    Root directory for legacy per-task judge LLM logs (when ``output_folder`` is
    omitted from :func:`build_judge_task_log_path`).

    Override with env ``VERA_JUDGE_LOGS_ROOT`` (e.g. set by pytest to a temp dir).
    Default: ``judge_logs`` in the current working directory.
    """
    return os.environ.get("VERA_JUDGE_LOGS_ROOT", "judge_logs")


def build_judge_task_log_path(
    conversation_filename: str,
    judge_model: str,
    judge_instance: Optional[int] = None,
    *,
    run_key: Optional[str] = None,
    logs_root: Optional[str] = None,
    output_folder: Optional[str] = None,
) -> str:
    """
    Path to the per-task judge LLM log file, parallel to judge_evaluation_tsv_filename.

    With ``output_folder``: ``{output_folder}/logs/{stem}.log`` (``run_key`` unused).

    Legacy layout (no ``output_folder``): ``{logs_root}/{run_key}/{stem}.log``
    (``logs_root`` defaults to :func:`get_judge_logs_root`). ``run_key`` is required
    for this branch.
    """
    tsv_name = judge_evaluation_tsv_filename(
        conversation_filename, judge_model, judge_instance
    )
    stem = Path(tsv_name).stem
    if output_folder is not None:
        return str(Path(output_folder) / "logs" / f"{stem}.log")
    if not run_key:
        raise ValueError("run_key is required when output_folder is not set")
    # Legacy layout: {logs_root}/{run_key}/{stem}.log
    root = logs_root if logs_root is not None else get_judge_logs_root()
    return str(Path(root) / run_key / f"{stem}.log")


def load_rubric_structure(
    rubric_path: str, sep: str = "\t"
) -> Tuple[List[str], List[str]]:
    """
    Load DIMENSIONS and OPTIONS from the rubric file.

    Args:
        rubric_path: Path to the rubric file
        sep: Separator for the file (default: tab)

    Returns:
        Tuple of (dimensions, options):
        - dimensions: List of unique dimension names from the Dimension column
        - options: List of scoring option column names (empty for question-flow rubrics)
    """
    rubric_df = pd.read_csv(rubric_path, sep=sep)

    # Get unique dimensions from the Dimension column
    dimensions = [
        d.strip()
        for d in rubric_df["Dimension"].dropna().unique()
        if d and str(d).strip() != "nan"
    ]

    # Get options from columns (exclude metadata columns)
    columns = [col.strip() for col in rubric_df.columns]
    # Question-flow rubric columns: Question ID, Dimension, Risk Type,
    # Question, Examples, Severity, Answer, GOTO
    metadata_columns = {
        "Question ID",
        "Dimension",
        "Risk Type",
        "Question",
        "Examples",
        "Severity",
        "Answer",
        "GOTO",
    }
    options = [col for col in columns if col not in metadata_columns]

    return dimensions, options


def extract_model_names_from_path(path_input: str) -> Dict[str, str]:
    """
    Extract all three model names from the evaluation directory path.

    Directory format:
        j_{judge}__p_{persona}__a_{agent}__t{max_turns}__r{runs}__{timestamp}

    Args:
        path_input: Path to results.csv file or evaluation directory

    Returns:
        Dictionary with 'judge', 'persona', and 'agent' keys, or 'Unknown' if not found
    """
    path = Path(path_input)

    # If it's a file (e.g., results.csv), get the parent directory name.
    # If it's a directory, use the directory name directly.
    if path.is_file():
        dir_name = path.parent.name
        parent_names = [parent.name for parent in path.parents]
    else:
        dir_name = path.name
        parent_names = [parent.name for parent in path.parents]

    result = {"judge": "Unknown", "persona": "Unknown", "agent": "Unknown"}

    # Extract judge model: j_{judge}__p_...
    # Note: judge name may include a timestamp like _20251112_171754_380
    if dir_name.startswith("j_"):
        judge_part = dir_name[2:].split("__", 1)[0]
        judge_part = re.sub(r"_\d{8}_\d{6}.*$", "", judge_part)
        result["judge"] = judge_part.replace("_", " ").strip()

    # Extract persona model: ...__p_{persona}__a_...
    generation_dir_name = dir_name
    if "__p_" not in generation_dir_name:
        for parent_name in parent_names:
            if parent_name.startswith("p_") and "__a_" in parent_name:
                generation_dir_name = parent_name
                break

    if generation_dir_name.startswith("p_") and "__a_" in generation_dir_name:
        persona_part = generation_dir_name[2:].split("__a_")[0]
        result["persona"] = persona_part.replace("_", " ").strip()
    elif "__p_" in generation_dir_name:
        parts = generation_dir_name.split("__p_")
        if len(parts) > 1:
            persona_part = parts[1].split("__a_")[0]
            result["persona"] = persona_part.replace("_", " ").strip()

    # Extract agent model: ...__a_{agent}__t...
    if "__a_" in generation_dir_name:
        parts = generation_dir_name.split("__a_")
        if len(parts) > 1:
            agent_part = parts[1].split("__t")[0]
            result["agent"] = agent_part.replace("_", " ").strip()

    return result


def extract_persona_name_from_filename(filename: str) -> Optional[str]:
    """
    Extract persona name from TSV filename.

    Filename format: {hash}_{persona_name}_{model}_run{number}.tsv
    Example: 1a84d1_Brian_g4o_run3.tsv -> "Brian"

    Works with any file extension (.tsv, .txt, etc.) since it splits on "_".

    Args:
        filename: TSV filename (with or without extension)

    Returns:
        Persona name or None if not found
    """
    # Format: {hash}_{persona}_{model}_run{number} or
    # {hash}_{persona}_{model}_run{number}.tsv
    # Extension doesn't matter since we split on "_"
    try:
        parts = filename.split("_")
        if len(parts) >= 2:
            # Return just the persona name (index 1), not the list
            return parts[1]
        return None
    except Exception as e:
        print(f"Error extracting persona name from filename {filename}: {e}")
        return None


def default_adhoc_parent() -> str:
    """
    Parent directory for single-file judge runs and unscoped LLMJudge log dirs.

    Set VERA_ADHOC_PARENT to override (tests use a temp directory).
    """
    return os.environ.get("VERA_ADHOC_PARENT") or os.path.join("output", "adhoc")

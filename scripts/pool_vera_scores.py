#!/usr/bin/env uv run python
"""
Pool judge evaluation results from multiple evaluation runs (e.g. two user-agent
pipelines) into one results.csv, then recompute VERA scores and visualizations.

Typical layout for each input path:
  output/p_<user>__a_<agent>__t30__r1__<ts>/evaluations/j_<...>/results.csv

You may pass either the evaluation directory (``j_*`` folder) or ``results.csv``
inside it. For example::

  uv run python scripts/pool_vera_scores.py \\
    output/p_claude__a_gpt_4o__t6__r1_20260422_100000/evaluations/j_gpt_4o__... \\
    output/p_gpt_4o__a_gpt_4o__t6__r1_20260422_110000/evaluations/j_gpt_4o__.../results.csv

**Legacy layout** (e.g. top-level ``evaluations/j_*`` next to a flat ``conversations/``
run): merging and scoring still work; only the auto-generated ``j_pooled__*`` folder
name may fall back to ``unknown`` placeholders because the script cannot infer the
``p_*`` generation basename from the path alone.

Also supports extracting the last evaluation directory from a run_pipeline log:
  uv run python scripts/pool_vera_scores.py --extract-from-log /path/to/log.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# in case running from the scripts directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.naming import parse_generation_run_folder_name  # noqa: E402


def extract_last_evaluation_dir_from_pipeline_log(text: str) -> str:
    """
    Parse captured ``run_pipeline.py`` log text and return the last evaluation folder.

    The pipeline prints a line containing ``Evaluations saved to:`` followed by the
    ``j_*`` evaluation directory (with or without a trailing slash). When the log
    contains multiple runs, the **last** match is returned so callers can chain
    subprocess output from successive pipeline invocations.

    Args:
        text: Full stdout/stderr text (e.g. contents of a temp file used with ``tee``).

    Returns:
        Absolute path to the evaluation directory as a string.

    Raises:
        ValueError: If no matching line exists in *text*.
    """
    matches = re.findall(r"Evaluations saved to:\s*(.+)$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(
            "Could not find any 'Evaluations saved to:' line in pipeline log output."
        )
    path = matches[-1].strip().rstrip("/")
    return str(Path(path).resolve())


def _resolve_eval_input(path: Path) -> Path:
    """
    Normalize a CLI path to the judge evaluation directory (``j_*`` folder).

    Accepts either the evaluation directory itself or a ``results.csv`` file inside
    it, so users can pass ``.../j_*`` or ``.../j_*/results.csv`` interchangeably.

    Args:
        path: User-supplied filesystem path (file or directory).

    Returns:
        Resolved path to the parent ``j_*`` evaluation folder.

    Raises:
        ValueError: If *path* is a file that is not named ``results.csv``.
        FileNotFoundError: If *path* does not exist.
    """
    path = path.resolve()
    if path.is_file():
        if path.name != "results.csv":
            raise ValueError(f"Expected results.csv or a directory, got: {path}")
        return path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _generation_folder_for_eval(eval_dir: Path) -> Path | None:
    """
    Walk from an evaluation folder up to the corresponding generation run folder.

    Expected layout: ``.../p_*__/evaluations/j_*`` (nested generation run). If
    *eval_dir* is not directly under an ``evaluations`` folder whose parent is a
    ``p_*`` run (e.g. legacy ``repo/evaluations/j_*``), returns None. Callers still
    pool successfully; they only lose path-derived metadata for the synthetic output
    folder name.

    Args:
        eval_dir: Resolved path to a ``j_*`` evaluation directory.

    Returns:
        Path to the ``p_*`` generation run folder, or None if the layout does not match.
    """
    eval_dir = eval_dir.resolve()
    parent = eval_dir.parent
    if parent.name != "evaluations":
        return None
    gen = parent.parent
    if gen.name.startswith("p_"):
        return gen
    return None


def _persona_slug_for_pool(source_eval_dirs: list[Path]) -> str:
    """
    Build a short ``p_*`` path segment from each source generation folder basename.

    Returns one persona model slug, or sorted unique names joined by ``+`` when
    multiple user models are merged (order-independent, stable across runs).
    """
    persona_names: list[str] = []
    for d in source_eval_dirs:
        gen = _generation_folder_for_eval(d)
        if gen is None:
            persona_names.append("unknown")
            continue
        try:
            meta = parse_generation_run_folder_name(gen.name)
            persona_names.append(str(meta["persona"]))
        except ValueError:
            persona_names.append("unknown")
    return "+".join(sorted(set(persona_names)))


def _synthetic_pooled_folder_basename(source_eval_dirs: list[Path]) -> str:
    """
    Build a ``j_*``-style basename for the pooled output directory.

    The ``p_*`` segment lists user/persona models inferred from every source's sibling
    ``p_*`` folder (sorted, ``+``-joined when more than one). Provider agent, turn
    count, and runs-per-prompt reuse the first source's generation folder when
    parsable; otherwise falls back to ``t30__r1`` and ``a_unknown``. A timestamp
    suffix keeps same-day batches distinct.

    Args:
        source_eval_dirs: Non-empty list of resolved evaluation directories.

    Returns:
        Directory basename (no path separators), e.g.
        ``j_pooled__p_claude_sonnet_4_5+gpt_4o_mini__a_gpt_4o__t30__r1__20260422_153000``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_slug = _persona_slug_for_pool(source_eval_dirs)
    gen = _generation_folder_for_eval(source_eval_dirs[0])
    if gen is not None:
        try:
            meta = parse_generation_run_folder_name(gen.name)
            agent = meta["agent"]
            t = meta["turns"]
            r = meta["runs"]
            return f"j_pooled__p_{persona_slug}__a_{agent}__t{t}__r{r}__{ts}"
        except ValueError:
            pass
    return f"j_pooled__p_{persona_slug}__a_unknown__t30__r1__{ts}"


def _annotate_pooled_results(
    results: dict[str, Any],
    combined: Any,
    source_eval_dirs: list[Path],
    rows_per_source: list[int],
) -> None:
    """
    Mutate *results* in place for pooled reporting and provenance.

    Sets top-level ``judge_model`` and ``persona_model`` to the literal ``"pooled"``
    (so charts and JSON match user expectations) and adds a ``pooled`` object with
    source paths, per-source row counts, total rows, and distinct ``judge_model``
    values present in the merged dataframe.

    Args:
        results: Score dict returned by ``score_results`` (modified in place).
        combined: Merged judge results dataframe (pandas ``DataFrame``).
        source_eval_dirs: Evaluation directories that were concatenated, in order.
        rows_per_source: Row count from each source, same order as *source_eval_dirs*.
    """
    judges = sorted(
        {str(x) for x in combined["judge_model"].dropna().astype(str).unique()}
    )
    results["judge_model"] = "pooled"
    results["persona_model"] = "pooled"
    results["pooled"] = {
        "source_evaluation_directories": [str(p) for p in source_eval_dirs],
        "rows_per_source": rows_per_source,
        "total_rows": int(len(combined)),
        "unique_judge_models_in_data": judges,
    }


def pool_evaluation_directories(
    source_paths: list[str | Path],
    output_parent: Path,
    *,
    personas_tsv: Path | None = None,
    skip_risk_analysis: bool = False,
) -> Path:
    """
    Merge several judge runs into one ``results.csv`` and compute VERA artifacts.

    Loads each evaluation via ``ensure_results_csv`` (rebuilding from TSVs if needed),
    concatenates rows, then writes merged ``results.csv``, ``pool_metadata.json``, and
    scoring artifacts under a new synthetic ``j_pooled__...`` directory.

    Args:
        source_paths: Paths to ``j_*`` folders or to ``results.csv`` inside them.
        output_parent: Directory under which the new ``j_pooled__...`` folder is
            created (e.g. repo ``output/``).
        personas_tsv: Personas file for risk-level analysis; defaults to
            ``data/personas.tsv`` under the repo when None.
        skip_risk_analysis: When True, skip ``score_results_by_risk`` and risk charts.

    Returns:
        Path to the synthetic evaluation folder (``results.csv`` and ``scores/``).

    Raises:
        FileNotFoundError: If a source path does not exist or is not a directory after
            resolving ``results.csv`` to its parent.
        ValueError: If the merged dataframe has no rows.
    """
    import pandas as pd

    from judge.score import (
        _save_results_json,
        print_scores,
        score_results,
        score_results_by_risk,
    )
    from judge.score_utils import ensure_results_csv
    from judge.score_viz import (
        create_risk_level_visualizations,
        create_visualizations,
    )

    eval_dirs = [_resolve_eval_input(Path(p)) for p in source_paths]
    for d in eval_dirs:
        if not d.is_dir():
            raise FileNotFoundError(d)

    dfs: list[pd.DataFrame] = []
    rows_per_source: list[int] = []
    for d in eval_dirs:
        df = ensure_results_csv(d)
        rows_per_source.append(len(df))
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    if len(combined) == 0:
        raise ValueError("Combined results dataframe is empty.")

    missing_gen = [d for d in eval_dirs if _generation_folder_for_eval(d) is None]
    if missing_gen:
        preview = "; ".join(str(p) for p in missing_gen[:3])
        if len(missing_gen) > 3:
            preview += f"; ... and {len(missing_gen) - 3} more"
        print(
            "Warning: some evaluation paths are not under .../p_*/evaluations/j_* "
            f"({preview}). "
            "Merged results and scores are unchanged; the new j_pooled__* folder name "
            "may use unknown placeholders for persona/agent/turns/runs. "
            "Use nested paths from generate.py / run_pipeline.py for descriptive "
            "names.",
            file=sys.stderr,
        )

    synth_name = _synthetic_pooled_folder_basename(eval_dirs)
    out_eval = (output_parent / synth_name).resolve()
    out_eval.mkdir(parents=True, exist_ok=True)
    results_csv = out_eval / "results.csv"
    combined.to_csv(results_csv, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_evaluation_directories": [str(p) for p in eval_dirs],
        "rows_per_source": rows_per_source,
        "pooled_results_csv": str(results_csv),
    }
    (out_eval / "pool_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    scores_json = out_eval / "scores" / "scores.json"
    results = score_results(str(results_csv), str(scores_json))
    _annotate_pooled_results(results, combined, eval_dirs, rows_per_source)
    _save_results_json(results, str(results_csv), str(scores_json))

    print_scores(results)

    viz_path = out_eval / "scores" / "scores_visualization.png"
    try:
        create_visualizations(results, viz_path)
    except Exception as e:
        print(f"Warning: could not create standard visualizations: {e}")

    personas_tsv = personas_tsv or (REPO_ROOT / "data" / "personas.tsv")
    if not skip_risk_analysis and personas_tsv.is_file():
        try:
            risk_results = score_results_by_risk(
                str(results_csv), str(personas_tsv), write_json=False
            )
            risk_results["judge_model"] = "pooled"
            risk_results["persona_model"] = "pooled"
            risk_json = out_eval / "scores" / "scores_by_risk.json"
            risk_json.parent.mkdir(parents=True, exist_ok=True)
            risk_json.write_text(json.dumps(risk_results, indent=2), encoding="utf-8")
            risk_viz = out_eval / "scores" / "scores_by_risk_visualization.png"
            create_risk_level_visualizations(risk_results, risk_viz)
        except Exception as e:
            print(f"Warning: could not create risk-level analysis: {e}")
    elif not skip_risk_analysis:
        print(
            f"Warning: personas TSV not found ({personas_tsv}), skipping risk analysis."
        )

    print("")
    print("Pooled outputs:")
    print(f"  {results_csv}")
    print(f"  {scores_json}")
    print(f"  {viz_path}")
    risk_json = out_eval / "scores" / "scores_by_risk.json"
    if risk_json.is_file():
        print(f"  {risk_json}")
        print(f"  {out_eval / 'scores' / 'scores_by_risk_visualization.png'}")
    print(f"  {out_eval / 'pool_metadata.json'}")
    return out_eval


def _cli_pool(args: argparse.Namespace) -> int:
    """
    Handle the ``pool_vera_scores`` CLI in merge mode (one or more evaluation paths).

    Resolves ``--output-dir`` (defaulting to the repo ``output/`` directory), then
    delegates to :func:`pool_evaluation_directories`.

    Args:
        args: Parsed namespace with ``eval_paths``, ``output_dir``, ``personas_tsv``,
            and ``skip_risk_analysis``.

    Returns:
        Process exit code: ``0`` on success, ``2`` if no evaluation paths were given.
    """
    if len(args.eval_paths) < 1:
        print(
            "error: pass at least one evaluation directory or results.csv",
            file=sys.stderr,
        )
        return 2

    out_parent = Path(args.output_dir).resolve() if args.output_dir else None
    if out_parent is None:
        out_parent = (REPO_ROOT / "output").resolve()
    out_parent.mkdir(parents=True, exist_ok=True)

    personas = Path(args.personas_tsv).resolve() if args.personas_tsv else None
    pool_evaluation_directories(
        args.eval_paths,
        out_parent,
        personas_tsv=personas,
        skip_risk_analysis=args.skip_risk_analysis,
    )
    return 0


def _cli_extract(args: argparse.Namespace) -> int:
    """
    Handle ``--extract-from-log``: print the last evaluation directory path.

    Reads the log file as UTF-8 (replacing undecodable bytes), parses it with
    :func:`extract_last_evaluation_dir_from_pipeline_log`, and prints the resolved
    path to stdout for use in shell command substitution.

    Args:
        args: Parsed namespace with ``extract_from_log`` set to the log file path.

    Returns:
        ``0`` on success, ``1`` if no evaluation line was found in the log.
    """
    log_path = Path(args.extract_from_log)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    try:
        path = extract_last_evaluation_dir_from_pipeline_log(text)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


def main() -> int:
    """
    CLI entry point: parse arguments and run pool or extract mode.

    In extract mode (``--extract-from-log``), runs :func:`_cli_extract` and exits.
    Otherwise requires at least one evaluation path and runs :func:`_cli_pool`.

    Returns:
        Process exit code from the selected subcommand.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pool multiple judge evaluation runs into one scored folder, "
            "or extract an evaluation path from run_pipeline log output."
        )
    )
    parser.add_argument(
        "eval_paths",
        nargs="*",
        help="Evaluation directories (j_*) or results.csv paths to merge",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Parent directory for pooled output (a synthetic j_pooled__* folder is "
            "created inside, with pool_metadata.json next to results.csv). "
            "Default: output/ under the repo."
        ),
    )
    parser.add_argument(
        "--personas-tsv",
        default=str(REPO_ROOT / "data" / "personas.tsv"),
        help="Personas file for risk-level scoring (default: data/personas.tsv)",
    )
    parser.add_argument(
        "--skip-risk-analysis",
        action="store_true",
        help="Skip risk-level scores and visualization",
    )
    parser.add_argument(
        "--extract-from-log",
        metavar="FILE",
        help="Print the last evaluation directory path found in a pipeline log file",
    )
    args = parser.parse_args()

    if args.extract_from_log:
        return _cli_extract(args)

    if not args.eval_paths:
        parser.error("pass at least one eval path, or use --extract-from-log")
    return _cli_pool(args)


if __name__ == "__main__":
    raise SystemExit(main())

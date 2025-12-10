"""
Judge Runner - High-level functions for batch conversation evaluation.
Contains the main logic extracted from main_judge.py to reduce code duplication.
"""

import asyncio
import os
from asyncio import Queue
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .llm_judge import LLMJudge
from .utils import load_rubric_structure

# In case this needs to be synced in the meta prompt for the judge
EVALUATION_SEPARATOR = ":"

# Load dimensions from rubric
DIMENSIONS, _ = load_rubric_structure("data/rubric.tsv")


def _parse_evaluation_to_dict(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse evaluation results into a flat dictionary.

    Args:
        evaluation: Raw evaluation dict with shape:
            {dimension: {"score": str, "reasoning": str,
                        "yes_question_id": str, "yes_reasoning": str}}

    Returns:
        Flattened dict with dimension scores and metadata
    """
    evaluation_dict = {}
    for dimension, values in evaluation.items():
        evaluation_dict[dimension] = values["score"]
        evaluation_dict[f"{dimension}_yes_question_id"] = values.get(
            "yes_question_id", ""
        )
        evaluation_dict[f"{dimension}_yes_reasoning"] = values.get("yes_reasoning", "")
    return evaluation_dict


async def _evaluate_single_conversation_with_judge(
    conversation_file: str,
    judge_model: str,
    judge_instance: int,
    output_folder: str,
) -> Dict[str, Any]:
    """
    Evaluate a single conversation with a single judge model instance.

    Args:
        conversation_file: Path to conversation file
        judge_model: Model name to use for judging
        judge_instance: Instance number for this judge (1, 2, 3, ...)
        output_folder: Folder to save evaluation results

    Returns:
        Dict with filename, run_id, judge_model, judge_instance,
        and all dimension scores
    """
    judge = LLMJudge(judge_model=judge_model)

    evaluation = await judge.evaluate_conversation_question_flow(
        conversation_file,
        output_folder=output_folder,
        auto_save=True,
        verbose=False,
    )

    try:
        evaluation_dict = _parse_evaluation_to_dict(evaluation)
    except Exception as e:
        print(
            f"Error parsing evaluation for {judge_model} instance {judge_instance}: {e}"
        )
        print("The following dict is malformed:")
        print(evaluation)
        evaluation_dict = {}

    return {
        "filename": Path(conversation_file).name,
        "run_id": Path(conversation_file).parent.name,
        "judge_model": judge_model,
        "judge_instance": judge_instance,
        **evaluation_dict,
    }


def _create_evaluation_jobs(
    conversation_file_paths: List[str],
    judge_models: Dict[str, int],
    output_folder: str,
) -> List[Tuple[str, str, int, str]]:
    """
    Create job tuples for all (conversation × judge × instance) combinations.

    Args:
        conversation_file_paths: List of conversation file paths
        judge_models: Dict mapping model names to number of instances
        output_folder: Folder to save evaluation results

    Returns:
        List of job tuples (conversation_file, judge_model, instance, output_folder)
    """
    jobs = []
    for conversation_file in conversation_file_paths:
        for judge_model, num_instances in judge_models.items():
            for instance in range(1, num_instances + 1):
                jobs.append((conversation_file, judge_model, instance, output_folder))
    return jobs


async def _worker(
    worker_id: int,
    queue: Queue,
    results: List[Dict[str, Any]],
    total_jobs: int,
    judge_model_filter: Optional[str] = None,
):
    """
    Worker that processes evaluation jobs from the queue.

    Args:
        worker_id: Unique identifier for this worker
        queue: Queue containing job tuples
        results: Shared list to append results to
        total_jobs: Total number of jobs for progress tracking
        judge_model_filter: If set, only process jobs for this judge model
    """
    while True:
        try:
            job = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        conversation_file, judge_model, instance, output_folder = job

        # Skip if worker is filtered to a specific judge model
        if judge_model_filter and judge_model != judge_model_filter:
            queue.task_done()
            continue

        completed = len(results)
        print(
            f"[Worker {worker_id}] ({completed + 1}/{total_jobs}) "
            f"{Path(conversation_file).name} | {judge_model} (instance {instance})"
        )

        try:
            result = await _evaluate_single_conversation_with_judge(
                conversation_file, judge_model, instance, output_folder
            )
            results.append(result)
        except Exception as e:
            print(
                f"[Worker {worker_id}] Failed to evaluate "
                f"{Path(conversation_file).name} with {judge_model}: {e}"
            )

        queue.task_done()


async def _run_workers_with_queue(
    jobs: List[Tuple[str, str, int, str]],
    max_concurrent: Optional[int],
    per_judge: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run evaluation jobs using a worker queue with concurrency control.

    Args:
        jobs: List of job tuples
              (conversation_file, judge_model, instance, output_folder)
        max_concurrent: Maximum number of concurrent workers (None = unlimited)
        per_judge: If True, max_concurrent applies per judge model;
                  if False, total

    Returns:
        List of evaluation results
    """
    results = []
    total_jobs = len(jobs)

    if per_judge:
        # Group jobs by judge model
        jobs_by_model = {}
        for job in jobs:
            judge_model = job[1]
            if judge_model not in jobs_by_model:
                jobs_by_model[judge_model] = []
            jobs_by_model[judge_model].append(job)

        # Create a queue per judge model
        all_workers = []
        for judge_model, model_jobs in jobs_by_model.items():
            queue = Queue()
            for job in model_jobs:
                await queue.put(job)

            # Determine number of workers for this model
            num_workers = len(model_jobs) if max_concurrent is None else max_concurrent
            print(
                f"Starting {num_workers} workers for {judge_model} "
                f"({len(model_jobs)} jobs)"
            )

            # Create workers for this judge model
            workers = [
                asyncio.create_task(
                    _worker(
                        f"{judge_model[:10]}-{i}",
                        queue,
                        results,
                        total_jobs,
                        judge_model_filter=None,
                    )
                )
                for i in range(num_workers)
            ]
            all_workers.extend(workers)

        # Wait for all workers to complete
        await asyncio.gather(*all_workers)

    else:
        # Single queue for all jobs
        queue = Queue()
        for job in jobs:
            await queue.put(job)

        # Determine number of workers
        num_workers = total_jobs if max_concurrent is None else max_concurrent
        print(f"Starting {num_workers} workers for all jobs ({total_jobs} total)")

        # Create workers
        workers = [
            asyncio.create_task(
                _worker(i, queue, results, total_jobs, judge_model_filter=None)
            )
            for i in range(num_workers)
        ]

        # Wait for all workers to complete
        await asyncio.gather(*workers)

    return results


async def batch_evaluate_with_individual_judges(
    conversation_file_paths: List[str],
    rubrics: List[str],
    judge_models: Dict[str, int],
    output_folder: str,
    limit: Optional[int],
    max_concurrent: Optional[int],
    per_judge: bool,
) -> List[Dict[str, Any]]:
    """
    Evaluate conversations with multiple judge models using queue workers.

    Each conversation is evaluated by all judge models and their instances.
    Workers process jobs from a queue with configurable concurrency.

    Args:
        conversation_file_paths: List of conversation file paths
        rubrics: List of rubric names to use (currently unused)
        judge_models: Dict mapping model names to number of instances
                     Example: {"claude-3-7-sonnet": 3, "gpt-4": 2}
        output_folder: Folder to save evaluation results
        limit: Optional limit on number of conversations to evaluate
        max_concurrent: Maximum number of concurrent workers
        per_judge: If True, max_concurrent applies per judge model; if False, total

    Returns:
        Flattened list of evaluation results with one row per
        (conversation, judge_model, judge_instance) tuple
    """
    if limit is not None:
        conversation_file_paths = conversation_file_paths[:limit]

    total_files = len(conversation_file_paths)
    total_judge_instances = sum(judge_models.values())
    total_evaluations = total_files * total_judge_instances

    print(
        f"Evaluating {total_files} conversations with "
        f"{len(judge_models)} judge models "
        f"({total_judge_instances} total instances)..."
    )
    print(f"Total evaluations to run: {total_evaluations}")
    print(
        f"Concurrency: {max_concurrent} workers "
        f"({'per judge model' if per_judge else 'total'})"
    )

    # Create all evaluation jobs
    jobs = _create_evaluation_jobs(conversation_file_paths, judge_models, output_folder)

    # Run workers with queue
    results = await _run_workers_with_queue(jobs, max_concurrent, per_judge)

    print(f"Completed {len(results)}/{total_evaluations} evaluations successfully")
    return results


async def judge_conversations(
    judge_models: Dict[str, int],
    conversation_folder: str,
    rubrics: List[str] = ["rubric.csv"],
    output_root: str = "evaluations",
    limit: Optional[int] = None,
    verbose: bool = True,
    output_folder: Optional[str] = None,
    save_aggregated_results: bool = True,
    filename: Optional[str] = "results.csv",
    max_concurrent: Optional[int] = None,
    per_judge: bool = False,
) -> List[Dict[str, Any]]:
    """
    Judge conversations in a folder with multiple judge models.

    Args:
        judge_models: Dict mapping model names to number of instances
                     Example: {"claude-3-7-sonnet": 3, "gpt-4": 2}
        conversation_folder: Folder containing conversation files
        rubrics: List of rubric names to use
        output_root: Root folder for evaluation outputs
        limit: Optional limit on number of conversations to process
        verbose: Whether to print status messages
        output_folder: Custom output folder (auto-generated if None)
        save_aggregated_results: Whether to save results to CSV
        filename: Name for aggregated results CSV
        max_concurrent: Maximum number of concurrent workers
        per_judge: If True, max_concurrent applies per judge model; if False, total

    Returns:
        Flattened list of evaluation results with one row per
        (conversation, judge_model, judge_instance) tuple

    Raises:
        FileNotFoundError: If folder or files not found
    """
    if output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        judges_str = "_".join(
            f"{model}x{count}" for model, count in judge_models.items()
        )
        output_folder = (
            f"{output_root}/j_{judges_str}_{timestamp}__"
            f"{Path(conversation_folder).name}"
        )

    os.makedirs(output_folder, exist_ok=True)

    # Check folder exists
    folder_path = Path(conversation_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {conversation_folder}")

    # Find conversation files
    conversation_files = list(folder_path.glob("*.txt"))
    if not conversation_files:
        raise FileNotFoundError(f"No .txt files found in: {conversation_folder}")

    total_found = len(conversation_files)

    if limit:
        conversation_files = conversation_files[:limit]
        if verbose:
            print(f"🔍 Found {total_found} files, judging {limit} (debug mode)")
    else:
        if verbose:
            print(f"🔍 Found {total_found} files to judge")

    # Convert to strings
    conversation_file_paths = [str(f) for f in conversation_files]

    # Run batch evaluation with multiple judges
    results = await batch_evaluate_with_individual_judges(
        conversation_file_paths,
        rubrics,
        judge_models,
        output_folder,
        limit,
        max_concurrent,
        per_judge,
    )

    if save_aggregated_results:
        # Column order: filename, run_id, judge_model, judge_instance, dimensions
        if results:
            columns = ["filename", "run_id", "judge_model", "judge_instance"] + [
                k
                for k in results[0].keys()
                if k not in ["filename", "run_id", "judge_model", "judge_instance"]
            ]
        else:
            # Use known dimensions from rubric
            columns = ["filename", "run_id", "judge_model", "judge_instance"] + DIMENSIONS
        pd.DataFrame(results, columns=columns).to_csv(
            f"{output_folder}/{filename}", index=False
        )
    if verbose:
        print(f"✅ Completed {len(results)} evaluations → {output_folder}/")

    return results


async def judge_single_conversation(
    judge: LLMJudge, conversation_file: str, rubrics: List[str], output_folder: str
) -> Optional[Dict[str, Any]]:
    """
    Judge a single conversation file.

    Args:
        judge: LLMJudge instance
        conversation_file: Path to conversation file
        rubrics: List of rubric names to use
        output_folder: Output folder for results

    Returns:
        Evaluation results or None if failed
    """
    if not Path(conversation_file).exists():
        print(f"❌ File not found: {conversation_file}")
        return None

    print(f"📄 Judging: {Path(conversation_file).name}")

    result = await judge.evaluate_conversation_question_flow(
        conversation_file, output_folder=output_folder, auto_save=True
    )

    print(f"🟢 Done: {Path(conversation_file).name}")
    return result

#!/usr/bin/env python3

import asyncio
import logging
import os
import time
import uuid
from asyncio import Queue
from datetime import datetime
from typing import AbstractSet, Any, Dict, List, Optional, Tuple

from llm_clients import LLMFactory
from llm_clients.llm_interface import LLMGenerationFailed, Role
from utils.logging_utils import (
    cleanup_logger,
    log_conversation_end,
    log_conversation_start,
    log_conversation_turn,
    setup_conversation_logger,
)
from utils.naming import (
    TRANSCRIPT_RUN_SUFFIX_RE,
    persona_token_for_transcript_stem,
)

from .conversation_simulator import ConversationSimulator
from .utils import load_prompts_from_csv


class ConversationRunner:
    """Handles running LLM conversations with logging and file management."""

    def __init__(
        self,
        persona_model_config: Dict[str, Any],
        agent_model_config: Dict[str, Any],
        run_id: str,
        max_turns: int = 6,
        runs_per_prompt: int = 3,
        folder_name: str = "conversations",
        max_concurrent: Optional[int] = None,
        max_total_words: Optional[int] = None,
        max_personas: Optional[int] = None,
        persona_speaks_first: bool = True,
        resume: bool = False,
    ):
        self.persona_model_config = persona_model_config
        self.agent_model_config = agent_model_config
        self.max_turns = max_turns
        self.runs_per_prompt = runs_per_prompt
        self.folder_name = folder_name
        self.run_id = run_id

        # Limit concurrent conversations to avoid overwhelming the server
        # Default: None - run all conversations concurrently
        self.max_concurrent = max_concurrent
        self.max_total_words = max_total_words
        self.max_personas = max_personas
        self.persona_speaks_first = persona_speaks_first
        self.resume = resume

    @staticmethod
    def _resolve_persona_safe_from_stem(
        stem: str, persona_safe_names: AbstractSet[str]
    ) -> Optional[str]:
        """
        Pick the longest persona_safe in persona_safe_names that matches this stem.

        Transcript stem (after tag_) is {persona_safe}_{model_name}; model_name may
        contain underscores, so we match known persona_safe names instead of splitting
        on underscores.

        Example: stem ``Anna_gemini-2.5-flash`` with ``{"Ann", "Anna"}`` in the set
        matches both prefixes; returns ``Anna`` (longest).
        """
        candidates = [
            p for p in persona_safe_names if stem == p or stem.startswith(f"{p}_")
        ]
        if not candidates:
            return None
        return max(candidates, key=len)

    def _parse_transcript_suffix_for_resume(
        self, suffix: str, persona_safe_names: AbstractSet[str]
    ) -> Optional[tuple[str, int]]:
        """Parse `{persona_safe}_{model}_run{N}.txt` using known persona_safe names."""
        match = TRANSCRIPT_RUN_SUFFIX_RE.search(suffix)
        if not match:
            return None
        run = int(match.group("run"))
        stem = suffix[: match.start()]
        persona_safe = self._resolve_persona_safe_from_stem(stem, persona_safe_names)
        if persona_safe is None:
            return None
        return (persona_safe, run)

    def _index_existing_conversations(
        self, persona_safe_names: AbstractSet[str]
    ) -> set[tuple[str, int]]:
        """
        Index existing transcript files as (persona_safe, run_number).

        Parses `_runN.txt` first, then resolves persona_safe via longest-prefix match
        against persona_safe_names so model segments with underscores do not corrupt
        the persona key.
        """
        existing: set[tuple[str, int]] = set()
        if not os.path.isdir(self.folder_name):
            return existing

        for filename in os.listdir(self.folder_name):
            if not filename.endswith(".txt"):
                continue
            parts = filename.split("_", 1)
            if len(parts) != 2:
                continue
            parsed = self._parse_transcript_suffix_for_resume(
                parts[1], persona_safe_names
            )
            if parsed is None:
                continue
            existing.add(parsed)
        return existing

    @staticmethod
    def _has_existing_transcript(
        persona_safe: str,
        run_number: int,
        existing_keys: set[tuple[str, int]],
    ) -> bool:
        """Return True when a transcript exists for this exact persona/run."""
        return (persona_safe, run_number) in existing_keys

    def _create_conversation_jobs(
        self, persona_names: Optional[List[str]] = None
    ) -> List[Tuple[dict, int, int, int]]:
        """Create job tuples for all persona/run combinations."""
        personas = load_prompts_from_csv(persona_names, max_personas=self.max_personas)
        jobs: List[Tuple[dict, int, int, int]] = []
        conversation_index = 1
        for persona in personas:
            for run in range(1, self.runs_per_prompt + 1):
                jobs.append(
                    (
                        {
                            "prompt": persona["prompt"],
                            "name": persona["Name"],
                            "run": run,
                        },
                        self.max_turns,
                        conversation_index,
                        run,
                    )
                )
                conversation_index += 1
        return jobs

    async def _worker(
        self,
        worker_id: int,
        queue: Queue,
        results: List[Dict[str, Any]],
        total_jobs: int,
    ) -> None:
        """Worker that processes conversation generation jobs from a queue."""
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                persona_config, max_turns, conversation_index, run_number = job
                conversation_name = persona_config.get("name", "Unknown")
                print(
                    f"[Worker {worker_id}] ({len(results) + 1}/{total_jobs}) "
                    f"{conversation_name} (run {run_number})"
                )
                result = await self.run_single_conversation(
                    persona_config=persona_config,
                    max_turns=max_turns,
                    conversation_index=conversation_index,
                    run_number=run_number,
                )
            except Exception as exc:
                # Don't let one failed job cancel all workers.
                # Keep result schema stable.
                result = {
                    "index": job[2] if len(job) > 2 else -1,
                    "llm1_model": self.persona_model_config.get("model", "unknown"),
                    "llm1_prompt": (
                        job[0].get("name", "Unknown")
                        if isinstance(job[0], dict)
                        else "Unknown"
                    ),
                    "run_number": job[3] if len(job) > 3 else 0,
                    "turns": 0,
                    "filename": None,
                    "log_file": None,
                    "duration": 0.0,
                    "early_termination": False,
                    "conversation": [],
                    "skipped": True,
                    "error": str(exc),
                }
                print(f"[Worker {worker_id}] Failed job: {result['error']}")
            finally:
                queue.task_done()

            results.append(result)

    async def run_single_conversation(
        self,
        persona_config: dict,
        max_turns: int,
        conversation_index: int,
        run_number: int,
        **kwargs: dict,
    ) -> Dict[str, Any]:
        """Run a single simulated conversation (persona vs provider LLM).

        Uses fresh LLM instances per call; safe for concurrent use. Logs turns,
        writes transcript to self.folder_name, then cleans up logger and LLMs.

        Args:
            persona_config (dict): Must have "prompt" and "name". Persona LLM
                identity comes from ``self.persona_model_config`` (including ``model``).
            max_turns (int): Max conversation turns for a conversation.
            conversation_index (int): Index in the batch of conversations.
            run_number (int): Run index for this prompt (e.g. 1 of runs_per_prompt).
            **kwargs: Unused; reserved for future use.

        Returns:
            Dict[str, Any]: index, llm1_model, llm1_prompt, run_number, turns,
            filename, log_file, duration, early_termination, conversation.
        """
        model_name = self.persona_model_config["model"]
        system_prompt = persona_config["prompt"]
        persona_name = persona_config["name"]

        # Generate filename base using persona name, model, and run number
        tag = uuid.uuid4().hex[:6]
        filename_base = f"{tag}_{persona_name}_{model_name}_run{run_number}"
        os.makedirs(f"{self.folder_name}", exist_ok=True)
        log_file_path = os.path.join("logging", self.run_id, f"{filename_base}.log")

        logger: Optional[logging.Logger] = None
        persona: Optional[Any] = None
        agent: Optional[Any] = None
        start_time = time.time()
        result: Optional[Dict[str, Any]] = None

        try:
            logger = setup_conversation_logger(filename_base, run_id=self.run_id)

            persona = LLMFactory.create_llm(
                model_name=model_name,
                name=f"{model_name} {persona_name}",
                system_prompt=system_prompt,
                role=Role.PERSONA,
                **self.persona_model_config,
            )

            agent_kwargs = {
                k: v
                for k, v in self.agent_model_config.items()
                if k not in ("model", "name", "system_prompt")
            }
            agent = LLMFactory.create_llm(
                model_name=self.agent_model_config["model"],
                name=self.agent_model_config.get("name", "Provider"),
                system_prompt=self.agent_model_config.get(
                    "system_prompt", "You are a helpful AI assistant."
                ),
                role=Role.PROVIDER,
                **agent_kwargs,
            )

            log_conversation_start(
                logger=logger,
                llm1_model_str=model_name,
                llm1_prompt=persona_name,
                llm2_name=agent.name,
                llm2_model_str=getattr(agent, "model_name", "unknown"),
                max_turns=max_turns,
                persona_speaks_first=self.persona_speaks_first,
                llm1_model=persona,
                llm2_model=agent,
            )

            simulator = ConversationSimulator(persona, agent)

            try:
                conversation = await simulator.generate_conversation(
                    max_turns=max_turns,
                    max_total_words=self.max_total_words,
                    persona_speaks_first=self.persona_speaks_first,
                )
            except LLMGenerationFailed as e:
                end_time = time.time()
                conversation_time = end_time - start_time
                print(f"Skipped conversation ({persona_name}, run {run_number}): {e}")
                logger.error(
                    "CONVERSATION FAILED | persona=%s run=%s error=%s",
                    persona_name,
                    run_number,
                    str(e),
                )
                result = {
                    "index": conversation_index,
                    "llm1_model": model_name,
                    "llm1_prompt": persona_name,
                    "run_number": run_number,
                    "turns": 0,
                    "filename": None,
                    "log_file": log_file_path,
                    "duration": conversation_time,
                    "early_termination": False,
                    "conversation": [],
                    "skipped": True,
                    "error": str(e),
                }
            else:
                for i, turn in enumerate(conversation, 1):
                    log_conversation_turn(
                        logger=logger,
                        turn_number=i,
                        speaker=turn.get("speaker", "Unknown"),
                        input_message=turn.get("input", ""),
                        response=turn.get("response", ""),
                        early_termination=turn.get("early_termination", False),
                        logging=turn.get("logging", {}),
                    )

                end_time = time.time()
                conversation_time = end_time - start_time
                early_termination = any(
                    turn.get("early_termination", False) for turn in conversation
                )

                log_conversation_end(
                    logger=logger,
                    total_turns=len(conversation),
                    early_termination=early_termination,
                    total_time=conversation_time,
                )

                simulator.save_conversation(f"{filename_base}.txt", self.folder_name)

                result = {
                    "index": conversation_index,
                    "llm1_model": model_name,
                    "llm1_prompt": persona_name,
                    "run_number": run_number,
                    "turns": len(conversation),
                    "filename": f"{self.folder_name}/{filename_base}.txt",
                    "log_file": log_file_path,
                    "duration": conversation_time,
                    "early_termination": early_termination,
                    "conversation": conversation,
                    "skipped": False,
                }
        except Exception as exc:
            end_time = time.time()
            if logger is not None:
                logger.error(
                    "RUN FAILED | persona=%s run=%s error=%s",
                    persona_name,
                    run_number,
                    str(exc),
                )
            result = {
                "index": conversation_index,
                "llm1_model": model_name,
                "llm1_prompt": persona_name,
                "run_number": run_number,
                "turns": 0,
                "filename": None,
                "log_file": log_file_path,
                "duration": end_time - start_time,
                "early_termination": False,
                "conversation": [],
                "skipped": True,
                "skip_reason": "error",
                "error": str(exc),
            }
            print(f"Skipped conversation ({persona_name}, run {run_number}): {exc}")
        finally:
            if logger is not None:
                cleanup_logger(logger)

            for llm in (persona, agent):
                if llm is not None:
                    try:
                        await llm.cleanup()
                    except Exception as e:
                        print(f"Warning: Failed to cleanup LLM: {e}")

        assert result is not None
        return result

    async def run_conversations(
        self, persona_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run multiple conversations concurrently using queue workers."""
        personas = load_prompts_from_csv(persona_names, max_personas=self.max_personas)
        persona_safe_names = {
            persona_token_for_transcript_stem(p["Name"]) for p in personas
        }
        existing_keys = (
            self._index_existing_conversations(persona_safe_names)
            if self.resume
            else set()
        )

        # Create jobs for all conversations (each prompt run multiple times)
        jobs: List[Tuple[dict, int, int, int]] = []
        skipped_results: List[Dict[str, Any]] = []
        conversation_index = 1

        for persona in personas:
            for run in range(1, self.runs_per_prompt + 1):
                persona_safe = persona_token_for_transcript_stem(persona["Name"])
                if self._has_existing_transcript(persona_safe, run, existing_keys):
                    skipped_results.append(
                        {
                            "index": conversation_index,
                            "llm1_model": self.persona_model_config["model"],
                            "llm1_prompt": persona["Name"],
                            "run_number": run,
                            "turns": 0,
                            "filename": None,
                            "log_file": None,
                            "duration": 0.0,
                            "early_termination": False,
                            "conversation": [],
                            "skipped": True,
                            "skip_reason": "existing",
                            "error": "Transcript already exists in output folder",
                        }
                    )
                    conversation_index += 1
                    continue
                jobs.append(
                    (
                        {
                            "model": self.persona_model_config["model"],
                            "prompt": persona["prompt"],
                            "name": persona["Name"],
                            "run": run,
                        },
                        self.max_turns,
                        conversation_index,
                        run,
                    )
                )
                conversation_index += 1

        total_jobs = len(jobs)
        start_time = datetime.now()
        queue: Queue = Queue()
        for job in jobs:
            await queue.put(job)

        if self.max_concurrent is not None and self.max_concurrent < 0:
            raise ValueError(
                "max_concurrent must be None, 0 (no limit), or a positive integer"
            )

        if self.max_concurrent in (None, 0):
            num_workers = total_jobs
            print(f"Running {total_jobs} conversations concurrently (no limit)")
        else:
            num_workers = self.max_concurrent
            print(
                f"Running {total_jobs} conversations with max concurrency: "
                f"{self.max_concurrent}"
            )

        results: List[Dict[str, Any]] = []
        workers = [
            asyncio.create_task(self._worker(i, queue, results, total_jobs))
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

        results = skipped_results + results

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        skipped_n = sum(1 for r in results if r.get("skipped"))
        skipped_existing_n = sum(
            1
            for r in results
            if r.get("skipped") and r.get("skip_reason") == "existing"
        )
        skipped_error_n = sum(
            1 for r in results if r.get("skipped") and r.get("skip_reason") == "error"
        )
        print(
            f"\nCompleted {len(results) - skipped_n} / {len(results)} "
            f"conversations in "
            f"{total_time:.2f} seconds"
        )
        if skipped_existing_n:
            print(f"  ({skipped_existing_n} skipped: transcript already exists)")
        if skipped_error_n:
            print(f"  ({skipped_error_n} skipped due to errors)")

        return results

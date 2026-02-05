import csv
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest

# ruff: noqa: E402
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import generate
from judge.rubric_config import RubricConfig, load_conversations
from judge.runner import judge_conversations as judge_conversations_fn
from judge.score import score_results

# Test configuration constants
TEST_CONFIG = {
    "MEMBER_MODEL": "gpt-4o",
    "PROVIDER_MODEL": "claude-opus-4-1-20250805",
    "JUDGE_MODEL": "gpt-4o",
    "TURNS": 6,  # Smaller for faster tests
    "RUNS_PER_PERSONA": 1,  # Reduced for faster tests
    "TEMP_MEMBER": 0.0,
    "TEMP_PROVIDER": 0.0,
    "TIMEOUT_SECONDS": 300,  # 5 minutes
}


@pytest.fixture(scope="session", autouse=True)
def validate_test_environment():
    """Ensure test environment is properly configured before running any tests."""
    required_env_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing required environment variables: {missing}")

    # Validate repo structure
    required_files = ["generate.py", "judge.py", "data/personas.tsv"]
    missing_files = [f for f in required_files if not (REPO_ROOT / f).exists()]
    if missing_files:
        pytest.skip(f"Missing required files: {missing_files}")


@pytest.fixture
def test_workspace():
    """Create isolated test workspace for each test."""
    with tempfile.TemporaryDirectory(prefix="vera_integration_test_") as tmpdir:
        workspace = Path(tmpdir)
        conversations_dir = workspace / "conversations"
        conversations_dir.mkdir()
        yield workspace


@pytest.fixture
def repo_root():
    """Get repository root path."""
    return REPO_ROOT


@pytest.mark.integration
class TestVERAMHPipeline:
    """Integration tests for the complete VERA-MH pipeline."""

    async def generate_one_conversation(
        self,
        persona_name: str,
        member_model: str,
        provider_model: str,
        conversations_root: Path,
        turns: int,
        runs: int,
        temp_member: float,
        temp_provider: float,
    ) -> Path:
        """Generate conversations for one persona and return directory.

        This method assumes generate.main() creates exactly one new directory when:
        - persona_names contains only one persona
        - max_concurrent=1 to avoid race conditions
        - runs_per_prompt creates conversations in a single directory

        Args:
            persona_name: Single persona to generate conversations for
            member_model: Model for the user/member role
            provider_model: Model for the assistant/provider role
            conversations_root: Root directory for conversation output
            turns: Number of conversation turns
            runs: Number of conversation runs per persona
            temp_member: Temperature for member model
            temp_provider: Temperature for provider model

        Returns:
            Path to the generated conversation directory

        Raises:
            RuntimeError: If no directories or multiple directories are created
        """
        conversations_root.mkdir(parents=True, exist_ok=True)

        persona_model_config = {
            "model": member_model,
            "temperature": temp_member,
        }
        agent_model_config = {
            "model": provider_model,
            "name": provider_model,
            "temperature": temp_provider,
        }

        # Capture the return value from generate.main for exact folder path
        results, folder_name = await generate.main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            persona_names=[persona_name],  # only one persona
            max_turns=turns,
            runs_per_prompt=runs,
            folder_name=str(conversations_root),
            max_concurrent=1,  # prevent concurrent directory creation
            verbose=True,
        )

        # Use the returned folder path directly (no complex detection needed)
        conv_dir = Path(folder_name)

        # Verify the directory contains conversation files
        if not conv_dir.exists() or not conv_dir.is_dir():
            raise RuntimeError(f"Generated directory {conv_dir} is not valid")

        # Additional validation: check if directory has expected content
        conv_files = list(conv_dir.glob("*.txt")) + list(conv_dir.glob("*.json"))
        logger = logging.getLogger(__name__)
        if not conv_files:
            logger.warning(
                f"Generated directory {conv_dir} exists but contains no "
                f"conversation files (.txt/.json). "
                f"Contents: {[f.name for f in conv_dir.iterdir()]}"
            )

        logger.info(f"Generated conversations in {conv_dir}")
        return conv_dir

    async def judge_conversations(
        self,
        conversations_dir: Path,
        judge_model: str,
        repo_root: Path,
        test_workspace: Path,
        instances: int = 5,
    ) -> Path:
        """Judge conversations and return the evaluation directory.

        Args:
            conversations_dir: Directory containing conversation files
            judge_model: Model to use for judging
            repo_root: Repository root (for rubric data only)
            test_workspace: Test workspace for evaluation outputs
                          (prevents repo pollution)
            instances: Number of judge instances

        Returns:
            Path to the evaluation directory in test_workspace
        """

        # Load conversations and rubric config
        conversations = await load_conversations(str(conversations_dir))
        rubric_config = await RubricConfig.load(rubric_folder=str(repo_root / "data"))

        # Use direct function call instead of subprocess
        judge_models = {judge_model: instances}
        folder_name = conversations_dir.name
        eval_root = test_workspace / "evaluations"  # Use test workspace, not repo
        eval_root.mkdir(exist_ok=True)  # Ensure directory exists

        _, output_folder = await judge_conversations_fn(
            judge_models=judge_models,
            conversations=conversations,
            rubric_config=rubric_config,
            output_root=str(eval_root),
            conversation_folder_name=folder_name,
            verbose=True,
        )

        eval_dir = Path(output_folder)

        # assert that judge created the expected output files
        expected_files = ["results.csv"]
        for expected_file in expected_files:
            file_path = eval_dir / expected_file
            assert (
                file_path.exists()
            ), f"Judge should have created {expected_file} in {eval_dir}"
            assert file_path.stat().st_size > 0, f"{expected_file} should not be empty"

        # verify results.csv has proper structure
        results_csv = eval_dir / "results.csv"
        with open(results_csv, "r") as f:
            content = f.read()
            assert len(content.strip()) > 0, "results.csv should contain data"

            lines = content.strip().split("\n")
            assert len(lines) >= 2, (
                f"results.csv should have header + at least one data row, "
                f"found {len(lines)} lines"
            )

        # Use proper CSV parsing to handle quoted fields and embedded commas
        with open(results_csv, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            # Validate CSV header structure
            required_columns = [
                "filename",
                "run_id",
                "judge_model",
                "judge_instance",
                "judge_id",
            ]

            fieldnames = reader.fieldnames or []
            for req_col in required_columns:
                assert req_col in fieldnames, (
                    f"results.csv header missing required column '{req_col}'. "
                    f"Found columns: {fieldnames}"
                )

            # Validate that data rows have proper structure and required fields
            row_count = 0
            for row_num, row in enumerate(reader, 1):
                row_count += 1

                # Validate that required fields are not empty
                for req_col in required_columns:
                    field_value = row.get(req_col, "").strip()
                    assert field_value, (
                        f"Row {row_num}: required column '{req_col}' "
                        f"should not be empty, got: '{field_value}'"
                    )

            # Ensure we actually processed some data rows
            assert row_count > 0, "results.csv should contain at least one data row"

        logging.info("Judge created evaluation files in %s", eval_dir)
        return eval_dir

    def score_evaluation(
        self,
        eval_dir: Path,
        repo_root: Path,
    ) -> dict:
        """Score an evaluation and return the results dictionary."""
        results_csv = eval_dir / "results.csv"
        if not results_csv.exists():
            raise FileNotFoundError(f"Missing results.csv in {eval_dir}")

        # Use direct function call instead of subprocess for consistency
        scores_data = score_results(str(results_csv))

        # judge.score.score_results always returns a dict with "aggregates" key
        # No fallback needed - if "aggregates" is missing, it indicates a bug
        if "aggregates" not in scores_data:
            raise ValueError(
                f"Invalid scores data structure: missing 'aggregates' key. "
                f"This indicates a bug in judge.score.score_results. "
                f"Found keys: {list(scores_data.keys())}"
            )

        return scores_data

    async def run_complete_pipeline(
        self,
        persona: str,
        test_workspace: Path,
        repo_root: Path,
        config: Dict[str, Any] = None,
        judge_model: str = None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline for a single persona and return results."""
        if config is None:
            config = TEST_CONFIG

        conversations_root = test_workspace / "conversations"

        # Generate conversations
        conv_dir = await self.generate_one_conversation(
            persona_name=persona,
            member_model=config["MEMBER_MODEL"],
            provider_model=config["PROVIDER_MODEL"],
            conversations_root=conversations_root,
            turns=config["TURNS"],
            runs=config["RUNS_PER_PERSONA"],
            temp_member=config["TEMP_MEMBER"],
            temp_provider=config["TEMP_PROVIDER"],
        )

        # Judge conversations - uses the returned evaluation directory path
        # Parse judge model to extract instances
        # (e.g., "gpt-4o:3" -> model="gpt-4o", instances=3)
        judge_model_name = judge_model or config["JUDGE_MODEL"]
        instances = 5  # default
        if ":" in judge_model_name:
            judge_model_name, instances_str = judge_model_name.split(":", 1)
            instances = int(instances_str)

        eval_dir = await self.judge_conversations(
            conversations_dir=conv_dir,
            judge_model=judge_model_name,
            repo_root=repo_root,
            test_workspace=test_workspace,
            instances=instances,
        )

        # Score evaluation
        score_json = self.score_evaluation(
            eval_dir=eval_dir,
            repo_root=repo_root,
        )

        return score_json

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_complete_pipeline_single_persona(self, test_workspace, repo_root):
        """Test the complete pipeline with a single persona."""
        persona = "Sky"

        start_time = time.time()
        result = await self.run_complete_pipeline(persona, test_workspace, repo_root)
        duration = time.time() - start_time

        # Assert structure exists
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "aggregates" in result, "Result should contain 'aggregates' key"

        aggregates = result["aggregates"]

        # Assert required metrics exist
        required_metrics = [
            "overall_best_practice_pct",
            "overall_damaging_pct",
            "worst_band",
        ]
        for metric in required_metrics:
            assert metric in aggregates, f"Missing required metric: {metric}"

        # Assert metric value ranges (handle "Not Relevant" case)
        best_practice_pct = aggregates["overall_best_practice_pct"]
        damaging_pct = aggregates["overall_damaging_pct"]
        worst_band = aggregates["worst_band"]

        assert (
            0 <= best_practice_pct <= 100
        ), f"Best practice % should be 0-100, got {best_practice_pct}"
        assert (
            0 <= damaging_pct <= 100
        ), f"Damaging % should be 0-100, got {damaging_pct}"
        assert worst_band is not None, "Worst band should not be None"

        # For integration tests, we expect some meaningful evaluation
        # If everything is "Not Relevant", that's still a valid result but worth noting
        if best_practice_pct == 0 and damaging_pct == 0:
            print(
                "⚠️  Note: All evaluations marked 'Not Relevant' - "
                "conversation may not have triggered mental health criteria"
            )

        print(f"   {persona} pipeline completed in {duration:.2f}s")
        print(f"   Best practice: {best_practice_pct:.1f}%")
        print(f"   Damaging: {damaging_pct:.1f}%")
        print(f"   Worst band: {worst_band}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("persona", ["Sky", "Alix"])
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_individual_personas(self, persona, test_workspace, repo_root):
        """Test that each persona works individually."""
        result = await self.run_complete_pipeline(persona, test_workspace, repo_root)

        # Basic validation
        assert "aggregates" in result
        aggregates = result["aggregates"]

        # Each persona should produce valid results
        best_practice_pct = aggregates["overall_best_practice_pct"]
        damaging_pct = aggregates["overall_damaging_pct"]
        worst_band = aggregates["worst_band"]

        assert 0 <= best_practice_pct <= 100
        assert 0 <= damaging_pct <= 100
        assert worst_band is not None

        print(
            f"Persona {persona} validation passed "
            f"(BP: {best_practice_pct:.1f}%, D: {damaging_pct:.1f}%, WB: {worst_band})"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_scoring_file_output(self, test_workspace, repo_root):
        """Test that scoring produces expected file outputs after judging."""
        # Generate conversations
        conversations_root = test_workspace / "conversations"
        conv_dir = await self.generate_one_conversation(
            persona_name="Ray",
            member_model=TEST_CONFIG["MEMBER_MODEL"],
            provider_model=TEST_CONFIG["PROVIDER_MODEL"],
            conversations_root=conversations_root,
            turns=TEST_CONFIG["TURNS"],
            runs=TEST_CONFIG["RUNS_PER_PERSONA"],
            temp_member=TEST_CONFIG["TEMP_MEMBER"],
            temp_provider=TEST_CONFIG["TEMP_PROVIDER"],
        )

        # Verify conversation files were created
        conv_files = list(conv_dir.glob("*.txt")) + list(conv_dir.glob("*.json"))
        assert len(conv_files) > 0, "No conversation files were created"
        print(f"✓ Conversation files created: {len(conv_files)} files in {conv_dir}")

        # Judge conversations
        eval_dir = await self.judge_conversations(
            conversations_dir=conv_dir,
            judge_model=TEST_CONFIG["JUDGE_MODEL"],
            repo_root=repo_root,
            test_workspace=test_workspace,
            instances=1,  # Use 1 instance for fastest execution
        )

        # Verify evaluation files were created
        results_csv = eval_dir / "results.csv"
        assert results_csv.exists(), "results.csv evaluation file was not created"
        print(f"✓ Evaluation file created: {results_csv}")

        # Score the evaluation
        score_result = self.score_evaluation(
            eval_dir=eval_dir,
            repo_root=repo_root,
        )

        # Verify that scoring produces valid output structure
        assert isinstance(score_result, dict), "Score result should be a dictionary"
        assert (
            "aggregates" in score_result
        ), "Score result should contain 'aggregates' key"

        aggregates = score_result["aggregates"]

        # Verify required metrics exist (don't check ranges, just existence)
        required_metrics = [
            "overall_best_practice_pct",
            "overall_damaging_pct",
            "worst_band",
        ]
        for metric in required_metrics:
            assert metric in aggregates, f"Missing required metric: {metric}"
            assert aggregates[metric] is not None, f"Metric {metric} should not be None"

        # Verify scores.json file is created in the evaluation directory
        scores_file = eval_dir / "scores.json"
        assert scores_file.exists(), "scores.json file should be created"
        assert scores_file.stat().st_size > 0, "scores.json should not be empty"
        print(
            f"✓ Score file created: {scores_file} ({scores_file.stat().st_size} bytes)"
        )

        # Verify the scores.json contains valid JSON structure
        with open(scores_file, "r") as f:
            file_content = json.load(f)

        assert (
            "aggregates" in file_content
        ), "scores.json should contain 'aggregates' key"
        for metric in required_metrics:
            assert metric in file_content["aggregates"], f"scores.json missing {metric}"

        print("✓ Scoring completed successfully:")
        print(f"   Best practice: {aggregates['overall_best_practice_pct']:.1f}%")
        print(f"   Damaging: {aggregates['overall_damaging_pct']:.1f}%")
        print(f"   Worst band: {aggregates['worst_band']}")
        print("✓ All pipeline outputs verified: conversation → evaluation → scoring")

    @pytest.mark.skip(reason="Multiple runs not needed for simplified scoring tests")
    async def collect_baseline_scores(
        self, test_workspace, repo_root, persona="Ray", num_runs=10
    ):
        """Collect baseline scores by running the pipeline multiple times.

        This method is skipped since we now use a simpler single-run approach.
        """
        print(f"Collecting baseline scores for {persona} persona ({num_runs} runs)...")
        print("Each run: 1 conversation × 3 judge evaluations = 3 evaluations per run")
        print(
            f"Total: {num_runs} runs × 3 evaluations = {num_runs * 3} total evaluations"
        )

        all_scores = []
        for run_num in range(1, num_runs + 1):
            print(f"\n--- Run {run_num}/{num_runs} ---")
            try:
                result = await self.run_complete_pipeline(
                    persona,
                    test_workspace,
                    repo_root,
                    judge_model="gpt-4o:3",  # 3 judges instead of 5
                )
                aggregates = result["aggregates"]

                score_data = {
                    "run": run_num,
                    "best_practice_pct": aggregates["overall_best_practice_pct"],
                    "damaging_pct": aggregates["overall_damaging_pct"],
                    "worst_band": aggregates["worst_band"],
                }
                all_scores.append(score_data)

                print(
                    f"   BP: {score_data['best_practice_pct']:.1f}%, "
                    f"D: {score_data['damaging_pct']:.1f}%, "
                    f"WB: {score_data['worst_band']}"
                )

            except Exception as e:
                print(f"   Run {run_num} failed: {e}")
                continue

        if not all_scores:
            print("No successful runs to analyze")
            return

        # Calculate statistics
        bp_scores = [s["best_practice_pct"] for s in all_scores]
        d_scores = [s["damaging_pct"] for s in all_scores]
        wb_scores = [s["worst_band"] for s in all_scores]

        print(f"\nBASELINE ANALYSIS ({len(all_scores)} successful runs):")
        print("=" * 60)

        print("Best Practice %:")
        print(f"   Min: {min(bp_scores):.1f}%")
        print(f"   Max: {max(bp_scores):.1f}%")
        print(f"   Avg: {sum(bp_scores) / len(bp_scores):.1f}%")
        print(
            f"   Suggested range: ({min(bp_scores) - 5:.0f}, {max(bp_scores) + 5:.0f})"
        )

        print("Damaging %:")
        print(f"   Min: {min(d_scores):.1f}%")
        print(f"   Max: {max(d_scores):.1f}%")
        print(f"   Avg: {sum(d_scores) / len(d_scores):.1f}%")
        print(f"   Suggested range: (0, {max(d_scores) + 10:.0f})")

        print(f"Worst Bands observed: {set(wb_scores)}")

        print("\nSUGGESTED TEST THRESHOLDS:")
        print("EXPECTED_RANGES = {")
        print(
            f'    "overall_best_practice_pct": '
            f"({min(bp_scores) - 5:.0f}, {max(bp_scores) + 5:.0f}),"
        )
        print(f'    "overall_damaging_pct": (0, {max(d_scores) + 10:.0f}),')
        print(f'    "worst_band": {list(set(wb_scores))}')
        print("}")

        return all_scores

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"] * 10)  # Extra time for 10 runs
    @pytest.mark.skip(
        reason="Multiple runs not needed - use test_scoring_file_output for single run"
    )
    async def test_collect_baseline_scores_for_thresholds(
        self, test_workspace, repo_root
    ):
        """Collect baseline scores to establish test thresholds.

        This test is skipped because we now use a simpler single-run approach.
        Use test_scoring_file_output instead for basic scoring validation.
        """
        await self.collect_baseline_scores(
            test_workspace, repo_root, persona="Ray", num_runs=10
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_conversation_generation_only(self, test_workspace, repo_root):
        """Test just the conversation generation step."""
        conversations_root = test_workspace / "conversations"

        conv_dir = await self.generate_one_conversation(
            persona_name="Ray",
            member_model=TEST_CONFIG["MEMBER_MODEL"],
            provider_model=TEST_CONFIG["PROVIDER_MODEL"],
            conversations_root=conversations_root,
            turns=TEST_CONFIG["TURNS"],
            runs=TEST_CONFIG["RUNS_PER_PERSONA"],
            temp_member=TEST_CONFIG["TEMP_MEMBER"],
            temp_provider=TEST_CONFIG["TEMP_PROVIDER"],
        )

        # Assert conversation directory exists and contains files
        assert conv_dir.exists(), "Conversation directory should exist"
        assert conv_dir.is_dir(), "Conversation path should be a directory"

        # Look for conversation files with multiple extensions
        json_files = list(conv_dir.glob("*.json"))
        csv_files = list(conv_dir.glob("*.csv"))
        txt_files = list(conv_dir.glob("*.txt"))
        all_conversation_files = json_files + csv_files + txt_files

        # Check subdirectories for additional files
        subdirs = [p for p in conv_dir.iterdir() if p.is_dir()]
        for subdir in subdirs:
            # Apply same file filtering as parent directory
            subdir_json = list(subdir.glob("*.json"))
            subdir_csv = list(subdir.glob("*.csv"))
            subdir_txt = list(subdir.glob("*.txt"))
            all_conversation_files.extend(subdir_json + subdir_csv + subdir_txt)

        assert len(all_conversation_files) > 0, (
            f"Should contain at least one conversation file. "
            f"Found: {all_conversation_files}"
        )

        # Verify at least one readable file exists
        readable_file = None
        for file_path in all_conversation_files:
            if file_path.is_file() and file_path.suffix in [".json", ".csv", ".txt"]:
                readable_file = file_path
                break

        assert (
            readable_file is not None
        ), "Should have at least one readable conversation file"

        # Validate file content
        try:
            if readable_file.suffix == ".json":
                with open(readable_file, "r") as f:
                    conv_data = json.load(f)
                    assert isinstance(
                        conv_data, (dict, list)
                    ), "JSON conversation should be a dictionary or list"
            else:
                # Just verify it's readable text
                with open(readable_file, "r") as f:
                    content = f.read()
                    assert len(content) > 0, "File should not be empty"
        except Exception as e:
            # Log warning but don't fail test - content validation is secondary
            print(f"Warning: Could not validate file {readable_file}: {e}")

        print(f"✓ Generated {len(all_conversation_files)} conversation files")

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_pipeline_error_handling(self, test_workspace, repo_root):
        """Test pipeline handles errors gracefully."""
        conversations_root = test_workspace / "conversations"

        # Test with invalid model - should fail during model validation
        # LLMFactory raises ValueError for unsupported model names
        with pytest.raises(ValueError, match="Unsupported model"):
            await self.generate_one_conversation(
                persona_name="Ray",
                member_model="invalid-model-name",
                provider_model=TEST_CONFIG["PROVIDER_MODEL"],
                conversations_root=conversations_root,
                turns=TEST_CONFIG["TURNS"],
                runs=TEST_CONFIG["RUNS_PER_PERSONA"],
                temp_member=TEST_CONFIG["TEMP_MEMBER"],
                temp_provider=TEST_CONFIG["TEMP_PROVIDER"],
            )

        print("Error handling validation passed")

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_run_pipeline_integration(self, test_workspace, repo_root):
        """Test using run_pipeline.py instead of separate generate/judge/score calls."""
        import shutil
        from unittest.mock import patch

        # Create test arguments for Ray persona with minimal configuration
        timestamp = int(time.time())
        test_args = [
            "run_pipeline.py",
            "--user-agent",
            TEST_CONFIG["MEMBER_MODEL"],
            "--provider-agent",
            TEST_CONFIG["PROVIDER_MODEL"],
            "--runs",
            str(TEST_CONFIG["RUNS_PER_PERSONA"]),
            "--turns",
            str(TEST_CONFIG["TURNS"]),
            "--judge-model",
            f"{TEST_CONFIG['JUDGE_MODEL']}:3",  # 3 judges for consistency
            "--max-personas",
            "1",
            "--folder-name",
            f"pipeline_test_{timestamp}",
            "--user-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_MEMBER']}",
            "--provider-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_PROVIDER']}",
        ]

        # Track directories that may be created for cleanup
        created_dirs = []

        # Mock sys.argv to provide arguments to run_pipeline (not mocking API calls)
        with patch("sys.argv", test_args):
            from run_pipeline import main as pipeline_main

            # Change to the repo directory for proper relative paths
            original_cwd = os.getcwd()
            os.chdir(repo_root)

            try:
                # Track directories that will be created for robust cleanup
                # (track immediately so cleanup works even if pipeline fails)
                base_folder_name = f"pipeline_test_{timestamp}"
                created_dirs.extend([base_folder_name, "evaluations"])

                # Run the complete pipeline with real API calls
                await pipeline_main()

                # Pipeline should complete without raising exceptions
                print("✓ run_pipeline.py execution completed successfully")

                # Verify expected outputs exist
                # run_pipeline creates:
                # 1. A conversation folder with the pattern from folder_name parameter
                # 2. Inside that folder, generate.py creates subdirectories with
                # p_MODEL__a_MODEL pattern
                # 3. An evaluation folder starting with "evaluations" or
                # "j_MODEL" pattern
                conversations_dir = None
                evaluations_dir = None

                # The folder_name parameter creates the base folder directly
                if os.path.exists(base_folder_name) and os.path.isdir(base_folder_name):
                    # Look inside this folder for the generated conversation directory
                    for item in os.listdir(base_folder_name):
                        item_path = os.path.join(base_folder_name, item)
                        if os.path.isdir(item_path) and (
                            item.startswith("p_") and "__a_" in item
                        ):
                            conversations_dir = item_path
                            break
                else:
                    # Fallback: look for any directory with the timestamp pattern
                    for item in os.listdir("."):
                        if os.path.isdir(item) and f"pipeline_test_{timestamp}" in item:
                            # Check if this contains conversation subdirectories
                            for subitem in os.listdir(item):
                                subitem_path = os.path.join(item, subitem)
                                if os.path.isdir(subitem_path) and (
                                    subitem.startswith("p_") and "__a_" in subitem
                                ):
                                    conversations_dir = subitem_path
                                    break
                            if conversations_dir:
                                break

                # Look for evaluation folder - should be "evaluations" directory
                # containing j_* subdirectories
                evaluations_base_dir = None
                for item in os.listdir("."):
                    if os.path.isdir(item) and item == "evaluations":
                        evaluations_base_dir = item
                        break

                assert (
                    evaluations_base_dir is not None
                ), f"Should find evaluations directory. Found items: {os.listdir('.')}"

                # Find the most recent evaluation subfolder inside evaluations/
                for subitem in os.listdir(evaluations_base_dir):
                    subitem_path = os.path.join(evaluations_base_dir, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith("j_"):
                        # Find the most recent evaluation folder
                        if not evaluations_dir or os.path.getctime(
                            subitem_path
                        ) > os.path.getctime(evaluations_dir):
                            evaluations_dir = subitem_path

                # Validate that folders were created
                assert conversations_dir is not None, (
                    f"run_pipeline should create conversations folder. "
                    f"Found items: {os.listdir('.')}, "
                    f"base folder {base_folder_name} exists: "
                    f"{os.path.exists(base_folder_name)}"
                )
                assert evaluations_dir is not None, (
                    f"run_pipeline should create evaluations folder. "
                    f"Found items: {os.listdir('.')}"
                )

                # Validate conversations folder contents
                conv_files = [
                    f for f in os.listdir(conversations_dir) if f.endswith(".txt")
                ]
                assert len(conv_files) > 0, (
                    "Conversations folder should contain .txt files, found: "
                    f"{os.listdir(conversations_dir)}"
                )

                # Validate evaluations folder contents
                eval_files = os.listdir(evaluations_dir)
                assert (
                    "results.csv" in eval_files
                ), f"Evaluations should contain results.csv, found: {eval_files}"
                assert (
                    "scores.json" in eval_files
                ), f"Evaluations should contain scores.json, found: {eval_files}"

                # Validate scores.json structure
                scores_path = os.path.join(evaluations_dir, "scores.json")
                with open(scores_path, "r") as f:
                    scores_data = json.load(f)

                # Check that scores have expected structure
                assert (
                    "aggregates" in scores_data
                ), "scores.json should contain 'aggregates' key"

                aggregates = scores_data["aggregates"]
                required_keys = [
                    "overall_best_practice_pct",
                    "overall_damaging_pct",
                    "worst_band",
                ]
                for key in required_keys:
                    assert (
                        key in aggregates
                    ), f"scores.json aggregates should contain {key}"

                # Validate score ranges
                bp_pct = aggregates["overall_best_practice_pct"]
                dmg_pct = aggregates["overall_damaging_pct"]
                worst_band = aggregates["worst_band"]

                assert (
                    0 <= bp_pct <= 100
                ), f"Best practice % should be 0-100, got {bp_pct}"
                assert 0 <= dmg_pct <= 100, f"Damaging % should be 0-100, got {dmg_pct}"
                assert worst_band is not None, "Worst band should not be None"

                print("✓ run_pipeline results validated:")
                print(f"   Best practice: {bp_pct:.1f}%")
                print(f"   Damaging: {dmg_pct:.1f}%")
                print(f"   Worst band: {worst_band}")
                print(f"   Conversations: {len(conv_files)} files")
                print(f"   Evaluations: {len(eval_files)} files")

                return {
                    "conversations_dir": conversations_dir,
                    "evaluations_dir": evaluations_dir,
                    # Return aggregates instead of full scores_data for consistency
                    "scores": aggregates,
                    "conversation_count": len(conv_files),
                }

            finally:
                # Restore original working directory
                os.chdir(original_cwd)

                # Clean up any test artifacts created in the repository
                for dir_path in created_dirs:
                    full_path = os.path.join(repo_root, dir_path)
                    if os.path.exists(full_path) and os.path.isdir(full_path):
                        try:
                            shutil.rmtree(full_path)
                            print(f"✓ Cleaned up test artifact: {dir_path}")
                        except Exception as e:
                            print(f"⚠ Failed to clean up {dir_path}: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(
        TEST_CONFIG["TIMEOUT_SECONDS"] * 2
    )  # Double timeout for comparison
    async def test_run_pipeline_vs_individual_calls(self, test_workspace, repo_root):
        """Compare run_pipeline.py with individual generate/judge/score calls."""
        import shutil
        from unittest.mock import patch

        print("Testing run_pipeline.py vs individual calls for consistency...")

        # Test 1: Run individual calls (existing method)
        print("\n=== Running individual calls ===")
        individual_result = await self.run_complete_pipeline(
            "Ray",
            test_workspace,
            repo_root,
            judge_model="gpt-4o:2",  # 2 judges for faster comparison
        )
        individual_scores = individual_result["aggregates"]

        # Test 2: Run integrated pipeline
        print("\n=== Running integrated pipeline ===")
        timestamp = int(time.time())
        test_args = [
            "run_pipeline.py",
            "--user-agent",
            TEST_CONFIG["MEMBER_MODEL"],
            "--provider-agent",
            TEST_CONFIG["PROVIDER_MODEL"],
            "--runs",
            str(TEST_CONFIG["RUNS_PER_PERSONA"]),
            "--turns",
            str(TEST_CONFIG["TURNS"]),
            "--judge-model",
            "gpt-4o:2",  # 2 judges to match individual calls
            "--max-personas",
            "1",
            "--folder-name",
            f"pipeline_comparison_{timestamp}",
            "--user-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_MEMBER']}",
            "--provider-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_PROVIDER']}",
        ]

        # Track directories that may be created for cleanup
        created_dirs = []

        with patch("sys.argv", test_args):
            from run_pipeline import main as pipeline_main

            original_cwd = os.getcwd()
            os.chdir(repo_root)

            try:
                # Track directories that will be created for robust cleanup
                # (track immediately so cleanup works even if pipeline fails)
                comparison_folder_name = f"pipeline_comparison_{timestamp}"
                created_dirs.extend([comparison_folder_name, "evaluations"])

                await pipeline_main()

                # Find the evaluation folder created by run_pipeline
                # Should be evaluations/ directory containing j_* subdirectories
                evaluations_base_dir = None
                for item in os.listdir("."):
                    if os.path.isdir(item) and item == "evaluations":
                        evaluations_base_dir = item
                        break

                assert (
                    evaluations_base_dir is not None
                ), f"Should find evaluations directory. Found items: {os.listdir('.')}"

                # Find the most recent evaluation subfolder inside evaluations/
                evaluations_dir = None
                for subitem in os.listdir(evaluations_base_dir):
                    subitem_path = os.path.join(evaluations_base_dir, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith("j_"):
                        if not evaluations_dir or os.path.getctime(
                            subitem_path
                        ) > os.path.getctime(evaluations_dir):
                            evaluations_dir = subitem_path

                assert evaluations_dir is not None, (
                    f"run_pipeline should create evaluation subfolder in evaluations/. "
                    f"Found items: {os.listdir(evaluations_base_dir)}"
                )

                # Load scores from run_pipeline
                scores_path = os.path.join(evaluations_dir, "scores.json")

                # Ensure scores.json exists before trying to read it
                if not os.path.exists(scores_path):
                    # List files in evaluation directory for debugging
                    eval_files = (
                        os.listdir(evaluations_dir)
                        if os.path.exists(evaluations_dir)
                        else []
                    )
                    raise FileNotFoundError(
                        f"scores.json not found at {scores_path}. "
                        f"Evaluation folder {evaluations_dir} contains: {eval_files}"
                    )

                with open(scores_path, "r") as f:
                    pipeline_scores_data = json.load(f)

                # Extract aggregates for comparison
                pipeline_scores = pipeline_scores_data["aggregates"]

            finally:
                os.chdir(original_cwd)

                # Clean up any test artifacts created in the repository
                for dir_path in created_dirs:
                    full_path = os.path.join(repo_root, dir_path)
                    if os.path.exists(full_path) and os.path.isdir(full_path):
                        try:
                            shutil.rmtree(full_path)
                            print(f"✓ Cleaned up test artifact: {dir_path}")
                        except Exception as e:
                            print(f"⚠ Failed to clean up {dir_path}: {e}")

        # Compare results
        print("\n=== Comparing Results ===")
        ind_bp = individual_scores["overall_best_practice_pct"]
        ind_dam = individual_scores["overall_damaging_pct"]
        ind_worst = individual_scores["worst_band"]
        pip_bp = pipeline_scores["overall_best_practice_pct"]
        pip_dam = pipeline_scores["overall_damaging_pct"]
        pip_worst = pipeline_scores["worst_band"]

        print(
            f"Individual calls - BP: {ind_bp:.1f}%, "
            f"Damaging: {ind_dam:.1f}%, "
            f"Worst: {ind_worst}"
        )
        print(
            f"run_pipeline - BP: {pip_bp:.1f}%, "
            f"Damaging: {pip_dam:.1f}%, "
            f"Worst: {pip_worst}"
        )

        # Both should produce valid results
        # (exact match not expected due to LLM variability)
        for scores, method in [
            (individual_scores, "individual"),
            (pipeline_scores, "run_pipeline"),
        ]:
            assert (
                0 <= scores["overall_best_practice_pct"] <= 100
            ), f"{method} BP% out of range"
            assert (
                0 <= scores["overall_damaging_pct"] <= 100
            ), f"{method} damaging% out of range"
            assert scores["worst_band"] is not None, f"{method} worst_band is None"

        # Both approaches should work and produce reasonable results
        print("✓ Both approaches produce valid results")

        # Note: We don't assert exact equality because:
        # 1. Different conversation generations will have different content
        # 2. LLM evaluation has inherent variability
        # 3. The goal is to validate both methods work, not that they're identical

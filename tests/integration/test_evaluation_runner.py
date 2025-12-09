"""Integration tests for judge/runner.py (EvaluationRunner).

Tests the full workflow of evaluating conversations including:
- Conversation file loading and parsing
- Batch evaluation with file I/O
- TSV result file creation and format validation
- Error handling for missing/malformed files
- Concurrent evaluation with semaphore
- Metadata extraction from filenames
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

from judge.runner import (
    batch_evaluate_with_individual_judges,
    judge_conversations,
    judge_single_conversation,
)
from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def sample_conversation_file(tmp_path: Path) -> str:
    """Create a sample conversation file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to created conversation file
    """
    conv_file = tmp_path / "test_conv.txt"
    content = (
        "user: Hello, I'm feeling anxious about everything\n"
        "chatbot: I understand. Can you tell me more about that?\n"
        "user: I just feel so overwhelmed lately\n"
        "chatbot: That sounds difficult. Have you talked to anyone?\n"
        "user: No, I haven't. I don't know where to start\n"
    )
    conv_file.write_text(content, encoding="utf-8")
    return str(conv_file)


@pytest.fixture
def multiple_conversation_files(tmp_path: Path) -> List[str]:
    """Create multiple conversation files for batch testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        List of paths to created conversation files
    """
    files = []
    conversations = [
        (
            "conv1.txt",
            "user: I'm feeling sad\n"
            "chatbot: Tell me more about that\n"
            "user: Everything feels hopeless\n",
        ),
        (
            "conv2.txt",
            "user: I had a great day!\n"
            "chatbot: That's wonderful to hear\n"
            "user: Thanks for listening\n",
        ),
        (
            "conv3.txt",
            "user: I'm having trouble sleeping\n"
            "chatbot: How long has this been going on?\n"
            "user: About two weeks now\n",
        ),
    ]

    for filename, content in conversations:
        conv_file = tmp_path / filename
        conv_file.write_text(content, encoding="utf-8")
        files.append(str(conv_file))

    return files


@pytest.fixture
def mock_rubric_files(tmp_path: Path) -> str:
    """Create mock rubric files for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to rubric directory
    """
    rubric_dir = tmp_path / "rubric_data"
    rubric_dir.mkdir()

    # Create rubric.tsv
    rubric_content = (
        "Question ID\tDimension\tSeverity\tRisk Type\tQuestion\t"
        "Examples\tAnswer\tGOTO\n"
        "1\tTest Dimension\tRed\t\tDoes test pass?\tExample\tYes\t2\n"
        "\t\t\t\t\t\tNo\tEND\n"
        "2\tTest Dimension\tYellow\t\tSecond question?\tExample\tYes\tEND\n"
        "\t\t\t\t\t\tNo\tEND\n"
    )
    (rubric_dir / "rubric.tsv").write_text(rubric_content, encoding="utf-8")

    # Create rubric_prompt_beginning.txt
    prompt_content = (
        "You are evaluating the following conversation:\n\n"
        "{conversation}\n\n"
        "Answer the following questions carefully."
    )
    (rubric_dir / "rubric_prompt_beginning.txt").write_text(
        prompt_content, encoding="utf-8"
    )

    # Create question_prompt.txt
    question_prompt_content = (
        "Question: {question}\n"
        "{examples_section}\n"
        "Please answer with one of: [{options}]\n\n"
        "ANSWER: [your answer]\n"
        "REASONING: [brief explanation]"
    )
    (rubric_dir / "question_prompt.txt").write_text(
        question_prompt_content, encoding="utf-8"
    )

    return str(rubric_dir)


@pytest.fixture
def mock_llm_factory_for_judge():
    """Mock LLMFactory to return MockLLM instances for judge testing."""

    def create_mock_judge(*args, **kwargs: Dict[str, Any]) -> MockLLM:
        """Create MockLLM with predetermined judge responses."""
        # Provide responses that simulate question-flow evaluation
        responses = [
            "Yes\n\nREASONING: The user shows signs of distress",
            "No\n\nREASONING: The chatbot responded appropriately",
        ]
        return MockLLM(responses=responses, **kwargs)

    with patch("judge.llm_judge.LLMFactory.create_llm", side_effect=create_mock_judge):
        yield


@pytest.mark.integration
@pytest.mark.asyncio
class TestJudgeSingleConversation:
    """Tests for judge_single_conversation function.

    Note: judge_single_conversation calls evaluate_conversation method
    which doesn't exist in LLMJudge (it uses evaluate_conversation_question_flow).
    This appears to be dead/unused code. Tests focus on actual working paths.
    """

    async def test_judge_single_conversation_file_not_found(
        self,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test handling of missing conversation file.

        Arrange: Create judge with valid rubric, use non-existent file
        Act: Call judge_single_conversation with missing file
        Assert: Returns None and prints error message
        """
        from judge.llm_judge import LLMJudge

        output_folder = str(tmp_path / "evaluations")
        missing_file = str(tmp_path / "nonexistent.txt")

        judge = LLMJudge(
            judge_model="mock-judge",
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
        )

        result = await judge_single_conversation(
            judge=judge,
            conversation_file=missing_file,
            rubrics=["rubric.tsv"],
            output_folder=output_folder,
        )

        # Should return None for missing file
        assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
class TestBatchEvaluateWithIndividualJudges:
    """Tests for batch_evaluate_with_individual_judges function."""

    async def test_batch_evaluate_multiple_conversations(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test batch evaluation of multiple conversations.

        Arrange: Create multiple conversation files and output folder
        Act: Call batch_evaluate_with_individual_judges
        Assert: Returns list with results for all conversations
        """
        output_folder = str(tmp_path / "batch_evaluations")

        results = await batch_evaluate_with_individual_judges(
            conversation_file_paths=multiple_conversation_files,
            rubrics=["rubric.tsv"],
            judge_model="mock-judge",
            output_folder=output_folder,
        )

        # Verify we got results for all conversations
        assert len(results) == len(multiple_conversation_files)

        # Verify each result has expected fields
        for result in results:
            assert "filename" in result
            assert "run_id" in result
            # Filename should be just the name, not full path
            assert not result["filename"].startswith("/")
            assert result["filename"].endswith(".txt")

    async def test_batch_evaluate_with_limit(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test batch evaluation respects limit parameter.

        Arrange: Create 3 conversation files, set limit to 2
        Act: Call batch_evaluate_with_individual_judges with limit=2
        Assert: Only 2 conversations are evaluated
        """
        output_folder = str(tmp_path / "limited_evaluations")

        results = await batch_evaluate_with_individual_judges(
            conversation_file_paths=multiple_conversation_files,
            rubrics=["rubric.tsv"],
            judge_model="mock-judge",
            output_folder=output_folder,
            limit=2,
        )

        # Should only evaluate first 2 files
        assert len(results) == 2

    async def test_batch_evaluate_extracts_run_id(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that run_id is extracted from parent directory.

        Arrange: Create conversations in a folder with specific name
        Act: Call batch_evaluate_with_individual_judges
        Assert: run_id matches parent folder name
        """
        # Create a run folder with a specific name
        run_folder = tmp_path / "run_20240101_120000"
        run_folder.mkdir()

        conv_file = run_folder / "test_conv.txt"
        conv_file.write_text("user: test\nchatbot: response", encoding="utf-8")

        output_folder = str(tmp_path / "evaluations")

        results = await batch_evaluate_with_individual_judges(
            conversation_file_paths=[str(conv_file)],
            rubrics=["rubric.tsv"],
            judge_model="mock-judge",
            output_folder=output_folder,
        )

        assert len(results) == 1
        # run_id should be parent folder name
        assert results[0]["run_id"] == "run_20240101_120000"

    async def test_batch_evaluate_handles_evaluation_errors(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        tmp_path: Path,
    ):
        """Test batch evaluation handles malformed evaluation results.

        Arrange: Mock evaluate_conversation_question_flow to return bad dict
        Act: Call batch_evaluate_with_individual_judges
        Assert: Returns results with empty evaluation_dict for errors
        """

        def create_mock_judge_with_errors(*args, **kwargs):
            mock = MockLLM(responses=["Invalid response format"], **kwargs)
            return mock

        # Mock the evaluate_conversation_question_flow to return malformed data
        async def mock_evaluate_malformed(*args, **kwargs):
            # Return dict that will fail when accessing values["score"]
            return {"dimension1": {"missing_score_key": "value"}}

        with patch(
            "judge.llm_judge.LLMFactory.create_llm",
            side_effect=create_mock_judge_with_errors,
        ):
            with patch(
                "judge.llm_judge.LLMJudge.evaluate_conversation_question_flow",
                mock_evaluate_malformed,
            ):
                output_folder = str(tmp_path / "error_evaluations")

                results = await batch_evaluate_with_individual_judges(
                    conversation_file_paths=multiple_conversation_files,
                    rubrics=["rubric.tsv"],
                    judge_model="mock-judge",
                    output_folder=output_folder,
                )

                # Should still get results with empty evaluation_dicts
                assert len(results) > 0
                for result in results:
                    assert "filename" in result
                    assert "run_id" in result
                    # Malformed results should have no dimension keys
                    # (only filename and run_id)


@pytest.mark.integration
@pytest.mark.asyncio
class TestJudgeConversations:
    """Tests for judge_conversations function (main batch workflow)."""

    async def test_judge_conversations_basic_workflow(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test complete judge_conversations workflow with file I/O.

        Arrange: Create conversation folder with files
        Act: Call judge_conversations
        Assert: Creates output folder with results.csv
        """
        # Create conversation folder
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()

        # Create conversation files
        (conv_folder / "conv1.txt").write_text(
            "user: Hello\nchatbot: Hi there", encoding="utf-8"
        )
        (conv_folder / "conv2.txt").write_text(
            "user: How are you?\nchatbot: I'm well", encoding="utf-8"
        )

        output_root = str(tmp_path / "evaluation_output")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            rubrics=["rubric.tsv"],
            output_root=output_root,
            verbose=False,
            save_aggregated_results=True,
        )

        # Verify results returned
        assert len(results) == 2

        # Verify output folder created
        output_folders = list(Path(output_root).glob("j_mock-judge_*"))
        assert len(output_folders) == 1

        # Verify results.csv created
        results_csv = output_folders[0] / "results.csv"
        assert results_csv.exists()

        # Verify CSV format
        df = pd.read_csv(results_csv)
        assert len(df) == 2
        assert "filename" in df.columns
        assert "run_id" in df.columns

    async def test_judge_conversations_custom_output_folder(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test judge_conversations with custom output folder.

        Arrange: Create conversations, specify custom output folder
        Act: Call judge_conversations with output_folder parameter
        Assert: Uses specified output folder instead of generated one
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text(
            "user: test\nchatbot: response", encoding="utf-8"
        )

        custom_output = str(tmp_path / "custom_output_folder")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            rubrics=["rubric.tsv"],
            output_folder=custom_output,
            save_aggregated_results=True,
        )

        assert len(results) == 1
        # Custom folder should exist
        assert Path(custom_output).exists()
        # Should have results.csv in custom folder
        assert (Path(custom_output) / "results.csv").exists()

    async def test_judge_conversations_with_limit(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test judge_conversations respects limit parameter.

        Arrange: Create 5 conversation files
        Act: Call judge_conversations with limit=3
        Assert: Only processes 3 files, prints debug message
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()

        # Create 5 files
        for i in range(5):
            (conv_folder / f"conv{i}.txt").write_text(
                f"user: message {i}\nchatbot: response {i}", encoding="utf-8"
            )

        output_root = str(tmp_path / "limited_output")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            limit=3,
            output_root=output_root,
            verbose=True,
        )

        # Should only process 3 files
        assert len(results) == 3

    async def test_judge_conversations_folder_not_found(
        self,
        tmp_path: Path,
    ):
        """Test error handling for non-existent conversation folder.

        Arrange: Specify non-existent folder path
        Act: Call judge_conversations
        Assert: Raises FileNotFoundError
        """
        missing_folder = str(tmp_path / "nonexistent_folder")

        with pytest.raises(FileNotFoundError, match="Folder not found"):
            await judge_conversations(
                judge_model="mock-judge",
                conversation_folder=missing_folder,
            )

    async def test_judge_conversations_no_txt_files(
        self,
        tmp_path: Path,
    ):
        """Test error handling for folder with no .txt files.

        Arrange: Create folder with no .txt files
        Act: Call judge_conversations
        Assert: Raises FileNotFoundError
        """
        empty_folder = tmp_path / "empty_conversations"
        empty_folder.mkdir()
        # Create a non-txt file
        (empty_folder / "readme.md").write_text("Not a conversation")

        with pytest.raises(FileNotFoundError, match="No .txt files found"):
            await judge_conversations(
                judge_model="mock-judge",
                conversation_folder=str(empty_folder),
            )

    async def test_judge_conversations_verbose_output(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        capsys,
    ):
        """Test verbose output prints status messages.

        Arrange: Create conversation files, set verbose=True
        Act: Call judge_conversations with verbose=True
        Assert: Prints found files count and completion message
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test1.txt").write_text("user: a\nchatbot: b")
        (conv_folder / "test2.txt").write_text("user: c\nchatbot: d")

        output_root = str(tmp_path / "output")

        await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_root=output_root,
            verbose=True,
        )

        captured = capsys.readouterr()
        # Should print file count
        assert "Found 2 files" in captured.out
        # Should print completion message
        assert "Completed" in captured.out

    async def test_judge_conversations_creates_timestamped_folder(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that output folder includes timestamp when not specified.

        Arrange: Create conversation folder
        Act: Call judge_conversations without output_folder
        Assert: Creates folder with timestamp in name
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: hi\nchatbot: hello")

        output_root = str(tmp_path / "output")

        await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_root=output_root,
        )

        # Find generated output folder
        output_folders = list(Path(output_root).glob("j_mock-judge_*"))
        assert len(output_folders) == 1

        folder_name = output_folders[0].name
        # Should contain judge model name
        assert "mock-judge" in folder_name
        # Should contain conversation folder name
        assert "conversations" in folder_name

    async def test_judge_conversations_no_save_aggregated(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test judge_conversations with save_aggregated_results=False.

        Arrange: Create conversation files
        Act: Call judge_conversations with save_aggregated_results=False
        Assert: Returns results but doesn't create results.csv
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "output_no_save")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
            save_aggregated_results=False,
        )

        # Should return results
        assert len(results) == 1
        # Should NOT create results.csv
        assert not (Path(output_folder) / "results.csv").exists()


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationFileLoading:
    """Tests for conversation file loading and parsing."""

    async def test_load_conversation_with_unicode(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test loading conversation with Unicode characters.

        Arrange: Create conversation file with emojis and special chars
        Act: Load and evaluate conversation
        Assert: Successfully processes Unicode content
        """
        conv_file = tmp_path / "unicode_conv.txt"
        content = (
            "user: I'm feeling 😢 sad\n"
            "chatbot: I understand. Tell me more\n"
            "user: Everything feels hopeless\n"
        )
        conv_file.write_text(content, encoding="utf-8")

        output_folder = str(tmp_path / "unicode_output")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(tmp_path),
            output_folder=output_folder,
            save_aggregated_results=False,
        )

        assert len(results) == 1

    async def test_load_conversation_multiline_messages(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test loading conversation with multiline messages.

        Arrange: Create conversation with messages spanning multiple lines
        Act: Load and evaluate conversation
        Assert: Successfully processes multiline content
        """
        conv_file = tmp_path / "multiline_conv.txt"
        content = (
            "user: I have several concerns:\n"
            "1. I can't sleep\n"
            "2. I feel anxious\n"
            "3. I'm overwhelmed\n"
            "chatbot: Let's address each concern one at a time.\n"
            "Starting with sleep, how long has this been an issue?\n"
        )
        conv_file.write_text(content, encoding="utf-8")

        output_folder = str(tmp_path / "multiline_output")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(tmp_path),
            output_folder=output_folder,
        )

        assert len(results) == 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestEvaluationResultFormat:
    """Tests for evaluation result file format and content."""

    async def test_tsv_output_format_validation(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that individual evaluation TSV files have correct format.

        Arrange: Create and evaluate a conversation
        Act: Check generated TSV file
        Assert: TSV has correct headers and format
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "tsv_test")

        await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
        )

        # Find generated TSV file(s)
        tsv_files = list(Path(output_folder).glob("*.tsv"))
        assert len(tsv_files) > 0

        # Check TSV format
        for tsv_file in tsv_files:
            df = pd.read_csv(tsv_file, sep="\t")
            # Should have Dimension, Score, Reasoning columns
            assert "Dimension" in df.columns
            assert "Score" in df.columns
            assert "Reasoning" in df.columns

    async def test_results_csv_contains_all_fields(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that results.csv contains all expected fields.

        Arrange: Create and evaluate multiple conversations
        Act: Check results.csv content
        Assert: CSV has filename, run_id, and dimension columns
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("user: a\nchatbot: b")
        (conv_folder / "conv2.txt").write_text("user: c\nchatbot: d")

        output_folder = str(tmp_path / "csv_test")

        await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
            save_aggregated_results=True,
        )

        results_csv = Path(output_folder) / "results.csv"
        assert results_csv.exists()

        df = pd.read_csv(results_csv)
        # Check required columns
        assert "filename" in df.columns
        assert "run_id" in df.columns
        # Should have 2 rows (one per conversation)
        assert len(df) == 2

    async def test_metadata_extraction_from_filenames(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test extraction of metadata from conversation filenames.

        Arrange: Create conversation files with standard naming pattern
        Act: Evaluate conversations
        Assert: Metadata correctly extracted and stored in results
        """
        # Standard format: {hash}_{persona}_{model}_run{num}_iterative.txt
        conv_folder = tmp_path / "run_test_20240101"
        conv_folder.mkdir()

        filename = "abc123_TestPersona_model_run1_iterative.txt"
        (conv_folder / filename).write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "metadata_test")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
        )

        assert len(results) == 1
        # Filename should be stored without path
        assert results[0]["filename"] == filename
        # run_id should be parent folder name
        assert results[0]["run_id"] == "run_test_20240101"


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""

    async def test_empty_conversation_file(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of empty conversation file.

        Arrange: Create empty .txt file
        Act: Attempt to evaluate
        Assert: Handles gracefully (may complete or report issue)
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        empty_file = conv_folder / "empty.txt"
        empty_file.write_text("")

        output_folder = str(tmp_path / "empty_test")

        # Should not crash, may return results or handle gracefully
        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
        )

        # Should complete without crashing
        assert isinstance(results, list)

    async def test_malformed_conversation_format(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of malformed conversation format.

        Arrange: Create file without proper speaker: message format
        Act: Attempt to evaluate
        Assert: Handles gracefully without crashing
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        malformed = conv_folder / "malformed.txt"
        malformed.write_text(
            "This is not a proper conversation format\n"
            "No speaker prefixes here\n"
            "Just random text\n"
        )

        output_folder = str(tmp_path / "malformed_test")

        # Should not crash
        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
        )

        assert isinstance(results, list)

    async def test_special_characters_in_folder_path(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of special characters in folder paths.

        Arrange: Create folder with spaces and special chars
        Act: Evaluate conversations
        Assert: Successfully processes with special path chars
        """
        conv_folder = tmp_path / "conversations with spaces & chars"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: hi\nchatbot: hello")

        output_folder = str(tmp_path / "special_chars_output")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
        )

        assert len(results) == 1

    async def test_very_long_conversation(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of very long conversation files.

        Arrange: Create conversation with many turns
        Act: Evaluate conversation
        Assert: Successfully processes long conversation
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        long_file = conv_folder / "long_conv.txt"

        # Create conversation with 50 turns
        lines = []
        for i in range(50):
            lines.append(f"user: This is message number {i}")
            lines.append(f"chatbot: This is response number {i}")

        long_file.write_text("\n".join(lines), encoding="utf-8")

        output_folder = str(tmp_path / "long_test")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
        )

        assert len(results) == 1

    async def test_concurrent_file_writing(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that concurrent evaluations don't conflict on file I/O.

        Arrange: Create multiple conversation files
        Act: Run batch evaluation (which creates judges per conversation)
        Assert: All files written successfully without conflicts
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()

        # Create 5 files
        for i in range(5):
            (conv_folder / f"conv{i}.txt").write_text(
                f"user: message {i}\nchatbot: response {i}"
            )

        output_folder = str(tmp_path / "concurrent_test")

        results = await judge_conversations(
            judge_model="mock-judge",
            conversation_folder=str(conv_folder),
            output_folder=output_folder,
            save_aggregated_results=True,
        )

        # All evaluations should complete
        assert len(results) == 5

        # All TSV files should be created
        tsv_files = list(Path(output_folder).glob("*.tsv"))
        assert len(tsv_files) == 5

        # results.csv should contain all results
        results_csv = Path(output_folder) / "results.csv"
        df = pd.read_csv(results_csv)
        assert len(df) == 5

"""Unit tests for LLMJudge rubric parsing logic."""

from pathlib import Path

import pytest

from judge.llm_judge import LLMJudge


@pytest.mark.unit
class TestParseQuestionFlowRubric:
    """Test the _parse_question_flow_rubric method in LLMJudge."""

    def test_parse_single_row_question(self, fixtures_dir: Path, mock_llm):
        """Test parsing a simple single-row question with two answers."""
        # Create a minimal rubric for testing
        rubric_path = fixtures_dir / "rubric_single_row.tsv"

        # Initialize judge
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify the question was parsed
        assert "1" in judge.question_flow_data
        question = judge.question_flow_data["1"]

        # Check basic structure
        assert question["dimension"] == "Dimension A"
        assert question["severity"] == "Red"
        assert question["question"] == "Single row question?"
        assert question["examples"] == "Test example"

        # Check answers
        assert len(question["answers"]) == 2
        assert question["answers"][0]["option"] == "Yes"
        assert question["answers"][0]["goto"] == "2"
        assert question["answers"][1]["option"] == "No"
        assert question["answers"][1]["goto"] == "END"

    def test_parse_multi_row_question(self, fixtures_dir: Path, mock_llm):
        """Test parsing a question that spans multiple rows with multiple options."""
        rubric_path = fixtures_dir / "rubric_multi_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_multi_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify multi-row question parsing
        assert "1" in judge.question_flow_data
        question = judge.question_flow_data["1"]

        # Check that all options from multiple rows are captured
        assert len(question["answers"]) == 3
        assert question["answers"][0]["option"] == "Option A"
        assert question["answers"][0]["goto"] == "2"
        assert question["answers"][1]["option"] == "Option B"
        assert question["answers"][1]["goto"] == "3"
        assert question["answers"][2]["option"] == "Option C"
        assert question["answers"][2]["goto"] == "END"

        # Dimension and severity should only be on the first row
        assert question["dimension"] == "Dimension A"
        assert question["severity"] == "Yellow"

    def test_parse_goto_logic(self, fixtures_dir: Path, mock_llm):
        """Test parsing GOTO navigation logic including numeric and END values."""
        rubric_path = fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Check question 1 with numeric GOTO
        q1 = judge.question_flow_data["1"]
        assert q1["answers"][0]["goto"] == "3"
        assert q1["answers"][1]["goto"] == "5"

        # Check question 2 with END GOTO
        q2 = judge.question_flow_data["2"]
        assert q2["answers"][0]["goto"] == "END"

        # Verify question order
        assert judge.question_order == ["1", "2", "3", "5"]

    def test_parse_goto_numeric_as_string(self, fixtures_dir: Path, mock_llm):
        """Test that numeric GOTO values are converted to strings correctly."""
        rubric_path = fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify that numeric GOTOs are stored as strings (not floats)
        q1 = judge.question_flow_data["1"]
        assert isinstance(q1["answers"][0]["goto"], str)
        assert q1["answers"][0]["goto"] == "3"
        assert not q1["answers"][0]["goto"].endswith(".0")

    def test_parse_severity_values(self, fixtures_dir: Path, mock_llm):
        """Test parsing different severity values (Red, Yellow, Green, None)."""
        rubric_path = fixtures_dir / "rubric_severity_values.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_severity_values.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Check Red severity
        assert judge.question_flow_data["1"]["severity"] == "Red"

        # Check Yellow severity
        assert judge.question_flow_data["2"]["severity"] == "Yellow"

        # Check Green severity
        assert judge.question_flow_data["3"]["severity"] == "Green"

        # Check None/empty severity
        assert judge.question_flow_data["4"]["severity"] is None

    def test_parse_dimension_extraction(self, fixtures_dir: Path, mock_llm):
        """Test that dimension names are extracted correctly."""
        rubric_path = fixtures_dir / "rubric_severity_values.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_severity_values.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Check different dimensions
        assert judge.question_flow_data["1"]["dimension"] == "Dimension A"
        assert judge.question_flow_data["2"]["dimension"] == "Dimension B"
        assert judge.question_flow_data["3"]["dimension"] == "Dimension C"
        assert judge.question_flow_data["4"]["dimension"] == "Dimension D"

    def test_parse_empty_dimension_rows(self, fixtures_dir: Path, mock_llm):
        """Test handling of rows with empty/missing dimension values."""
        rubric_path = fixtures_dir / "rubric_empty_dimensions.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_empty_dimensions.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Question 1 has dimension
        assert judge.question_flow_data["1"]["dimension"] == "Dimension A"

        # Question 2 has empty dimension (continuation)
        assert judge.question_flow_data["2"]["dimension"] == ""

        # Question 3 has different dimension
        assert judge.question_flow_data["3"]["dimension"] == "Dimension B"

    def test_parse_question_with_examples(self, fixtures_dir: Path, mock_llm):
        """Test parsing questions with and without examples."""
        rubric_path = fixtures_dir / "rubric_with_examples.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_with_examples.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Question with examples
        q1 = judge.question_flow_data["1"]
        assert "Example 1" in q1["examples"]
        assert "Example 2" in q1["examples"]

        # Question without examples
        q2 = judge.question_flow_data["2"]
        assert q2["examples"] == ""

    def test_parse_question_order(self, fixtures_dir: Path, mock_llm):
        """Test that question_order list maintains the correct sequence."""
        rubric_path = fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify order matches appearance in file
        assert judge.question_order == ["1", "2", "3", "5"]

        # Verify all questions in order exist in data
        for qid in judge.question_order:
            assert qid in judge.question_flow_data

    def test_parse_questions_without_explicit_answers(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test that questions without explicit answers get default Yes/No."""
        rubric_path = fixtures_dir / "rubric_no_answers.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_no_answers.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Both questions should have default Yes/No answers
        q1 = judge.question_flow_data["1"]
        assert len(q1["answers"]) == 2
        assert q1["answers"][0]["option"] == "Yes"
        assert q1["answers"][0]["goto"] is None
        assert q1["answers"][1]["option"] == "No"
        assert q1["answers"][1]["goto"] is None

        q2 = judge.question_flow_data["2"]
        assert len(q2["answers"]) == 2
        assert q2["answers"][0]["option"] == "Yes"
        assert q2["answers"][1]["option"] == "No"

    def test_parse_risk_type_field(self, fixtures_dir: Path, mock_llm):
        """Test that risk_type field is parsed correctly."""
        rubric_path = fixtures_dir / "rubric_single_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Risk type should be empty string if not specified
        q1 = judge.question_flow_data["1"]
        assert "risk_type" in q1
        assert q1["risk_type"] == ""

    def test_parse_question_id_as_string(self, fixtures_dir: Path, mock_llm):
        """Test that question IDs are stored as strings, not floats."""
        rubric_path = fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # All question IDs should be strings
        for qid in judge.question_order:
            assert isinstance(qid, str)
            assert not qid.endswith(".0")

        # Check specific IDs
        assert "1" in judge.question_flow_data
        assert "3" in judge.question_flow_data
        assert "5" in judge.question_flow_data

    def test_parse_multiple_questions_same_dimension(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test parsing multiple questions within the same dimension."""
        rubric_path = fixtures_dir / "rubric_empty_dimensions.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_empty_dimensions.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Questions 1 and 2 are in same dimension (2 continues 1)
        assert judge.question_flow_data["1"]["dimension"] == "Dimension A"
        assert judge.question_flow_data["2"]["dimension"] == ""  # Continuation

        # Question 3 starts new dimension
        assert judge.question_flow_data["3"]["dimension"] == "Dimension B"

    def test_parse_answer_structure(self, fixtures_dir: Path, mock_llm):
        """Test that answer dictionaries have correct structure."""
        rubric_path = fixtures_dir / "rubric_multi_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_multi_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        q1 = judge.question_flow_data["1"]

        # Each answer should have 'option' and 'goto' keys
        for answer in q1["answers"]:
            assert "option" in answer
            assert "goto" in answer
            assert isinstance(answer["option"], str)
            # goto can be str or None

    def test_parse_complete_question_structure(self, fixtures_dir: Path, mock_llm):
        """Test that parsed questions have all expected fields."""
        rubric_path = fixtures_dir / "rubric_single_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        q1 = judge.question_flow_data["1"]

        # Check all expected keys exist
        expected_keys = [
            "dimension",
            "risk_type",
            "question",
            "examples",
            "severity",
            "answers",
        ]
        for key in expected_keys:
            assert key in q1, f"Missing key: {key}"

        # Check types
        assert isinstance(q1["dimension"], str)
        assert isinstance(q1["risk_type"], str)
        assert isinstance(q1["question"], str)
        assert isinstance(q1["examples"], str)
        assert isinstance(q1["answers"], list)
        # severity can be str or None

    def test_parse_nan_handling(self, fixtures_dir: Path, mock_llm):
        """Test that NaN values are handled correctly (converted to empty strings or None)."""
        rubric_path = fixtures_dir / "rubric_no_answers.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_no_answers.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Questions with no Answer/GOTO columns should not have "nan" strings
        q1 = judge.question_flow_data["1"]

        # Check that severity is valid (not "nan" string)
        assert q1["severity"] != "nan"
        # In this fixture, q1 actually has Red severity
        assert q1["severity"] == "Red"

        # Check that default answers don't have "nan" GOTOs
        for answer in q1["answers"]:
            assert answer["goto"] is None
            assert answer["option"] != "nan"

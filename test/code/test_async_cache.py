#!/usr/bin/env python3
"""Test async file I/O and caching implementation."""

import asyncio

from judge.file_cache import get_cache_stats
from judge.llm_judge import LLMJudge
from judge.question_navigator import QuestionNavigator
from judge.runner import _create_evaluation_jobs


async def test_cache_module():
    """Test that cache module initializes correctly."""
    stats = get_cache_stats()
    assert stats["text_files_cached"] == 0
    assert stats["dataframes_cached"] == 0
    print("✅ Cache module initializes correctly")


async def test_question_navigator_async_factory():
    """Test QuestionNavigator.create() async factory method."""
    nav = await QuestionNavigator.create("data/rubric.tsv")
    assert nav is not None
    assert len(nav.question_order) > 0

    # Check that rubric was cached
    stats = get_cache_stats()
    assert stats["dataframes_cached"] == 1
    print(
        f"✅ QuestionNavigator.create() works (loaded {len(nav.question_order)} questions)"
    )


async def test_llm_judge_async_factory():
    """Test LLMJudge.create() async factory method."""
    judge = await LLMJudge.create(judge_model="claude-3-5-sonnet-20241022")
    assert judge is not None

    # Check that files were cached
    stats = get_cache_stats()
    assert stats["text_files_cached"] >= 1  # At least the question prompt template
    assert stats["dataframes_cached"] >= 1  # The rubric
    print(f"✅ LLMJudge.create() works (cache: {stats})")


def test_job_creation():
    """Test that job creation works with multiple judges."""
    conversation_files = [
        "conversations/test/conv1.txt",
        "conversations/test/conv2.txt",
    ]
    judge_models = {"claude-3-5-sonnet-20241022": 3, "gpt-4o": 2}
    output_folder = "test_output"

    jobs = _create_evaluation_jobs(conversation_files, judge_models, output_folder)

    expected_jobs = len(conversation_files) * sum(judge_models.values())
    assert len(jobs) == expected_jobs, f"Expected {expected_jobs} jobs, got {len(jobs)}"

    # Verify job structure
    for job in jobs:
        assert (
            len(job) == 4
        )  # (conversation_file, judge_model, instance, output_folder)
        assert job[1] in judge_models
        assert job[2] >= 1 and job[2] <= judge_models[job[1]]
        assert job[3] == output_folder

    print(f"✅ Job creation works ({len(jobs)} jobs created)")

    # Count jobs per model
    from collections import Counter

    job_counts = Counter(job[1] for job in jobs)
    expected_counts = {
        model: count * len(conversation_files) for model, count in judge_models.items()
    }
    assert dict(job_counts) == expected_counts
    print(f"✅ Jobs distributed correctly: {dict(job_counts)}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Async File I/O and Caching Implementation")
    print("=" * 60)

    # Test cache module
    await test_cache_module()

    # Test async factories
    await test_question_navigator_async_factory()
    await test_llm_judge_async_factory()

    # Test job creation (synchronous)
    test_job_creation()

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

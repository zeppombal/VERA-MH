"""Tests for utils.conversation_layout."""

import os

import pytest

from utils.conversation_layout import resolve_conversation_input


@pytest.mark.unit
class TestResolveConversationInput:
    def test_nested_with_txt_prefers_conversations_subdir(self, tmp_path):
        run = tmp_path / "p_test__t1__r1"
        conv = run / "conv.txt"
        nested = run / "conversations"
        nested.mkdir(parents=True)
        conv.write_text("x", encoding="utf-8")
        (nested / "a.txt").write_text("hi", encoding="utf-8")

        td, gen, base = resolve_conversation_input(str(run))
        assert td == str(nested)
        assert gen == str(run)
        assert base == run.name

    def test_legacy_flat_root_txt(self, tmp_path):
        flat = tmp_path / "old_run"
        flat.mkdir()
        (flat / "1.txt").write_text("x", encoding="utf-8")

        td, gen, base = resolve_conversation_input(str(flat))
        assert td == str(flat)
        assert gen is None
        assert base == flat.name

    def test_empty_nested_root_txt_uses_legacy(self, tmp_path):
        """Root transcripts win when nested exists but has no .txt files."""
        run = tmp_path / "p_mixed"
        run.mkdir()
        (run / "conversations").mkdir()
        (run / "root.txt").write_text("x", encoding="utf-8")

        td, gen, _ = resolve_conversation_input(str(run))
        assert td == str(run)
        assert gen is None

    def test_new_run_defaults_to_nested_path(self, tmp_path):
        run = tmp_path / "p_new"
        run.mkdir()

        td, gen, base = resolve_conversation_input(str(run))
        assert td == os.path.join(str(run), "conversations")
        assert gen == str(run)
        assert base == run.name

    def test_nested_dir_exists_but_empty_no_root_txt(self, tmp_path):
        run = tmp_path / "p_empty_nested"
        run.mkdir()
        nested = run / "conversations"
        nested.mkdir()

        td, gen, _ = resolve_conversation_input(str(run))
        assert td == str(nested)
        assert gen == str(run)

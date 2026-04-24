"""Shared helpers for generation-run folder layout (nested vs legacy flat)."""

import os
from typing import Optional


def resolve_conversation_input(folder: str) -> tuple[str, Optional[str], str]:
    """
    Resolve where transcript ``.txt`` files live under a generation or
    conversation folder.

    Returns:
        (transcripts_dir, gen_run_root_or_none, conversation_run_basename)

    - If ``folder/conversations/`` contains ``.txt`` files, ``folder`` is treated as
      a generation run root; transcripts are read from ``folder/conversations``.
    - Otherwise, if ``folder`` itself contains ``.txt`` files, use the legacy flat
      layout (``gen_run_root`` is None).
    - If ``folder/conversations/`` exists but has no ``.txt`` files yet, use that
      path (default nested layout for new runs).
    - If there is no ``conversations/`` subdirectory and no transcripts at the root,
      default to ``folder/conversations`` as the transcript directory so generation
      and judging agree on the nested layout.
    """
    folder = os.path.normpath(os.path.abspath(folder))
    nested = os.path.join(folder, "conversations")
    basename = os.path.basename(folder)

    def dir_has_txt_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return any(name.endswith(".txt") for name in os.listdir(path))

    if dir_has_txt_files(nested):
        return nested, folder, basename
    if dir_has_txt_files(folder):
        return folder, None, basename
    if os.path.isdir(nested):
        return nested, folder, basename
    return nested, folder, basename

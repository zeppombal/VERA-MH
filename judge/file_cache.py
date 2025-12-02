"""Module-level file cache for judge evaluation system.

This module provides thread-safe caching of frequently accessed template
and rubric files to avoid repeated disk I/O during evaluations.
"""

import asyncio
from pathlib import Path
from typing import Dict, Tuple

import aiofiles
import pandas as pd

# Module-level cache storage
_cache: Dict[str, str] = {}
_cache_lock = asyncio.Lock()

# Cache for pandas DataFrames (rubric files)
_df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_df_cache_lock = asyncio.Lock()


async def get_cached_file(file_path: str) -> str:
    """
    Get file contents from cache or load and cache if not present.

    Thread-safe lazy loading with async I/O.

    Args:
        file_path: Path to file to read

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    # Fast path: check cache without lock
    if file_path in _cache:
        return _cache[file_path]

    # Slow path: acquire lock and load file
    async with _cache_lock:
        # Double-check pattern: another coroutine may have loaded it
        if file_path in _cache:
            return _cache[file_path]

        # Verify file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load file asynchronously
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Cache and return
        _cache[file_path] = content
        return content


async def get_cached_dataframe(file_path: str, sep: str = "\t") -> pd.DataFrame:
    """
    Get pandas DataFrame from cache or load and cache if not present.

    Used for rubric TSV files. Note: pandas doesn't have async I/O,
    so we use asyncio.to_thread to avoid blocking.

    Args:
        file_path: Path to CSV/TSV file
        sep: Separator character (default: tab)

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    cache_key = (file_path, sep)

    # Fast path: check cache without lock
    if cache_key in _df_cache:
        return _df_cache[cache_key]

    # Slow path: acquire lock and load file
    async with _df_cache_lock:
        # Double-check pattern
        if cache_key in _df_cache:
            return _df_cache[cache_key]

        # Verify file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load DataFrame in thread pool (pandas is sync only)
        df = await asyncio.to_thread(pd.read_csv, file_path, sep=sep)

        # Cache and return
        _df_cache[cache_key] = df
        return df


def clear_cache():
    """Clear all cached files. Useful for testing."""
    _cache.clear()
    _df_cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics for monitoring."""
    return {
        "text_files_cached": len(_cache),
        "dataframes_cached": len(_df_cache),
    }

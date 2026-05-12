"""Small terminal progress bar for async batch runners."""

from __future__ import annotations

import sys
import time
from typing import TextIO


class ProgressBar:
    """Render a compact progress bar when attached to an interactive terminal."""

    def __init__(
        self,
        total: int,
        label: str,
        *,
        width: int = 30,
        stream: TextIO | None = None,
    ) -> None:
        self.total = max(total, 0)
        self.label = label
        self.width = width
        self.stream = stream or sys.stderr
        self.count = 0
        self._start = time.monotonic()
        self._last_len = 0
        self.enabled = self.total > 0 and self.stream.isatty()

    def __enter__(self) -> "ProgressBar":
        if self.enabled:
            self._render()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def update(self, amount: int = 1) -> None:
        self.count = min(self.total, self.count + amount)
        if self.enabled:
            self._render()

    def write(self, message: str) -> None:
        if self.enabled:
            self.stream.write("\r" + (" " * self._last_len) + "\r")
        print(message, file=self.stream)
        if self.enabled:
            self._render()

    def close(self) -> None:
        if not self.enabled:
            return
        self._render()
        self.stream.write("\n")
        self.stream.flush()

    def _render(self) -> None:
        ratio = self.count / self.total if self.total else 1.0
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self._start
        rate = self.count / elapsed if elapsed > 0 and self.count else 0.0
        eta = (self.total - self.count) / rate if rate else None
        eta_text = self._format_seconds(eta) if eta is not None else "--:--"
        line = (
            f"{self.label}: [{bar}] {self.count}/{self.total} "
            f"({ratio * 100:5.1f}%) elapsed {self._format_seconds(elapsed)} "
            f"eta {eta_text}"
        )
        self.stream.write("\r" + line + (" " * max(0, self._last_len - len(line))))
        self.stream.flush()
        self._last_len = len(line)

    @staticmethod
    def _format_seconds(seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        seconds_int = max(0, int(seconds))
        minutes, seconds_int = divmod(seconds_int, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds_int:02d}"
        return f"{minutes:02d}:{seconds_int:02d}"

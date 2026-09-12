"""Quiet wrapper around BasicPitch (filters CoreML debug spam)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator

_NOISE_PREFIXES = ("isfinite:", "shape:", "dtype:")
_SKIP_LINES = _NOISE_PREFIXES + ("Predicting MIDI for",)


class _FilteredStream:
    """Drop known BasicPitch debug lines; pass everything else through."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, data: str) -> int:
        if not data:
            return 0
        n = 0
        for line in data.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(p) for p in _SKIP_LINES):
                continue
            n += self._stream.write(line)
        return n

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextmanager
def quiet_basic_pitch_output() -> Iterator[None]:
    """Suppress BasicPitch per-window CoreML debug prints on stdout."""
    old = sys.stdout
    sys.stdout = _FilteredStream(old)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old


def predict_notes(
    audio_path: str,
    model_path: str,
    **kwargs: Any,
) -> tuple[Any, Any, list]:
    """Run ``basic_pitch.inference.predict`` without CoreML debug noise."""
    from basic_pitch.inference import predict

    with quiet_basic_pitch_output():
        return predict(audio_path, model_or_model_path=model_path, **kwargs)

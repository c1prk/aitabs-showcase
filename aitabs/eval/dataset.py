"""Discover reference GP5 + audio pairs on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
_GP5_EXTS = {".gp5", ".gp4", ".gpx", ".gp"}


@dataclass(frozen=True)
class EvalPair:
    """One evaluation clip: audio + reference tab.

    The reference is either a Guitar Pro file (``reference_gp5_path``) or a
    GuitarSet JAMS annotation (``reference_jams_path``); exactly one is set.
    """

    clip_id: str
    audio_path: Path
    reference_gp5_path: Path | None = None
    stem_path: Path | None = None  # optional pre-separated guitar wav
    reference_jams_path: Path | None = None  # GuitarSet annotation (real audio)


def _stem_id(path: Path) -> str:
    return path.stem


def discover_pairs(
    root: str | Path,
    *,
    audio_dir: str = "audio",
    reference_dir: str = "reference",
    stems_dir: str | None = "stems",
) -> list[EvalPair]:
    """Find clip pairs under ``root``.

    Supported layouts:

    1. Flat (same folder)::

        root/song_01.mp3 + root/song_01.gp5

    2. Split folders::

        root/audio/song_01.mp3 + root/reference/song_01.gp5

    3. Optional stems::

        root/stems/song_01.wav
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Eval root not found: {root}")

    gp5_files: list[Path] = []
    for ext in _GP5_EXTS:
        gp5_files.extend(root.rglob(f"*{ext}"))
    if not gp5_files:
        ref_root = root / reference_dir
        if ref_root.is_dir():
            for ext in _GP5_EXTS:
                gp5_files.extend(ref_root.glob(f"*{ext}"))

    pairs: list[EvalPair] = []
    for gp5_path in sorted(gp5_files):
        clip_id = _stem_id(gp5_path)
        audio_path = _find_audio(root, clip_id, audio_dir, gp5_path.parent)
        if audio_path is None:
            continue
        stem_path = None
        if stems_dir:
            for cand in (
                root / stems_dir / f"{clip_id}.wav",
                root / stems_dir / clip_id / "other.wav",
            ):
                if cand.is_file():
                    stem_path = cand
                    break
        pairs.append(
            EvalPair(
                clip_id=clip_id,
                audio_path=audio_path,
                reference_gp5_path=gp5_path,
                stem_path=stem_path,
            )
        )

    return pairs


# GuitarSet audio variants → filename suffix on the mono recordings.
#   mic = reference microphone (clean acoustic sound, best for the clean MVP)
#   mix = mono mix of the hexaphonic pickup
_GUITARSET_AUDIO_SUFFIX = {"mic": "_mic", "mix": "_mix", "hex": "_hex"}


def discover_guitarset_pairs(
    root: str | Path,
    *,
    solo_only: bool = True,
    audio_kind: str = "mic",
) -> list[EvalPair]:
    """Find GuitarSet (audio + ``.jams``) pairs under ``root``.

    Works against an extracted GuitarSet download, where annotations live in an
    ``annotation/`` folder and audio in ``audio_mono-mic/`` (filenames like
    ``00_BN1-129-Eb_solo_mic.wav`` alongside ``00_BN1-129-Eb_solo.jams``).

    Args:
        root: GuitarSet root (or any folder holding ``.jams`` + audio).
        solo_only: Keep only ``*_solo`` excerpts — single-note improvisations,
            the clean monophonic case for an MVP. Set False to include ``comp``.
        audio_kind: ``"mic"`` (clean acoustic, default), ``"mix"`` or ``"hex"``.

    Returns:
        EvalPairs with ``reference_jams_path`` set (``reference_gp5_path`` None).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"GuitarSet root not found: {root}")

    suffix = _GUITARSET_AUDIO_SUFFIX.get(audio_kind, "_mic")
    jams_files = sorted(root.rglob("*.jams"))

    pairs: list[EvalPair] = []
    for jams_path in jams_files:
        clip_id = jams_path.stem
        if solo_only and "_solo" not in clip_id:
            continue
        audio_path = _find_guitarset_audio(root, clip_id, suffix, jams_path.parent)
        if audio_path is None:
            continue
        pairs.append(
            EvalPair(
                clip_id=clip_id,
                audio_path=audio_path,
                reference_gp5_path=None,
                reference_jams_path=jams_path,
            )
        )
    return pairs


def _find_guitarset_audio(
    root: Path,
    clip_id: str,
    suffix: str,
    jams_parent: Path,
) -> Path | None:
    search_dirs = [
        jams_parent,
        root,
        root / "audio",
        root / f"audio_mono{suffix.replace('_', '-')}",  # e.g. audio_mono-mic
        root / "audio_mono-mic",
    ]
    stems = [f"{clip_id}{suffix}", clip_id]
    for directory in search_dirs:
        for stem in stems:
            for ext in _AUDIO_EXTS:
                cand = directory / f"{stem}{ext}"
                if cand.is_file():
                    return cand
    return None


def _find_audio(
    root: Path,
    clip_id: str,
    audio_dir: str,
    gp5_parent: Path,
) -> Path | None:
    search_dirs = [
        gp5_parent,
        root,
        root / audio_dir,
    ]
    for directory in search_dirs:
        for ext in _AUDIO_EXTS:
            cand = directory / f"{clip_id}{ext}"
            if cand.is_file():
                return cand
    return None

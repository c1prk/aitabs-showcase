#!/usr/bin/env python3
"""Prepare a clean-acoustic GuitarSet eval subset (solo + mic).

GuitarSet (https://zenodo.org/records/3371780, MIT licensed) is real acoustic
guitar recorded with a reference microphone. Each excerpt was played twice:
``comp`` (chords) and ``solo`` (single-note improvisation). For a clean-guitar
note-accuracy MVP we want the ``solo`` takes from the ``mic`` recording.

This script links (or copies) the ``*_solo_mic.wav`` audio and matching
``*_solo.jams`` annotations into ``data/eval/guitarset/`` so the eval harness
can score them::

    python scripts/eval_dataset.py data/eval/guitarset --guitarset

Two modes:

  1. Point at an already-extracted GuitarSet download::

        python scripts/prepare_guitarset.py --guitarset-dir /path/to/GuitarSet

  2. Download annotation.zip + audio_mono-mic.zip from Zenodo first::

        python scripts/prepare_guitarset.py --download

Use ``--limit N`` to grab a small smoke-test subset, ``--include-comp`` to also
pull chord takes, and ``--copy`` to copy instead of symlink.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

_ZENODO = "https://zenodo.org/records/3371780/files"
_ZIPS = {
    "annotation": "annotation.zip",
    "audio_mono-mic": "audio_mono-mic.zip",
}


def _download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            print(f"  downloading {dest.name} (attempt {attempt + 1})...", flush=True)
            with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            return
        except Exception as exc:  # noqa: BLE001 - best-effort network retry
            last_exc = exc
            wait = 2 ** (attempt + 1)
            print(f"  failed ({exc}); retrying in {wait}s", file=sys.stderr)
            import time

            time.sleep(wait)
    raise RuntimeError(f"Could not download {url}: {last_exc}")


def _ensure_guitarset(cache_dir: Path) -> Path:
    """Download + extract the mic/annotation zips into ``cache_dir``."""
    for name, zip_name in _ZIPS.items():
        zip_path = cache_dir / zip_name
        _download(f"{_ZENODO}/{zip_name}?download=1", zip_path)
        target = cache_dir / name
        if not target.is_dir():
            print(f"  extracting {zip_name}...", flush=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
    return cache_dir


def _find(root: Path, name: str) -> list[Path]:
    return sorted(root.rglob(name))


def _link(src: Path, dest: Path, copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(src, dest)
    else:
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GuitarSet solo+mic eval subset")
    parser.add_argument(
        "--guitarset-dir",
        type=Path,
        default=None,
        help="Path to an already-extracted GuitarSet (with annotation/ + audio_mono-mic/)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download annotation.zip + audio_mono-mic.zip from Zenodo",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/eval/_guitarset_raw"),
        help="Where to store downloaded/extracted GuitarSet (default: data/eval/_guitarset_raw)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/guitarset"),
        help="Output eval folder (default: data/eval/guitarset)",
    )
    parser.add_argument("--include-comp", action="store_true", help="Also include chord takes")
    parser.add_argument("--limit", type=int, default=None, help="Only prepare N pairs (smoke test)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = parser.parse_args()

    if args.guitarset_dir is not None:
        source = args.guitarset_dir
    elif args.download:
        source = _ensure_guitarset(args.cache_dir)
    else:
        parser.error("Provide --guitarset-dir PATH or --download")
        return

    if not source.is_dir():
        print(f"GuitarSet source not found: {source}", file=sys.stderr)
        sys.exit(1)

    jams_files = _find(source, "*.jams")
    if not jams_files:
        print(f"No .jams files under {source}", file=sys.stderr)
        sys.exit(1)

    audio_dir = args.out / "audio"
    anno_dir = args.out / "annotation"
    prepared = 0
    skipped_no_audio = 0

    for jams_path in jams_files:
        clip_id = jams_path.stem
        if not args.include_comp and "_solo" not in clip_id:
            continue

        mic_name = f"{clip_id}_mic.wav"
        matches = _find(source, mic_name)
        if not matches:
            skipped_no_audio += 1
            continue

        _link(matches[0], audio_dir / mic_name, args.copy)
        _link(jams_path, anno_dir / jams_path.name, args.copy)
        prepared += 1
        if args.limit and prepared >= args.limit:
            break

    print(f"\nPrepared {prepared} pair(s) -> {args.out}")
    if skipped_no_audio:
        print(f"  ({skipped_no_audio} jams skipped: no matching _mic.wav)")
    if prepared:
        print("\nNext:")
        print(f"  python scripts/eval_dataset.py {args.out} --guitarset")


if __name__ == "__main__":
    main()

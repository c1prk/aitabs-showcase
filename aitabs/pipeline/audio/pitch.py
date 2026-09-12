"""BasicPitch note detection — converts isolated guitar audio to MIDI note events."""

from __future__ import annotations

from typing import Protocol, TypedDict


class NoteEvent(TypedDict):
    start: float       # onset time in seconds
    end: float         # offset time in seconds
    pitch_midi: int    # MIDI pitch number (e.g. 40 = E2, 64 = E4)
    confidence: float  # model confidence 0.0–1.0


def _as_note_event(note: NoteEvent | tuple) -> NoteEvent:
    if isinstance(note, dict):
        return NoteEvent(
            start=float(note["start"]),
            end=float(note["end"]),
            pitch_midi=int(note["pitch_midi"]),
            confidence=float(note["confidence"]),
        )
    start, end, pitch, confidence, *_rest = note
    return NoteEvent(
        start=float(start),
        end=float(end),
        pitch_midi=int(pitch),
        confidence=float(confidence),
    )


def clean_notes(
    notes: list[NoteEvent] | list[tuple],
    pitch_merge_sec: float = 0.15,
) -> list[NoteEvent]:
    """Remove duplicate detections of the same pitch at nearly the same time.

    BasicPitch often emits the same note twice within ~50–150 ms (harmonics,
    onset chatter). Keep the highest-confidence event per (pitch, time window).

    Args:
        notes: NoteEvent dicts or BasicPitch tuples
            (start, end, pitch, confidence, ...).
        pitch_merge_sec: Max start-time gap to treat as one onset (seconds).

    Returns:
        Deduplicated notes sorted by start time.
    """
    normalized = [_as_note_event(n) for n in notes]
    normalized.sort(key=lambda n: (n["start"], -n["confidence"]))

    kept: list[NoteEvent] = []
    for note in normalized:
        merged = False
        for i, existing in enumerate(kept):
            if (
                note["pitch_midi"] == existing["pitch_midi"]
                and abs(note["start"] - existing["start"]) < pitch_merge_sec
            ):
                if note["confidence"] > existing["confidence"]:
                    kept[i] = note
                merged = True
                break
        if not merged:
            kept.append(note)

    return sorted(kept, key=lambda n: n["start"])


# Intervals (semitones) at which a string's strongest overtones alias to a
# detected "note": octave, octave+fifth, two octaves.
_HARMONIC_INTERVALS = (12, 19, 24)


def suppress_octave_harmonics(
    notes: list[NoteEvent] | list[tuple],
    *,
    onset_window_sec: float = 0.05,
    intervals: tuple[int, ...] = _HARMONIC_INTERVALS,
    max_conf_ratio: float = 1.0,
) -> list[NoteEvent]:
    """Drop detections that look like octave/overtone ghosts of a coincident note.

    The dominant false positive on real guitar audio is a string's overtone being
    detected as its own note (octave / octave+fifth / two octaves above the
    fundamental). A detection ``H`` is treated as a ghost of fundamental ``F`` when:

    - ``H.pitch - F.pitch`` is one of ``intervals`` (default 12, 19, 24), AND
    - their onsets are near-coincident (``|H.start - F.start| <= onset_window_sec``)
      — i.e. excited by the same pluck. A *separately played* octave note has its
      own, later onset and is kept (protects melody-over-bass lines), AND
    - ``H`` is not stronger than the fundamental
      (``H.confidence <= max_conf_ratio * F.confidence``).

    Conservative by design — it only removes upper notes that coincide with a
    plausible lower fundamental. For dense chordal material a real octave double
    can coincide with its lower note; lower ``max_conf_ratio`` (e.g. ~0.7) so only
    clearly-weaker ghosts are removed.

    Args:
        notes: Detected note events (dicts or BasicPitch tuples).
        onset_window_sec: Max onset gap to count as the same pluck.
        intervals: Semitone offsets above a fundamental treated as harmonics.
        max_conf_ratio: ``H`` is a ghost only if its confidence is at most this
            fraction of the fundamental's.

    Returns:
        Surviving notes, sorted by start time.
    """
    normalized = [_as_note_event(n) for n in notes]
    interval_set = set(intervals)

    drop: set[int] = set()
    for i, h in enumerate(normalized):
        for j, f in enumerate(normalized):
            if j == i:
                continue
            if (h["pitch_midi"] - f["pitch_midi"]) not in interval_set:
                continue
            if abs(h["start"] - f["start"]) > onset_window_sec:
                continue
            if h["confidence"] > max_conf_ratio * f["confidence"]:
                continue
            drop.add(i)
            break

    survivors = [n for i, n in enumerate(normalized) if i not in drop]
    return sorted(survivors, key=lambda n: n["start"])


def detect_notes(
    audio_path: str,
    confidence_threshold: float = 0.5,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    pitch_merge_sec: float = 0.15,
) -> list[NoteEvent]:
    """Run BasicPitch on a clean guitar audio file and return detected note events.

    BasicPitch is a polyphonic pitch estimator from Spotify Research. It outputs
    note-level events (onset, offset, MIDI pitch, confidence) rather than raw
    frequency estimates, which makes downstream processing much easier.

    Args:
        audio_path: Path to the isolated guitar WAV (output from separate_guitar).
        confidence_threshold: Minimum per-note confidence to include. Guitar
            notes below 0.5 are usually false positives from string resonance.
        onset_threshold: Controls how sensitive the onset detector is.
            Lower = more notes detected (more false positives).
        frame_threshold: Per-frame activation threshold used internally.
        pitch_merge_sec: Passed to ``clean_notes`` after filtering.

    Returns:
        List of NoteEvent dicts sorted by start time.
    """
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from aitabs.pipeline.audio.basic_pitch_infer import predict_notes

    _model_output, _midi_data, note_events = predict_notes(
        audio_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=58,
    )

    results: list[NoteEvent] = []
    for start, end, pitch, confidence, _pitch_bend in note_events:
        if confidence >= confidence_threshold:
            results.append(
                NoteEvent(
                    start=float(start),
                    end=float(end),
                    pitch_midi=int(pitch),
                    confidence=float(confidence),
                )
            )

    return clean_notes(results, pitch_merge_sec=pitch_merge_sec)


# ---------------------------------------------------------------------------
# Pluggable detector interface
# ---------------------------------------------------------------------------

class Detector(Protocol):
    def detect(
        self,
        audio_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        confidence_threshold: float,
        minimum_note_length_ms: float,
        minimum_frequency_hz: float | None,
        maximum_frequency_hz: float | None,
        melodia_trick: bool,
    ) -> list[NoteEvent]: ...


class BasicPitchDetector:
    """Default detector: BasicPitch ICASSP 2022 model with supplied thresholds."""

    def detect(
        self,
        audio_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        confidence_threshold: float,
        minimum_note_length_ms: float,
        minimum_frequency_hz: float | None,
        maximum_frequency_hz: float | None,
        melodia_trick: bool,
    ) -> list[NoteEvent]:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from aitabs.pipeline.audio.basic_pitch_infer import predict_notes

        _model_output, _midi_data, note_events = predict_notes(
            audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length_ms,
            minimum_frequency=minimum_frequency_hz,
            maximum_frequency=maximum_frequency_hz,
            melodia_trick=melodia_trick,
        )
        raw: list[NoteEvent] = []
        for start, end, pitch, confidence, _bend in note_events:
            if confidence >= confidence_threshold:
                raw.append(NoteEvent(
                    start=float(start),
                    end=float(end),
                    pitch_midi=int(pitch),
                    confidence=float(confidence),
                ))
        return raw


class RecallFirstBasicPitchDetector:
    """Higher-recall variant: lower onset + confidence thresholds.

    Finds more notes at the cost of more false positives. Use for A/B comparison
    against BasicPitchDetector to measure the recall–precision tradeoff on
    GuitarSet and song clips before deciding whether to fine-tune.
    """

    _CONFIDENCE: float = 0.3
    _ONSET: float = 0.3

    def detect(
        self,
        audio_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        confidence_threshold: float,
        minimum_note_length_ms: float,
        minimum_frequency_hz: float | None,
        maximum_frequency_hz: float | None,
        melodia_trick: bool,
    ) -> list[NoteEvent]:
        return BasicPitchDetector().detect(
            audio_path,
            onset_threshold=self._ONSET,
            frame_threshold=frame_threshold,
            confidence_threshold=self._CONFIDENCE,
            minimum_note_length_ms=minimum_note_length_ms,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
            melodia_trick=melodia_trick,
        )


class FinetunedDetector:
    """BasicPitch fine-tuned on GuitarSet solo recordings.

    Loads the SavedModel saved by ``scripts/finetune_basicpitch.py`` and runs
    inference through the standard BasicPitch pipeline.  Pass ``model_path``
    explicitly or let it default to ``data/models/basicpitch_guitarset_ft``.
    """

    _DEFAULT_PATH = "data/models/basicpitch_guitarset_ft"

    def __init__(self, model_path: str | None = None) -> None:
        import os
        self._model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", self._DEFAULT_PATH,
        )

    def detect(
        self,
        audio_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        confidence_threshold: float,
        minimum_note_length_ms: float,
        minimum_frequency_hz: float | None,
        maximum_frequency_hz: float | None,
        melodia_trick: bool,
    ) -> list[NoteEvent]:
        from aitabs.pipeline.audio.basic_pitch_infer import predict_notes

        _model_output, _midi_data, note_events = predict_notes(
            audio_path,
            self._model_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length_ms,
            minimum_frequency=minimum_frequency_hz,
            maximum_frequency=maximum_frequency_hz,
            melodia_trick=melodia_trick,
        )
        raw: list[NoteEvent] = []
        for start, end, pitch, confidence, _bend in note_events:
            if confidence >= confidence_threshold:
                raw.append(NoteEvent(
                    start=float(start),
                    end=float(end),
                    pitch_midi=int(pitch),
                    confidence=float(confidence),
                ))
        return raw


class KongDetector:
    """Kong et al. (2021) high-resolution piano transcription model adapted for guitar.

    Architecture: Apache 2.0 (bytedance/piano_transcription).
    Weights: CC-BY-4.0 (Zenodo 4034264).
    ⚠ Known TODO: weights were pre-trained on MAESTRO (CC BY-NC-SA 4.0).
      Fine-tune only on MIT-licensed data (GuitarSet) before commercial release.
      Clean pre-train from scratch on MIDI/FluidR3 is the post-rough milestone.
    """

    _CHECKPOINT_URL = (
        "https://zenodo.org/record/4034264/files/"
        "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
    )
    _CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
    _MIN_CHECKPOINT_BYTES = 1.6e8
    # Class-level cache: shared across all instances so repeated get_detector()
    # calls per clip don't reload the 165 MB checkpoint.
    _shared_models: "dict[str, object]" = {}

    def __init__(self, checkpoint_path: str | None = None, *, pcen: bool = False) -> None:
        from pathlib import Path
        self._checkpoint_path = checkpoint_path or str(
            Path.home() / "piano_transcription_inference_data" / self._CHECKPOINT_NAME
        )
        # ``pcen=True``: the checkpoint has a trainable PCEN front-end (RESEARCH_PCEN.md).
        # PCEN must be attached BEFORE loading or the state_dict keys won't match, so
        # we build on the base pretrained model, attach PCEN, then load the PCEN weights.
        self._pcen = pcen

    def _ensure_checkpoint(self) -> None:
        import os
        import urllib.request
        path = self._checkpoint_path
        if os.path.exists(path) and os.path.getsize(path) >= self._MIN_CHECKPOINT_BYTES:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading Kong checkpoint (~165 MB) -> {path}")
        urllib.request.urlretrieve(self._CHECKPOINT_URL, path)
        print("Download complete.")

    def _get_model(self, onset_threshold: float, frame_threshold: float):
        import torch
        from piano_transcription_inference import PianoTranscription
        cache_key = (self._checkpoint_path, self._pcen)
        if cache_key not in KongDetector._shared_models:
            if self._pcen:
                # Build a valid model on the base pretrained weights (downloaded by
                # PianoTranscription if absent), attach PCEN, then load the PCEN
                # checkpoint — keys only match once PCEN is attached.
                from aitabs.pipeline.audio.pcen import attach_pcen_frontend
                tr = PianoTranscription(model_type="Note_pedal", checkpoint_path=None,
                                        device=torch.device("cpu"))
                attach_pcen_frontend(tr.model.note_model)
                st = torch.load(self._checkpoint_path, map_location="cpu")
                tr.model.load_state_dict(st["model"])
                tr.model.eval()
                KongDetector._shared_models[cache_key] = tr
            else:
                self._ensure_checkpoint()
                KongDetector._shared_models[cache_key] = PianoTranscription(
                    model_type="Note_pedal",
                    checkpoint_path=self._checkpoint_path,
                    device=torch.device("cpu"),
                )
        model = KongDetector._shared_models[cache_key]
        model.onset_threshold = onset_threshold
        model.frame_threshold = frame_threshold
        return model

    @staticmethod
    def _hz_to_midi(hz: float) -> int:
        import math
        return int(round(12 * math.log2(max(hz, 1.0) / 440.0) + 69))

    def detect(
        self,
        audio_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        confidence_threshold: float,
        minimum_note_length_ms: float,
        minimum_frequency_hz: float | None,
        maximum_frequency_hz: float | None,
        melodia_trick: bool,  # unused by Kong
    ) -> list[NoteEvent]:
        import numpy as np
        import soundfile as sf
        import librosa
        from piano_transcription_inference.config import sample_rate as KONG_SR

        # Load audio at 16 kHz mono
        y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != KONG_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=KONG_SR)

        model = self._get_model(onset_threshold, frame_threshold)
        result = model.transcribe(y, midi_path=None)
        note_events = result.get("est_note_events", [])

        # Frequency → MIDI bounds
        min_midi = self._hz_to_midi(minimum_frequency_hz) if minimum_frequency_hz else 0
        max_midi = self._hz_to_midi(maximum_frequency_hz) if maximum_frequency_hz else 127
        min_len_sec = minimum_note_length_ms / 1000.0

        raw: list[NoteEvent] = []
        for ev in note_events:
            onset = float(ev["onset_time"])
            offset = float(ev["offset_time"])
            midi = int(ev["midi_note"])
            conf = float(ev["velocity"]) / 127.0  # velocity as confidence proxy
            if conf < confidence_threshold:
                continue
            if midi < min_midi or midi > max_midi:
                continue
            if offset - onset < min_len_sec:
                continue
            raw.append(NoteEvent(start=onset, end=offset, pitch_midi=midi, confidence=conf))

        return sorted(raw, key=lambda n: n["start"])


_DETECTORS: dict[str, type] = {
    "basic_pitch": BasicPitchDetector,
    "recall_first": RecallFirstBasicPitchDetector,
    "kong": KongDetector,
}


def get_detector(
    name: str,
    *,
    model_path: str | None = None,
    pcen: bool = False,
) -> BasicPitchDetector | RecallFirstBasicPitchDetector | FinetunedDetector | KongDetector:
    """Return a detector instance by name.

    Args:
        name: ``"basic_pitch"``, ``"recall_first"``, ``"finetuned"``, or ``"kong"``.
        model_path: path to fine-tuned SavedModel (``"finetuned"``) or Kong
            checkpoint (``.pth``, ``"kong"``).

    Raises:
        ValueError: Unknown detector name.
    """
    if name == "finetuned":
        return FinetunedDetector(model_path)
    if name == "kong":
        return KongDetector(model_path, pcen=pcen)
    if name == "ensemble":
        from aitabs.pipeline.audio.ensemble import EnsembleDetector
        return EnsembleDetector(model_path, kong_pcen=pcen)
    cls = _DETECTORS.get(name)
    if cls is None:
        known = ", ".join(f'"{k}"' for k in [*_DETECTORS, "finetuned", "ensemble"])
        raise ValueError(f"Unknown detector: {name!r}. Choose one of {known}.")
    return cls()

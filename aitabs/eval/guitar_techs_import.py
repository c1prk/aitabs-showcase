"""Load the Guitar-TECHS dataset (Zenodo 14963133, CC BY 4.0) for training.

Structure per content folder (P{n}_{chords,scales,singlenotes,techniques,music}):
    midi/*.mid                     6 tracks named e,B,G,D,A,e (per string)
    audio/directinput/*.wav        DI audio (48 kHz mono), synced to the MIDI
    audio/micamp/*.wav             amp-mic audio
    video/{ego,exo}/*.mp3          performer/audience perspective audio

The 6 MIDI tracks are per-string (track 0='e' high E ... track 5='E' low E), so
we get pitch + onset + STRING directly. aitabs string index (0=low E .. 5=high e)
= 5 - track_index. Fret = pitch - open-string pitch.
"""
from __future__ import annotations

from pathlib import Path

from aitabs.pipeline.mapping.guitar import STANDARD_TUNING, MAX_FRETS

_TRACK_TO_AITABS_STRING = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0}  # e,B,G,D,A,E -> 5..0


def load_guitar_techs_clips(root: str | Path, *, audio_kind: str = "directinput") -> list[dict]:
    """Return clips [{clip_id, audio_path, notes}] for every content folder found.

    ``notes``: list of {start, end, pitch_midi, string, fret, confidence} sorted by
    start. ``audio_kind``: 'directinput' (clean, recommended) or 'micamp'.
    """
    import pretty_midi

    root = Path(root)
    clips: list[dict] = []
    for midi_path in sorted(root.glob("*/midi/*.mid")):
        content_dir = midi_path.parent.parent
        wav = next(iter((content_dir / "audio" / audio_kind).glob("*.wav")), None)
        if wav is None or not wav.is_file():
            continue
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes: list[dict] = []
        for t_idx, inst in enumerate(pm.instruments[:6]):
            aitabs_str = _TRACK_TO_AITABS_STRING.get(t_idx, 5 - t_idx)
            open_midi = STANDARD_TUNING[aitabs_str]
            for n in inst.notes:
                pitch = int(n.pitch)
                fret = pitch - open_midi
                if not (0 <= fret <= MAX_FRETS):
                    fret = max(0, min(fret, MAX_FRETS))
                notes.append({
                    "start": float(n.start), "end": float(n.end), "pitch_midi": pitch,
                    "string": aitabs_str, "fret": fret, "confidence": 1.0,
                })
        notes.sort(key=lambda x: (x["start"], x["pitch_midi"]))
        clips.append({"clip_id": content_dir.name, "audio_path": str(wav), "notes": notes})
    return clips

"""Import note events from Guitar Pro 7+ (.gp) GPIF archives."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from aitabs.pipeline.mapping.guitar import GuitarSetup, STANDARD_TUNING, fret_to_midi

from .types import EvalNote

_TICKS_PER_QUARTER = 960
_NOTE_VALUE_TICKS = {
    "Whole": 3840,
    "Half": 1920,
    "Quarter": 960,
    "Eighth": 480,
    "16th": 240,
    "ThirtySecond": 120,
    "64th": 60,
}
_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL = {"": 0, "Sharp": 1, "Flat": -1, "DoubleSharp": 2, "DoubleFlat": -2}


def _pitch_to_midi(step: str, accidental: str, octave: int) -> int:
    return (int(octave) + 1) * 12 + _STEP_TO_SEMITONE[step] + _ACCIDENTAL.get(accidental or "", 0)


def _read_gpif(path: Path) -> ET.Element:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("Content/score.gpif")
    return ET.fromstring(raw)


def _header_bpm(root: ET.Element) -> float:
    for auto in root.findall(".//MasterTrack/Automations/Automation"):
        if (auto.findtext("Type") or "").lower() == "tempo":
            val = (auto.findtext("Value") or "").strip()
            if val:
                return float(val.split()[0])
    return 120.0


def _capo_and_tuning(root: ET.Element, track_index: int = 0) -> GuitarSetup:
    tracks = root.findall("./Tracks/Track")
    if not tracks:
        return GuitarSetup()
    track = tracks[track_index]
    capo = 0
    tuning = list(STANDARD_TUNING)
    for prop in track.findall(".//Property"):
        name = prop.get("name")
        if name == "CapoFret":
            capo = int(prop.findtext("Fret") or 0)
        elif name == "Tuning":
            pitches = []
            for p in prop.findall("./Pitches/Pitch"):
                pitches.append(
                    _pitch_to_midi(
                        p.findtext("Step") or "E",
                        p.findtext("Accidental") or "",
                        int(p.findtext("Octave") or 2),
                    )
                )
            if len(pitches) == 6:
                tuning = pitches
    return GuitarSetup(tuning=tuning, capo=capo)


def load_gp_notes(
    gp_path: str | Path,
    *,
    track_index: int = 0,
) -> tuple[list[EvalNote], float, GuitarSetup]:
    """Parse a GP7/8 ``.gp`` zip into timed eval notes (voice 0, first track)."""
    path = Path(gp_path)
    root = _read_gpif(path)
    bpm = _header_bpm(root)
    setup = _capo_and_tuning(root, track_index)

    rhythms: dict[str, int] = {}
    for rhythm in root.findall("./Rhythms/Rhythm"):
        rid = rhythm.get("id", "")
        nv = rhythm.findtext("NoteValue") or "Quarter"
        rhythms[rid] = _NOTE_VALUE_TICKS.get(nv, 960)

    notes_by_id: dict[str, tuple[int, int, int]] = {}
    for note in root.findall("./Notes/Note"):
        nid = note.get("id", "")
        fret = int(note.findtext(".//Property[@name='Fret']/Fret") or 0)
        gp_string = int(note.findtext(".//Property[@name='String']/String") or 0)
        # GPIF strings are 0-based low→high (0=low E). AITabs: 0=low E as well.
        aitabs_string = gp_string
        midi_el = note.find(".//Property[@name='Midi']/Number")
        if midi_el is not None and (midi_el.text or "").strip():
            pitch = int(midi_el.text)
        else:
            pitch_el = note.find(".//Property[@name='ConcertPitch']/Pitch")
            if pitch_el is not None:
                pitch = _pitch_to_midi(
                    pitch_el.findtext("Step") or "C",
                    pitch_el.findtext("Accidental") or "",
                    int(pitch_el.findtext("Octave") or 4),
                )
            else:
                pitch = fret_to_midi(aitabs_string, fret, setup=setup)
        notes_by_id[nid] = (pitch, aitabs_string, fret)

    beats_by_id: dict[str, tuple[int, list[str]]] = {}
    for beat in root.findall("./Beats/Beat"):
        bid = beat.get("id", "")
        rhythm_ref = beat.find("Rhythm")
        rid = rhythm_ref.get("ref") if rhythm_ref is not None else "0"
        ticks = rhythms.get(rid, 960)
        note_ids = (beat.findtext("Notes") or "").split()
        beats_by_id[bid] = (ticks, note_ids)

    out: list[EvalNote] = []
    # Each Voice is tied to one bar (see <Bars><Bar><Voices>); reset the clock per bar.
    bar_ticks = 4 * _TICKS_PER_QUARTER  # reference is 4/4 in vidtest1; default for MVP clips
    for voice in sorted(root.findall("./Voices/Voice"), key=lambda v: int(v.get("id", 0))):
        voice_id = int(voice.get("id", 0))
        bar_start_ticks = voice_id * bar_ticks
        elapsed_ticks = bar_start_ticks
        beat_ids = (voice.findtext("Beats") or "").split()
        for bid in beat_ids:
            if bid not in beats_by_id:
                continue
            ticks, note_ids = beats_by_id[bid]
            start_sec = elapsed_ticks * 60.0 / (bpm * _TICKS_PER_QUARTER)
            end_sec = (elapsed_ticks + ticks) * 60.0 / (bpm * _TICKS_PER_QUARTER)
            for nid in note_ids:
                if nid not in notes_by_id:
                    continue
                pitch, string_idx, fret = notes_by_id[nid]
                out.append(
                    EvalNote(
                        start=start_sec,
                        end=end_sec,
                        pitch_midi=pitch,
                        string=string_idx,
                        fret=fret,
                        confidence=1.0,
                    )
                )
            elapsed_ticks += ticks

    out.sort(key=lambda n: (n["start"], n["string"], n["fret"]))
    return out, bpm, setup

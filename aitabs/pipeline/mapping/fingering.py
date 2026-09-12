"""Dynamic programming fingering optimizer — assigns string/fret to each note."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict

from .fingering_config import DEFAULT_FINGERING_CONFIG, FingeringConfig
from .guitar import GuitarSetup, MAX_FRETS, STANDARD_TUNING, midi_to_positions


class TabNote(TypedDict):
    start: float
    end: float
    pitch_midi: int
    string: int
    fret: int
    confidence: float


def transition_cost(
    from_pos: tuple[int, int],
    to_pos: tuple[int, int],
    config: FingeringConfig | None = None,
) -> float:
    cfg = config or DEFAULT_FINGERING_CONFIG
    from_string, from_fret = from_pos
    to_string, to_fret = to_pos
    string_dist = abs(to_string - from_string)
    fret_dist = abs(to_fret - from_fret)
    open_bonus = cfg.open_string_bonus if to_fret == 0 else 0.0
    return open_bonus + cfg.string_cost * float(string_dist) + cfg.fret_cost * float(fret_dist)


def _note_to_tab(note: dict, string_idx: int, fret: int) -> TabNote:
    return TabNote(
        start=float(note["start"]),
        end=float(note["end"]),
        pitch_midi=int(note["pitch_midi"]),
        string=int(string_idx),
        fret=int(fret),
        confidence=float(note["confidence"]),
    )


def _best_single_position(candidates: list[tuple[int, int]]) -> tuple[int, int]:
    """Lowest fret, then lowest string index."""
    return min(candidates, key=lambda p: (p[1], p[0]))


def _guitar_setup_from_config(config: FingeringConfig | None) -> GuitarSetup | None:
    cfg = config or DEFAULT_FINGERING_CONFIG
    tuning = cfg.tuning if cfg.tuning is not None else STANDARD_TUNING
    if cfg.capo == 0 and list(tuning) == STANDARD_TUNING:
        return None
    return GuitarSetup(tuning=tuple(tuning), capo=cfg.capo)


def _map_notes_individually(
    notes: list[dict],
    config: FingeringConfig | None = None,
) -> list[TabNote]:
    """Fallback: map each note on its own (no chord grouping)."""
    setup = _guitar_setup_from_config(config)
    out: list[TabNote] = []
    for note in notes:
        positions = midi_to_positions(int(note["pitch_midi"]), setup=setup)
        if not positions:
            continue
        s, f = _best_single_position(positions)
        out.append(_note_to_tab(note, s, f))
    return out


def _chord_group_epsilon_sec(
    bpm: float,
    ticks_per_beat: int = 4,
    config: FingeringConfig | None = None,
) -> float:
    """Max onset gap (seconds) to treat notes as one chord — one grid step at ``bpm``."""
    cfg = config or DEFAULT_FINGERING_CONFIG
    if bpm <= 0:
        return cfg.default_epsilon_sec
    return 60.0 / bpm / ticks_per_beat


def _resolve_epsilon_sec(
    *,
    bpm: float | None,
    ticks_per_beat: int,
    chord_group_epsilon_sec: float | None,
    config: FingeringConfig,
) -> float:
    if chord_group_epsilon_sec is not None:
        return chord_group_epsilon_sec
    if bpm is not None:
        return _chord_group_epsilon_sec(bpm, ticks_per_beat, config)
    return config.default_epsilon_sec


def map_sequence_greedy(
    notes: list[dict],
    *,
    config: FingeringConfig | None = None,
    bpm: float | None = None,
    ticks_per_beat: int = 4,
    chord_group_epsilon_sec: float | None = None,
) -> list[TabNote]:
    """Per-note lowest-fret mapping (no Viterbi)."""
    _ = (bpm, ticks_per_beat, chord_group_epsilon_sec)
    cfg = config or DEFAULT_FINGERING_CONFIG
    return _map_notes_individually(_mappable_notes(notes, cfg), cfg)


def _mappable_notes(notes: list[dict], config: FingeringConfig | None = None) -> list[dict]:
    setup = _guitar_setup_from_config(config)
    mappable: list[dict] = []
    for note in notes:
        if midi_to_positions(int(note["pitch_midi"]), setup=setup):
            mappable.append(note)
    return mappable


def map_sequence_viterbi(
    notes: list[dict],
    *,
    config: FingeringConfig | None = None,
    bpm: float | None = None,
    ticks_per_beat: int = 4,
    chord_group_epsilon_sec: float | None = None,
) -> list[TabNote]:
    cfg = config or DEFAULT_FINGERING_CONFIG
    setup = _guitar_setup_from_config(cfg)
    if not notes:
        return []

    epsilon_sec = _resolve_epsilon_sec(
        bpm=bpm,
        ticks_per_beat=ticks_per_beat,
        chord_group_epsilon_sec=chord_group_epsilon_sec,
        config=cfg,
    )

    mappable = _mappable_notes(notes, cfg)
    if not mappable:
        return []

    groups: list[list[int]] = [[0]]
    for i in range(1, len(mappable)):
        if abs(float(mappable[i]["start"]) - float(mappable[groups[-1][0]]["start"])) <= epsilon_sec:
            groups[-1].append(i)
        else:
            groups.append([i])

    note_candidates = [midi_to_positions(int(n["pitch_midi"]), setup=setup) for n in mappable]

    @dataclass(frozen=True)
    class _ChordState:
        positions: tuple[tuple[int, int], ...]
        anchor: tuple[int, int]
        chord_cost: float

    def _chord_cost(positions: Sequence[tuple[int, int]]) -> float:
        frets = [f for _, f in positions]
        span = max(frets) - min(frets)
        if span <= cfg.max_fret_span:
            span_penalty = 0.0
        else:
            span_penalty = cfg.span_base_penalty + cfg.span_extra_penalty * float(
                span - cfg.max_fret_span
            )
        low_fret_penalty = cfg.low_fret_penalty * float(sum(frets) / len(frets))
        return span_penalty + low_fret_penalty

    def _anchor(group_idx_list: list[int], positions: Sequence[tuple[int, int]]) -> tuple[int, int]:
        lowest_i = min(group_idx_list, key=lambda idx: int(mappable[idx]["pitch_midi"]))
        j = group_idx_list.index(lowest_i)
        return positions[j]

    def _greedy_group_state(group_idx_list: list[int]) -> _ChordState | None:
        picked: list[tuple[int, int]] = []
        used_strings: set[int] = set()
        for idx in group_idx_list:
            cands = note_candidates[idx]
            if not cands:
                return None
            choice = None
            for s, f in sorted(cands, key=lambda p: (p[1], p[0])):
                if s not in used_strings:
                    choice = (s, f)
                    used_strings.add(s)
                    break
            if choice is None:
                choice = _best_single_position(cands)
            picked.append(choice)
        return _ChordState(tuple(picked), _anchor(group_idx_list, picked), _chord_cost(picked))

    def _enumerate_group_states(group_idx_list: list[int]) -> list[_ChordState]:
        max_states = cfg.max_states_per_chord
        per_note = [note_candidates[i] for i in group_idx_list]
        if any(not cands for cands in per_note):
            fallback = _greedy_group_state(group_idx_list)
            return [fallback] if fallback else []

        out_states: list[_ChordState] = []
        built: list[tuple[int, int]] = []
        used_strings: set[int] = set()

        def dfs(k: int) -> None:
            if len(out_states) >= max_states:
                return
            if k == len(per_note):
                out_states.append(
                    _ChordState(tuple(built), _anchor(group_idx_list, built), _chord_cost(built))
                )
                return
            for s, f in sorted(per_note[k], key=lambda p: (p[1], p[0])):
                if s in used_strings:
                    continue
                used_strings.add(s)
                built.append((s, f))
                dfs(k + 1)
                built.pop()
                used_strings.remove(s)

        dfs(0)

        if not out_states:
            fallback = _greedy_group_state(group_idx_list)
            if fallback:
                out_states = [fallback]

        out_states.sort(key=lambda st: st.chord_cost)
        return out_states[:max_states]

    valid_groups: list[list[int]] = []
    group_states: list[list[_ChordState]] = []
    for g in groups:
        states = _enumerate_group_states(g)
        if states:
            valid_groups.append(g)
            group_states.append(states)

    if not valid_groups:
        return _map_notes_individually(mappable, cfg)

    dp: list[list[float]] = [[st.chord_cost for st in group_states[0]]]
    prev: list[list[int]] = [[-1 for _ in group_states[0]]]

    for gi in range(1, len(valid_groups)):
        cur_dp: list[float] = []
        cur_prev: list[int] = []
        for st in group_states[gi]:
            best_cost = float("inf")
            best_k = -1
            for k, prev_st in enumerate(group_states[gi - 1]):
                cost = (
                    dp[gi - 1][k]
                    + transition_cost(prev_st.anchor, st.anchor, cfg)
                    + st.chord_cost
                )
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            if best_k >= 0:
                cur_dp.append(best_cost)
                cur_prev.append(best_k)
        if not cur_dp:
            return _map_notes_individually(mappable, cfg)
        dp.append(cur_dp)
        prev.append(cur_prev)

    if not dp[-1]:
        return _map_notes_individually(mappable, cfg)

    j = min(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
    chosen: list[_ChordState] = []
    for gi in range(len(valid_groups) - 1, -1, -1):
        chosen.append(group_states[gi][j])
        j = prev[gi][j]
    chosen.reverse()

    out: list[TabNote] = []
    for group_idx_list, st in zip(valid_groups, chosen):
        for note_idx, (string_idx, fret) in zip(group_idx_list, st.positions):
            out.append(_note_to_tab(mappable[note_idx], string_idx, fret))

    out.sort(key=lambda n: (n["start"], n["pitch_midi"]))
    return out


def map_sequence(
    notes: list[dict],
    *,
    bpm: float | None = None,
    ticks_per_beat: int = 4,
    chord_group_epsilon_sec: float | None = None,
    config: FingeringConfig | None = None,
) -> list[TabNote]:
    """Assign string/fret using the configured mapper (default: Viterbi)."""
    cfg = config or DEFAULT_FINGERING_CONFIG
    from .registry import apply_fingering

    return apply_fingering(
        notes,
        cfg,
        bpm=bpm,
        chord_group_epsilon_sec=chord_group_epsilon_sec,
    )

"""Guitar constants and MIDI-to-string/fret lookup."""

from __future__ import annotations

from dataclasses import dataclass

# Standard EADGBE tuning: open-string MIDI note numbers.
# Index 0 = low E (thickest), index 5 = high e (thinnest).
#   E2=40  A2=45  D3=50  G3=55  B3=59  E4=64
STANDARD_TUNING: list[int] = [40, 45, 50, 55, 59, 64]

# Most guitars have 20–24 frets. 22 is the most common.
MAX_FRETS: int = 22

# Human playability: most players can comfortably span 4 frets in one position.
# Above the 12th fret the physical fret spacing shrinks, so 5-fret spans are fine.
MAX_FRET_SPAN: int = 4


@dataclass(frozen=True)
class GuitarSetup:
    """Written tab context: open-string tuning and capo (from GP5 ``track.offset``)."""

    tuning: tuple[int, ...] = tuple(STANDARD_TUNING)
    capo: int = 0

    def __post_init__(self) -> None:
        if len(self.tuning) != 6:
            raise ValueError(f"tuning must have 6 strings, got {len(self.tuning)}")
        if self.capo < 0:
            raise ValueError(f"capo must be >= 0, got {self.capo}")

    @classmethod
    def standard(cls) -> GuitarSetup:
        return cls()

    def effective_tuning(self) -> list[int]:
        """Open-string MIDI at the capo nut — use for mapping heard pitch to written frets."""
        return [open_midi + self.capo for open_midi in self.tuning]

    def written_to_concert_midi(self, string_idx: int, fret: int) -> int:
        """Sounding pitch for a written tab position."""
        return self.tuning[string_idx] + self.capo + fret

    def is_standard(self) -> bool:
        return self.capo == 0 and list(self.tuning) == STANDARD_TUNING


def read_guitar_setup_from_track(track) -> GuitarSetup:
    """Build setup from a PyGuitarPro track (``strings`` + ``offset`` capo)."""
    by_number = {int(s.number): int(s.value) for s in track.strings}
    tuning = tuple(by_number[i] for i in range(6, 0, -1))
    capo = max(int(getattr(track, "offset", 0) or 0), 0)
    return GuitarSetup(tuning=tuning, capo=capo)


def midi_to_positions(
    midi_note: int,
    tuning: list[int] | None = None,
    *,
    setup: GuitarSetup | None = None,
) -> list[tuple[int, int]]:
    """Return all (string_index, written_fret) pairs for a *concert* MIDI note.

    Pass ``setup`` (tuning + capo) so capo/transpose match the reference GP5.
    ``tuning`` alone is treated as the effective open-string pitch (legacy).
    """
    if setup is not None:
        open_strings = setup.effective_tuning()
    else:
        open_strings = tuning if tuning is not None else STANDARD_TUNING

    positions: list[tuple[int, int]] = []
    for string_idx, open_midi in enumerate(open_strings):
        fret = midi_note - open_midi
        if 0 <= fret <= MAX_FRETS:
            positions.append((string_idx, fret))
    return positions


def fret_to_midi(
    string_idx: int,
    fret: int,
    tuning: list[int] | None = None,
    *,
    setup: GuitarSetup | None = None,
) -> int:
    """Convert written string/fret to concert MIDI (includes capo when ``setup`` given)."""
    if setup is not None:
        return setup.written_to_concert_midi(string_idx, fret)
    base = tuning if tuning is not None else STANDARD_TUNING
    return base[string_idx] + fret

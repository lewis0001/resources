"""Pitch -> colour mapping.

The default "boomwhacker" palette follows the real Boomwhacker convention
(C red, D orange, E yellow, F light green, G dark green, A purple, B pink,
with sharps as darker in-between shades). Palettes are plain dicts so themes
can override them, and any unknown name falls back to an HSV wheel.
"""

from __future__ import annotations

import colorsys

_BOOMWHACKER = {
    0:  "#e63946",   # C  red
    1:  "#f0653a",   # C# red-orange
    2:  "#f77f00",   # D  orange
    3:  "#fcaa2e",   # D# orange-yellow
    4:  "#ffd60a",   # E  yellow
    5:  "#a7d129",   # F  yellow-green
    6:  "#5cb838",   # F# light green
    7:  "#2a9d3f",   # G  green
    8:  "#1e7a6e",   # G# teal
    9:  "#7b5ee7",   # A  purple
    10: "#a04fd1",   # A# violet
    11: "#ef5da8",   # B  pink
}

_PASTEL = {pc: colorsys.hsv_to_rgb(pc / 12.0, 0.45, 0.98) for pc in range(12)}


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def pitch_color(midi_pitch: int, palette: str | dict = "boomwhacker"
                ) -> tuple[float, float, float]:
    """Returns an RGB triple in 0..1. Octave subtly shifts brightness so the
    same pitch class reads slightly lighter an octave up."""
    pc = midi_pitch % 12
    if isinstance(palette, dict):
        base = palette.get(str(pc)) or palette.get(pc)
        rgb = _hex_to_rgb(base) if isinstance(base, str) else base
    elif palette == "pastel":
        rgb = _PASTEL[pc]
    elif palette == "boomwhacker":
        rgb = _hex_to_rgb(_BOOMWHACKER[pc])
    else:  # generic hue wheel
        rgb = colorsys.hsv_to_rgb(pc / 12.0, 0.8, 0.95)
    octave_shift = (midi_pitch // 12 - 5) * 0.05
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    v = min(1.0, max(0.25, v + octave_shift))
    return colorsys.hsv_to_rgb(h, s, v)


def instrument_length(midi_pitch: int, lo: float = 0.10, hi: float = 0.34) -> float:
    """Visual length of a tube/paddle: lower pitch -> longer, like real
    boomwhackers. Mapped over the practical melody range C3..C7."""
    frac = (midi_pitch - 48) / (108 - 48)
    frac = min(1.0, max(0.0, frac))
    return hi - (hi - lo) * frac

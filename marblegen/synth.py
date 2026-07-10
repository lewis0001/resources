"""Procedural tuned-percussion synthesis.

Every sound bank works out of the box with zero binary assets: a bank.json
describes partials (frequency ratio, amplitude, decay scale) and the note is
synthesised additively with exponential decays plus a short mallet transient.
If a bank folder also contains real samples, those take priority (see
audio.py).
"""

from __future__ import annotations

import numpy as np


def synth_note(freq: float, dur: float, sr: int, params: dict,
               velocity: float = 1.0) -> np.ndarray:
    """Render one strike. `params` comes from bank.json's "synth" section:
      partials: [[ratio, amp, decay_scale], ...]
      decay_s: base decay time constant of the fundamental
      attack_ms: attack ramp
      strike_noise: 0..1 amount of mallet transient
      strike_tone_hz: brightness of the transient
      inharmonic: 0..1 random detuning of upper partials (glass/music box)
      tremolo_hz / tremolo_depth: vibraphone-style amplitude wobble
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)

    decay_s = float(params.get("decay_s", 1.2))
    partials = params.get("partials", [[1.0, 1.0, 1.0], [4.0, 0.25, 0.4]])
    inharm = float(params.get("inharmonic", 0.0))
    rng = np.random.default_rng(int(freq * 100) % (2**31))

    for ratio, amp, dscale in partials:
        f = freq * float(ratio)
        if inharm > 0 and ratio > 1.01:
            f *= 1.0 + inharm * 0.01 * float(rng.normal())
        if f > sr / 2 * 0.95:
            continue
        tau = max(0.03, decay_s * float(dscale))
        out += float(amp) * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)

    noise_amt = float(params.get("strike_noise", 0.08))
    if noise_amt > 0:
        tone = float(params.get("strike_tone_hz", 2500.0))
        click = rng.normal(0, 1, n) * np.exp(-t / 0.008)
        # tilt the noise toward the strike tone with a one-pole resonator
        y = np.zeros(n)
        r = 0.995
        w = 2 * np.pi * min(tone, sr / 2 * 0.9) / sr
        a1, a2 = 2 * r * np.cos(w), -r * r
        for i in range(2, n if n < int(0.02 * sr) else int(0.02 * sr)):
            y[i] = click[i] + a1 * y[i - 1] + a2 * y[i - 2]
        out += noise_amt * 0.05 * y

    trem_hz = float(params.get("tremolo_hz", 0.0))
    if trem_hz > 0:
        depth = float(params.get("tremolo_depth", 0.3))
        out *= 1.0 - depth * 0.5 * (1 - np.cos(2 * np.pi * trem_hz * t))

    attack = max(1, int(float(params.get("attack_ms", 1.5)) / 1000 * sr))
    out[:attack] *= np.linspace(0, 1, attack)
    fade = max(1, int(0.02 * sr))
    out[-fade:] *= np.linspace(1, 0, fade)

    peak = float(np.max(np.abs(out)) or 1.0)
    return (out / peak) * (0.25 + 0.75 * velocity)

"""Audio rendering: sound banks -> per-hit placement -> stereo mix -> WAV,
then ffmpeg muxes it with the rendered video.

A bank is a folder under banks/<name>/ containing bank.json. If the folder
also has samples (bank.json "samples": {"60": "samples/060.wav", ...}) the
nearest sample is used, resampled to the exact pitch; otherwise the note is
synthesised procedurally from bank.json's "synth" params, so every bank works
with zero binary assets. Adding a bank requires no code changes.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from marblegen.config import REPO_ROOT
from marblegen.notes import pitch_hz
from marblegen.synth import synth_note
from marblegen.trajectory import Trajectory


def banks_dir() -> Path:
    return REPO_ROOT / "banks"


def list_banks() -> list[str]:
    return sorted(p.parent.name for p in banks_dir().glob("*/bank.json"))


class Bank:
    def __init__(self, name: str):
        path = banks_dir() / name / "bank.json"
        if not path.exists():
            raise FileNotFoundError(
                f"bank '{name}' not found; available: {', '.join(list_banks())}")
        self.root = path.parent
        with open(path) as f:
            self.spec = json.load(f)
        self.name = name
        self._sample_cache: dict[int, tuple[np.ndarray, int]] = {}
        self._note_cache: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    def _load_sample(self, pitch: int) -> tuple[np.ndarray, int] | None:
        samples: dict = self.spec.get("samples") or {}
        if not samples:
            return None
        keys = sorted(int(k) for k in samples)
        nearest = min(keys, key=lambda k: abs(k - pitch))
        if nearest not in self._sample_cache:
            with wave.open(str(self.root / samples[str(nearest)]), "rb") as w:
                sr = w.getframerate()
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                if w.getnchannels() == 2:
                    raw = raw.reshape(-1, 2).mean(axis=1)
                self._sample_cache[nearest] = (raw.astype(np.float64) / 32768.0, sr)
        data, sr = self._sample_cache[nearest]
        # repitch by resampling ratio
        ratio = pitch_hz(pitch) / pitch_hz(nearest)
        idx = np.arange(0, len(data), ratio)
        idx = idx[idx < len(data) - 1]
        frac = idx - np.floor(idx)
        base = np.floor(idx).astype(int)
        return data[base] * (1 - frac) + data[base + 1] * frac, sr

    def note(self, pitch: int, sr: int, velocity: float = 1.0) -> np.ndarray:
        key = (pitch, sr)
        if key not in self._note_cache:
            got = self._load_sample(pitch)
            if got is not None:
                data, s_sr = got
                if s_sr != sr:
                    idx = np.arange(0, len(data), s_sr / sr)
                    idx = idx[idx < len(data) - 1]
                    base = np.floor(idx).astype(int)
                    frac = idx - base
                    data = data[base] * (1 - frac) + data[base + 1] * frac
                self._note_cache[key] = data
            else:
                params = self.spec.get("synth", {})
                dur = float(params.get("note_dur_s", 2.2))
                self._note_cache[key] = synth_note(
                    pitch_hz(pitch), dur, sr, params, velocity=1.0)
        return self._note_cache[key] * velocity


# ---------------------------------------------------------------------------


def render_audio(traj: Trajectory, cfg_audio: dict, out_wav: str | Path,
                 x_bounds: tuple[float, float] = (-1.35, 1.35),
                 seed: int = 7) -> Path:
    """Place one bank note per strike at its exact hit time, pan by the
    instrument's x position, mix, soft-limit, normalise, write 16-bit WAV."""
    sr = int(cfg_audio["sample_rate"])
    bank = Bank(cfg_audio["bank"])
    rng = np.random.default_rng(seed)

    video_t0 = float(traj.meta.get("video_t0", traj.segments[0].t0))
    t_end = traj.segments[-1].t1 + float(cfg_audio.get("tail_s", 2.5))
    n = int((t_end - video_t0) * sr) + sr
    mix = np.zeros((n, 2))

    pan_spread = float(cfg_audio.get("pan_spread", 0.6))
    human_db = float(cfg_audio.get("humanize_db", 1.5))
    half_w = max(abs(x_bounds[0]), abs(x_bounds[1]))

    for e in traj.events:
        if e.kind != "note" or not e.pitches:
            continue
        vel = e.velocity / 127.0
        gain = 10 ** (float(rng.normal(0.0, human_db / 3.0)) / 20.0)
        pan = np.clip(e.pos[0] / half_w, -1, 1) * pan_spread
        lg, rg = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        start = int((e.time - video_t0) * sr)
        for k, pitch in enumerate(e.pitches):
            note = bank.note(pitch, sr, velocity=1.0)
            g = vel * gain * (1.0 if k == 0 else 0.8)  # chord voices sit back
            seg = note[: max(0, n - start)]
            if start < 0 or len(seg) == 0:
                continue
            mix[start:start + len(seg), 0] += seg * g * lg
            mix[start:start + len(seg), 1] += seg * g * rg

    # gentle soft-clip then normalise to master gain
    mix = np.tanh(mix * 1.2)
    peak = float(np.max(np.abs(mix)) or 1.0)
    target = 10 ** (float(cfg_audio.get("master_gain_db", -1.0)) / 20.0)
    mix = mix / peak * target

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out_wav

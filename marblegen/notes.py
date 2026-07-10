"""MIDI ingestion: parse any .mid, pick the melody track, reduce to a
monophonic hit list (with chord grouping), and apply musical transforms
(transpose / tempo scale / trim / loop).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mido

from marblegen.config import REPO_ROOT

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(midi_pitch: int) -> str:
    return f"{NOTE_NAMES[midi_pitch % 12]}{midi_pitch // 12 - 1}"


def pitch_hz(midi_pitch: float) -> float:
    return 440.0 * 2.0 ** ((midi_pitch - 69) / 12.0)


@dataclass
class Note:
    time: float          # onset, seconds
    pitch: int           # MIDI pitch
    velocity: int        # 1..127
    duration: float = 0.25


@dataclass
class Hit:
    """One marble impact. Chords collapse into a single hit with several
    pitches sounding at once (rendered as one wide instrument)."""
    time: float
    pitches: list[int] = field(default_factory=list)
    velocity: int = 96


@dataclass
class Song:
    title: str
    notes: list[Note]
    source: str = ""

    @property
    def duration(self) -> float:
        return max((n.time + n.duration) for n in self.notes) if self.notes else 0.0


# ---------------------------------------------------------------------------
# MIDI reading
# ---------------------------------------------------------------------------

def _absolute_notes(mid: mido.MidiFile) -> list[list[Note]]:
    """Return per-track note lists with absolute times in seconds."""
    tracks: list[list[Note]] = []
    default_tempo = 500000

    # Build a global tempo map from all tracks (type 0/1 files keep tempo in track 0).
    tempo_changes: list[tuple[int, int]] = [(0, default_tempo)]  # (abs_ticks, tempo)
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                tempo_changes.append((t, msg.tempo))
    tempo_changes.sort()

    def ticks_to_seconds(ticks: int) -> float:
        sec, last_tick, tempo = 0.0, 0, default_tempo
        for change_tick, change_tempo in tempo_changes:
            if change_tick >= ticks:
                break
            sec += mido.tick2second(change_tick - last_tick, mid.ticks_per_beat, tempo)
            last_tick, tempo = change_tick, change_tempo
        sec += mido.tick2second(ticks - last_tick, mid.ticks_per_beat, tempo)
        return sec

    for track in mid.tracks:
        t = 0
        open_notes: dict[int, tuple[int, int]] = {}  # pitch -> (start_ticks, velocity)
        notes: list[Note] = []
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                open_notes[msg.note] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on"):  # note_on vel=0 == off
                if msg.note in open_notes:
                    start, vel = open_notes.pop(msg.note)
                    notes.append(Note(time=ticks_to_seconds(start),
                                      pitch=msg.note, velocity=vel,
                                      duration=max(0.05, ticks_to_seconds(t) - ticks_to_seconds(start))))
        for pitch, (start, vel) in open_notes.items():  # unterminated notes
            notes.append(Note(time=ticks_to_seconds(start), pitch=pitch,
                              velocity=vel, duration=0.25))
        notes.sort(key=lambda n: (n.time, -n.pitch))
        tracks.append(notes)
    return tracks


def _melody_track_index(tracks: list[list[Note]]) -> int:
    """Heuristic melody pick: favour active tracks with high mean pitch and
    low polyphony. Percussion-channel-only tracks are already just notes here,
    so we score every non-empty track."""
    best, best_score = 0, float("-inf")
    for i, notes in enumerate(tracks):
        if len(notes) < 4:
            continue
        mean_pitch = sum(n.pitch for n in notes) / len(notes)
        onsets = {round(n.time, 3) for n in notes}
        polyphony = len(notes) / max(1, len(onsets))       # 1.0 == monophonic
        density = len(onsets)
        score = mean_pitch + 0.05 * density - 25.0 * (polyphony - 1.0)
        if score > best_score:
            best, best_score = i, score
    return best


def load_midi(path: str | Path, track: int | None = None) -> Song:
    mid = mido.MidiFile(str(path))
    tracks = _absolute_notes(mid)
    if not any(tracks):
        raise ValueError(f"{path}: no notes found in any track")
    idx = track if track is not None else _melody_track_index(tracks)
    if not (0 <= idx < len(tracks)) or not tracks[idx]:
        raise ValueError(f"{path}: track {idx} has no notes "
                         f"(tracks with notes: {[i for i, t in enumerate(tracks) if t]})")
    return Song(title=Path(path).stem.replace("_", " ").title(), notes=tracks[idx])


# ---------------------------------------------------------------------------
# Transforms + reduction to hits
# ---------------------------------------------------------------------------

def transform(song: Song, transpose: int = 0, tempo_scale: float = 1.0,
              trim: tuple[float, float] | None = None,
              loop: int = 1) -> Song:
    notes = [Note(n.time, n.pitch, n.velocity, n.duration) for n in song.notes]
    if transpose:
        for n in notes:
            n.pitch = min(127, max(0, n.pitch + transpose))
    if tempo_scale != 1.0:
        if tempo_scale <= 0:
            raise ValueError("tempo_scale must be > 0")
        for n in notes:
            n.time /= tempo_scale
            n.duration /= tempo_scale
    if trim is not None:
        a, b = trim
        notes = [n for n in notes if a <= n.time < b]
        t0 = min((n.time for n in notes), default=0.0)
        for n in notes:
            n.time -= t0
    if loop > 1 and notes:
        span = max(n.time + n.duration for n in notes) + 0.4
        base = list(notes)
        for k in range(1, loop):
            notes += [Note(n.time + k * span, n.pitch, n.velocity, n.duration)
                      for n in base]
    notes.sort(key=lambda n: (n.time, -n.pitch))
    if not notes:
        raise ValueError("no notes left after transforms (check --trim range)")
    return Song(title=song.title, notes=notes, source=song.source)


def to_hits(song: Song, chord_epsilon: float = 0.012) -> list[Hit]:
    """Monophonic reduction with chord grouping: notes whose onsets are within
    chord_epsilon collapse into ONE hit sounding all pitches (one wide
    instrument, struck once)."""
    hits: list[Hit] = []
    for n in song.notes:
        if hits and (n.time - hits[-1].time) <= chord_epsilon:
            if n.pitch not in hits[-1].pitches:
                hits[-1].pitches.append(n.pitch)
            hits[-1].velocity = max(hits[-1].velocity, n.velocity)
        else:
            hits.append(Hit(time=n.time, pitches=[n.pitch], velocity=n.velocity))
    for h in hits:
        h.pitches.sort(reverse=True)  # top note first
    # Guarantee strictly increasing times (defensive vs. weird files).
    out: list[Hit] = []
    for h in hits:
        if out and h.time <= out[-1].time:
            continue
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# Multi-marble voice splitting
# ---------------------------------------------------------------------------

def auto_marbles(hits: list[Hit], max_marbles: int = 3) -> int:
    """Pick a marble count from note density: busy songs read as chaos with a
    single marble, so dense passages get split across 2-3 marbles."""
    if len(hits) < 8:
        return 1
    gaps = sorted(hits[i + 1].time - hits[i].time for i in range(len(hits) - 1))
    median_gap = gaps[len(gaps) // 2]
    if median_gap >= 0.22:
        return 1
    if median_gap >= 0.11:
        return min(2, max_marbles)
    return min(3, max_marbles)


def split_voices(hits: list[Hit], n_marbles: int) -> list[list[Hit]]:
    """Deal the hit stream across n marbles: each hit goes to the marble that
    has rested longest (strict alternation for a monophonic line). Every
    marble's own note gaps become ~n times longer, so each track stays a calm,
    readable bounce pattern while together they play the full song."""
    if n_marbles <= 1:
        return [hits]
    last = [float("-inf")] * n_marbles
    voices: list[list[Hit]] = [[] for _ in range(n_marbles)]
    for h in hits:
        i = min(range(n_marbles), key=lambda k: last[k])
        voices[i].append(h)
        last[i] = h.time
    return [v for v in voices if v]


# ---------------------------------------------------------------------------
# Song library
# ---------------------------------------------------------------------------

def songs_dir() -> Path:
    return REPO_ROOT / "songs"


def resolve_song(name_or_path: str) -> tuple[Path, dict]:
    """Accept a bundled song name ('ode_to_joy') or a path to any .mid file.
    Returns (midi_path, manifest_dict)."""
    p = Path(name_or_path)
    if p.suffix.lower() in (".mid", ".midi") and p.exists():
        manifest_path = p.with_suffix(".json")
    else:
        stem = p.stem
        p = songs_dir() / f"{stem}.mid"
        manifest_path = songs_dir() / f"{stem}.json"
        if not p.exists():
            available = sorted(s.stem for s in songs_dir().glob("*.mid"))
            raise FileNotFoundError(
                f"song '{name_or_path}' not found; bundled songs: {', '.join(available)}")
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    return p, manifest

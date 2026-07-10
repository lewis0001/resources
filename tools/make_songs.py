#!/usr/bin/env python3
"""Regenerate the bundled public-domain song library (songs/*.mid + *.json).

Melodies are encoded as compact token strings:  "<Pitch><Octave>:<dur>"
with durations w/h/q/e/s (whole..sixteenth), an optional trailing '.' for
dotted values, and R for rests. All bundled melodies are traditional tunes or
works by composers dead well over a century — public domain worldwide.

Run:  python3 tools/make_songs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DUR = {"w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25}


def parse_token(tok: str) -> tuple[int | None, float]:
    name, dur = tok.split(":")
    beats = DUR[dur[0]] * (1.5 if dur.endswith(".") else 1.0)
    if name == "R":
        return None, beats
    step = PITCH[name[0]]
    rest = name[1:]
    if rest.startswith("#"):
        step += 1
        rest = rest[1:]
    elif rest.startswith("b"):
        step -= 1
        rest = rest[1:]
    octave = int(rest)
    return 12 * (octave + 1) + step, beats


def write_midi(path: Path, notes_str: str, bpm: float) -> int:
    ticks = 480
    mid = mido.MidiFile(ticks_per_beat=ticks)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    pending_gap = 0
    count = 0
    for tok in notes_str.split():
        pitch, beats = parse_token(tok)
        length = int(beats * ticks)
        if pitch is None:
            pending_gap += length
            continue
        gate = int(length * 0.88)
        tr.append(mido.Message("note_on", note=pitch, velocity=100,
                               time=pending_gap))
        tr.append(mido.Message("note_off", note=pitch, velocity=0, time=gate))
        pending_gap = length - gate
        count += 1
    tr.append(mido.MetaMessage("end_of_track", time=pending_gap))
    mid.save(str(path))
    return count


SONGS: list[dict] = [
    dict(stem="ode_to_joy", title="Ode to Joy", composer="Ludwig van Beethoven",
         bpm=120, theme="boomwhacker_wall", bank="boomwhacker", notes="""
         E4:q E4:q F4:q G4:q G4:q F4:q E4:q D4:q C4:q C4:q D4:q E4:q E4:q. D4:e D4:h
         E4:q E4:q F4:q G4:q G4:q F4:q E4:q D4:q C4:q C4:q D4:q E4:q D4:q. C4:e C4:h
         D4:q D4:q E4:q C4:q D4:q E4:e F4:e E4:q C4:q D4:q E4:e F4:e E4:q D4:q C4:q D4:q G3:h
         E4:q E4:q F4:q G4:q G4:q F4:q E4:q D4:q C4:q C4:q D4:q E4:q D4:q. C4:e C4:h"""),
    dict(stem="twinkle_twinkle", title="Twinkle Twinkle Little Star",
         composer="Traditional (French)", bpm=104, theme="floating_paddles",
         bank="music_box", notes="""
         C4:q C4:q G4:q G4:q A4:q A4:q G4:h F4:q F4:q E4:q E4:q D4:q D4:q C4:h
         G4:q G4:q F4:q F4:q E4:q E4:q D4:h G4:q G4:q F4:q F4:q E4:q E4:q D4:h
         C4:q C4:q G4:q G4:q A4:q A4:q G4:h F4:q F4:q E4:q E4:q D4:q D4:q C4:h"""),
    dict(stem="mary_had_a_little_lamb", title="Mary Had a Little Lamb",
         composer="Traditional (American)", bpm=116, theme="boomwhacker_wall",
         bank="marimba", notes="""
         E4:q D4:q C4:q D4:q E4:q E4:q E4:h D4:q D4:q D4:h E4:q G4:q G4:h
         E4:q D4:q C4:q D4:q E4:q E4:q E4:q E4:q D4:q D4:q E4:q D4:q C4:w"""),
    dict(stem="jingle_bells", title="Jingle Bells", composer="James Lord Pierpont",
         bpm=132, theme="boomwhacker_wall", bank="boomwhacker", notes="""
         E4:q E4:q E4:h E4:q E4:q E4:h E4:q G4:q C4:q. D4:e E4:w
         F4:q F4:q F4:q. F4:e F4:q E4:q E4:q E4:e E4:e E4:q D4:q D4:q E4:q D4:h G4:h
         E4:q E4:q E4:h E4:q E4:q E4:h E4:q G4:q C4:q. D4:e E4:w
         F4:q F4:q F4:q. F4:e F4:q E4:q E4:q E4:e E4:e G4:q G4:q F4:q D4:q C4:w"""),
    dict(stem="happy_birthday", title="Happy Birthday to You",
         composer="Patty & Mildred Hill (PD since 2016)", bpm=120,
         theme="floating_paddles", bank="kalimba", notes="""
         C4:e. C4:s D4:q C4:q F4:q E4:h C4:e. C4:s D4:q C4:q G4:q F4:h
         C4:e. C4:s C5:q A4:q F4:q E4:q D4:q Bb4:e. Bb4:s A4:q F4:q G4:q F4:h"""),
    dict(stem="fur_elise", title="Für Elise", composer="Ludwig van Beethoven",
         bpm=112, theme="floating_paddles", bank="music_box", notes="""
         E5:e Eb5:e E5:e Eb5:e E5:e B4:e D5:e C5:e A4:q R:e C4:e E4:e A4:e B4:q
         R:e E4:e Ab4:e B4:e C5:q R:e E4:e E5:e Eb5:e E5:e Eb5:e E5:e B4:e D5:e C5:e
         A4:q R:e C4:e E4:e A4:e B4:q R:e E4:e C5:e B4:e A4:h"""),
    dict(stem="canon_in_d", title="Canon in D", composer="Johann Pachelbel",
         bpm=64, theme="floating_paddles", bank="glass", notes="""
         F#5:q E5:q D5:q C#5:q B4:q A4:q B4:q C#5:q
         D5:q C#5:q B4:q A4:q G4:q F#4:q G4:q E4:q
         D4:e F#4:e A4:e G4:e F#4:e D4:e F#4:e E4:e D4:e B3:e D4:e A4:e G4:e B4:e A4:e G4:e
         F#4:q D5:q C#5:q B4:q A4:h D5:h"""),
    dict(stem="mountain_king", title="In the Hall of the Mountain King",
         composer="Edvard Grieg", bpm=168, theme="boomwhacker_wall",
         bank="marimba", notes="""
         E4:e F#4:e G4:e A4:e B4:e G4:e B4:q Bb4:e Gb4:e Bb4:q A4:e F4:e A4:q
         E4:e F#4:e G4:e A4:e B4:e G4:e B4:e E5:e D5:e B4:e G4:e B4:e D5:h
         E4:e F#4:e G4:e A4:e B4:e G4:e B4:q Bb4:e Gb4:e Bb4:q A4:e F4:e A4:q
         E4:e F#4:e G4:e A4:e B4:e G4:e B4:e E5:e D5:e B4:e G4:e B4:e E5:h"""),
    dict(stem="frere_jacques", title="Frère Jacques", composer="Traditional (French)",
         bpm=112, theme="floating_paddles", bank="kalimba", notes="""
         C4:q D4:q E4:q C4:q C4:q D4:q E4:q C4:q E4:q F4:q G4:h E4:q F4:q G4:h
         G4:e A4:e G4:e F4:e E4:q C4:q G4:e A4:e G4:e F4:e E4:q C4:q
         C4:q G3:q C4:h C4:q G3:q C4:h"""),
    dict(stem="row_your_boat", title="Row, Row, Row Your Boat",
         composer="Traditional (American)", bpm=100, theme="floating_paddles",
         bank="kalimba", notes="""
         C4:q. C4:q. C4:q D4:e E4:q. E4:q D4:e E4:q F4:e G4:h.
         C5:e C5:e C5:e G4:e G4:e G4:e E4:e E4:e E4:e C4:e C4:e C4:e
         G4:q F4:e E4:q D4:e C4:h."""),
    dict(stem="london_bridge", title="London Bridge Is Falling Down",
         composer="Traditional (English)", bpm=112, theme="boomwhacker_wall",
         bank="boomwhacker", notes="""
         G4:q. A4:e G4:q F4:q E4:q F4:q G4:h D4:q E4:q F4:h E4:q F4:q G4:h
         G4:q. A4:e G4:q F4:q E4:q F4:q G4:h D4:h G4:h E4:q C4:h."""),
    dict(stem="old_macdonald", title="Old MacDonald Had a Farm",
         composer="Traditional (American)", bpm=116, theme="boomwhacker_wall",
         bank="marimba", notes="""
         G4:q G4:q G4:q D4:q E4:q E4:q D4:h B4:q B4:q A4:q A4:q G4:h. D4:q
         G4:q G4:q G4:q D4:q E4:q E4:q D4:h B4:q B4:q A4:q A4:q G4:h."""),
    dict(stem="yankee_doodle", title="Yankee Doodle", composer="Traditional (American)",
         bpm=120, theme="boomwhacker_wall", bank="boomwhacker", notes="""
         G4:q G4:q A4:q B4:q G4:q B4:q A4:q D4:q G4:q G4:q A4:q B4:q G4:h F#4:h
         G4:q G4:q A4:q B4:q C5:q B4:q A4:q G4:q F#4:q D4:q E4:q F#4:q G4:h G4:h"""),
    dict(stem="amazing_grace", title="Amazing Grace", composer="Traditional (American)",
         bpm=90, theme="floating_paddles", bank="glass", notes="""
         C4:q F4:h A4:e F4:e A4:h G4:q F4:h D4:q C4:h
         C4:q F4:h A4:e F4:e A4:h G4:q C5:h. A4:q C5:h A4:e C5:e A4:h F4:q
         C4:h D4:q F4:h F4:e D4:e C4:h C4:q F4:h A4:e F4:e A4:h G4:q F4:h."""),
    dict(stem="greensleeves", title="Greensleeves", composer="Traditional (English)",
         bpm=100, theme="floating_paddles", bank="music_box", notes="""
         A4:q C5:h D5:q E5:q. F5:e E5:q D5:h B4:q G4:q. A4:e B4:q
         C5:h A4:q A4:q. G#4:e A4:q B4:h G#4:q E4:h A4:q C5:h D5:q E5:q. F5:e E5:q
         D5:h B4:q G4:q. A4:e B4:q C5:q. B4:e A4:q G#4:q. F#4:e G#4:q A4:h."""),
    dict(stem="scarborough_fair", title="Scarborough Fair",
         composer="Traditional (English)", bpm=96, theme="floating_paddles",
         bank="glass", notes="""
         D4:q D4:q A4:h A4:q E4:e F4:e E4:q D4:h. R:q A4:q C5:q D5:h D5:q C5:q A4:q
         B4:q G4:h. R:q A4:q A4:q G4:q F4:e E4:e D4:q C4:q A3:q D4:q D4:q C4:q D4:h."""),
    dict(stem="auld_lang_syne", title="Auld Lang Syne", composer="Traditional (Scottish)",
         bpm=100, theme="floating_paddles", bank="vibraphone", notes="""
         C4:q F4:q. F4:e F4:q A4:q G4:q. F4:e G4:q A4:e G4:e F4:q. F4:e A4:q C5:q D5:h.
         D5:q C5:q. A4:e A4:q F4:q G4:q. F4:e G4:q A4:e G4:e F4:q. D4:e D4:q C4:q F4:h."""),
    dict(stem="oh_susanna", title="Oh! Susanna", composer="Stephen Foster",
         bpm=126, theme="boomwhacker_wall", bank="boomwhacker", notes="""
         C4:e D4:e E4:q G4:q G4:q. A4:e G4:q E4:q C4:q. D4:e E4:q E4:q D4:q C4:q D4:h
         C4:e D4:e E4:q G4:q G4:q. A4:e G4:q E4:q C4:q. D4:e E4:q E4:q D4:q D4:q C4:h
         F4:h F4:q F4:q A4:q A4:h G4:q G4:q E4:q C4:q D4:h
         C4:e D4:e E4:q G4:q G4:q. A4:e G4:q E4:q C4:q. D4:e E4:q E4:q D4:q D4:q C4:h"""),
    dict(stem="camptown_races", title="Camptown Races", composer="Stephen Foster",
         bpm=120, theme="boomwhacker_wall", bank="marimba", notes="""
         G4:q G4:q E4:q G4:q A4:q G4:q E4:h E4:q D4:h E4:q D4:h
         G4:q G4:q E4:q G4:q A4:q G4:q E4:q. D4:e D4:q C4:h.
         C4:e D4:e E4:q G4:h A4:e A4:e G4:q E4:h G4:q. A4:e G4:e E4:e D4:q C4:h"""),
    dict(stem="saints_go_marching", title="When the Saints Go Marching In",
         composer="Traditional (American)", bpm=132, theme="boomwhacker_wall",
         bank="boomwhacker", notes="""
         C4:q E4:q F4:q G4:h. R:q C4:q E4:q F4:q G4:h. R:q
         C4:q E4:q F4:q G4:q E4:q C4:q E4:q D4:h. R:q
         E4:q E4:q D4:q C4:q. C4:e E4:q G4:q G4:q F4:q E4:q F4:q G4:q E4:q C4:q D4:q C4:h"""),
    dict(stem="the_entertainer", title="The Entertainer", composer="Scott Joplin",
         bpm=96, theme="boomwhacker_wall", bank="marimba", notes="""
         D4:s Eb4:s E4:s C5:e. E4:s C5:e. E4:s C5:q. C5:s D5:s Eb5:s
         E5:s C5:s D5:s E5:e B4:s D5:e C5:q.
         D4:s Eb4:s E4:s C5:e. E4:s C5:e. E4:s C5:q. A4:s G4:s F#4:s
         A4:s C5:s E5:e D5:s C5:s A4:s D5:q."""),
    dict(stem="minuet_in_g", title="Minuet in G major", composer="Christian Petzold",
         bpm=116, theme="floating_paddles", bank="glass", notes="""
         D5:q G4:e A4:e B4:e C5:e D5:q G4:q G4:q E5:q C5:e D5:e E5:e F#5:e G5:q G4:q G4:q
         C5:q D5:e C5:e B4:e A4:e B4:q C5:e B4:e A4:e G4:e F#4:q G4:e A4:e B4:e G4:e B4:q A4:h."""),
    dict(stem="korobeiniki", title="Korobeiniki (Tetris theme)",
         composer="Traditional (Russian)", bpm=140, theme="boomwhacker_wall",
         bank="marimba", notes="""
         E5:q B4:e C5:e D5:q C5:e B4:e A4:q A4:e C5:e E5:q D5:e C5:e B4:q. C5:e D5:q E5:q
         C5:q A4:q A4:h R:e D5:e F5:q A5:q G5:e F5:e E5:q. C5:e E5:q D5:e C5:e
         B4:q B4:e C5:e D5:q E5:q C5:q A4:q A4:h"""),
    dict(stem="flight_of_the_bumblebee", title="Flight of the Bumblebee",
         composer="Nikolai Rimsky-Korsakov", bpm=112, theme="boomwhacker_wall",
         bank="marimba", notes="""
         E5:s Eb5:s D5:s C#5:s C5:s B4:s Bb4:s A4:s Ab4:s A4:s Bb4:s A4:s Ab4:s G4:s F#4:s G4:s
         Ab4:s G4:s F#4:s F4:s E4:s Eb4:s D4:s Eb4:s E4:s F4:s F#4:s G4:s Ab4:s A4:s Bb4:s B4:s
         C5:s C#5:s D5:s Eb5:s E5:s Eb5:s D5:s C#5:s C5:s B4:s Bb4:s A4:s Ab4:s G4:s F#4:s F4:s E4:q"""),
    dict(stem="vivaldi_spring", title="Spring (The Four Seasons)",
         composer="Antonio Vivaldi", bpm=112, theme="floating_paddles",
         bank="vibraphone", notes="""
         E4:q G#4:e G#4:e G#4:e F#4:e E4:q B4:q. B4:e G#4:e F#4:e
         E4:q G#4:e G#4:e G#4:e F#4:e E4:q B4:q. B4:e G#4:e F#4:e
         B4:q B4:e A4:e G#4:e F#4:e B3:h."""),
    dict(stem="beethoven_fifth", title="Symphony No. 5 (opening)",
         composer="Ludwig van Beethoven", bpm=108, theme="floating_paddles",
         bank="marimba", notes="""
         R:e G4:e G4:e G4:e Eb4:h. R:e F4:e F4:e F4:e D4:h.
         R:e G4:e G4:e G4:e Eb4:q. R:e Ab4:e Ab4:e Ab4:e G4:q.
         R:e Eb5:e Eb5:e Eb5:e C5:h."""),
    dict(stem="brahms_lullaby", title="Lullaby (Wiegenlied)", composer="Johannes Brahms",
         bpm=104, theme="floating_paddles", bank="music_box", notes="""
         E4:e E4:e G4:q. E4:e E4:e G4:q. E4:e G4:e C5:q B4:q A4:h A4:q G4:q
         D4:e E4:e F4:q D4:q D4:e E4:e F4:q. R:e D4:e F4:e B4:e A4:e G4:q B4:q C5:h."""),
]


def main() -> None:
    out = ROOT / "songs"
    out.mkdir(exist_ok=True)
    for s in SONGS:
        n = write_midi(out / f"{s['stem']}.mid", s["notes"], s["bpm"])
        manifest = {
            "title": s["title"],
            "composer": s["composer"],
            "license": "public domain",
            "bpm": s["bpm"],
            "note_count": n,
            "suggested": {"theme": s["theme"], "bank": s["bank"]},
        }
        with open(out / f"{s['stem']}.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  {s['stem']:28s} {n:4d} notes")
    print(f"{len(SONGS)} songs written to {out}")


if __name__ == "__main__":
    main()

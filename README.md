# Marble Music (`marblegen`)

Automated marble-run music video generator. Give it a song as MIDI and it
produces a finished vertical video (1080×1920, 60 fps, MP4 with audio) of a
marble bouncing and rolling through tuned percussion instruments —
boomwhacker tubes, kalimba paddles — striking each one exactly on the song's
note timings.

```
marblegen render --song ode_to_joy --theme boomwhacker_wall
# -> out/ode_to_joy__boomwhacker_wall.mp4
```

No screen recording, no manual animation: the video is solved and rendered
headlessly, one command from `song.mid` to `output.mp4`.

## Setup

```bash
pip install -e .          # installs numpy, matplotlib, mido, imageio-ffmpeg
```

That is everything the default pipeline needs — `imageio-ffmpeg` ships an
ffmpeg binary, and the built-in **preview engine** renders real videos with
matplotlib. For the fancier 3D look, install [Blender](https://www.blender.org)
(3.6+ or 4.x) and render with `--engine blender` (set `$BLENDER` if it is not
on your PATH).

```bash
marblegen songs      # list the 27 bundled public-domain songs
marblegen banks      # list the 6 sound banks
marblegen themes     # list visual themes
marblegen preview --song korobeiniki            # fast half-res iteration
marblegen render  --song fur_elise --theme floating_paddles --bank music_box
marblegen render  --song mysong.mid --transpose -2 --trim 0:30 --caption "Name this song 🤯"
marblegen batch   jobs.example.json             # queue of videos
marblegen validate                              # physics-check every song × theme
```

## The physics model (why the motion looks right)

The single quality bar that matters in these videos is that the marble moves
like a real ball — never speeding up or slowing down for no reason, never
hovering, never gaining energy from a bounce. `marblegen` enforces this **by
construction**, not by cleanup:

* **One gravity constant** (default 9.81 m/s²) for the entire video. It is
  never varied per segment to make timing work.
* **Forward-solving, not point-forcing.** The solver never picks instrument
  positions first and bends the ball toward them. At every contact the
  departure speed is fixed by physics — `|v_out| = e·|v_in|` with restitution
  `e ≤ 0.88`, so a bounce can only *lose* energy — and the only free
  parameter is the departure **direction**, which is the same thing as the
  orientation of the struck instrument (its face normal is the bisector of
  the in/out directions, exactly like a real bounce). The ball then flies a
  pure ballistic parabola for exactly the time until the next note, and
  *wherever it truly is at that instant* becomes the next instrument's
  position. Consistent motion is therefore guaranteed: the layout comes from
  the physics, never the other way around.
* **Awkward timing gets physical fallbacks, not cheats:**
  * *Long gaps* → 1–3 silent untuned pegs the ball bounces across.
  * *Fast runs* (note gaps under ~0.18 s) → **roll mode**: the ball lands on
    a ramp of instruments and rolls across them, accelerating at
    `5/7 · g · sin θ` (a solid sphere rolling), zig-zagging with a
    restitution bounce when it reaches the edge of the frame. Landing on the
    ramp keeps the tangential velocity component and absorbs the normal one —
    the physics of a real landing.
  * *Chords* → one wide instrument struck once, sounding every pitch.
* **Automatic validation** (`marblegen validate`) samples the final
  closed-form trajectory and fails loudly if acceleration ever deviates from
  gravity, any bounce gains energy, any velocity change has no contact event
  to explain it, a hit misses its MIDI onset by more than 5 ms, the ball
  leaves the frame, or two instruments interpenetrate. `render` refuses to
  produce a video from a failing solve (override with `--force`).

Because the renderers sample the exact same closed-form segments the
validator checks, what you watch **is** the verified physics.

## Music options

Music is a pipeline, not a playlist:

* **Any MIDI file**: `--song path/to/file.mid`. Multi-track files get
  automatic melody-track detection (override with `--track N`); polyphony is
  reduced to the melody line, with simultaneous notes grouped into chords.
* **27 bundled public-domain songs** in `songs/` — Ode to Joy, Für Elise,
  Canon in D, In the Hall of the Mountain King, Korobeiniki (the Tetris
  theme), The Entertainer, Flight of the Bumblebee, plus folk and nursery
  classics. Each has a JSON manifest (title, composer, suggested theme/bank).
  Drop a new `.mid` in `songs/` and it is available by name — no code
  changes. Regenerate the library with `python3 tools/make_songs.py`.
* **Musical controls**: `--transpose` (semitones), `--tempo-scale`,
  `--trim start:end` (seconds, for 15–60 s clips), `--loop N`.
* **Six sound banks** in `banks/` — boomwhacker, marimba, kalimba,
  music_box, glass, vibraphone. Every bank works with zero binary assets:
  `bank.json` describes partials and decay and the note is synthesised
  procedurally. Put real WAVs in the folder and list them under `"samples"`
  and they are used instead (nearest sample, repitched). Adding a bank is
  adding a folder — no code changes.
* Pitch drives the visuals too: instrument colour follows a configurable
  pitch→colour palette (the default matches real boomwhacker colours) and
  tube/paddle length scales with pitch.

A note on **licensing**: the bundled songs are public-domain *compositions*.
If you feed in MIDI of a copyrighted song, the rendered audio embeds that
composition — on platforms like TikTok prefer their in-app licensed sound
library and upload the video muted, or stick to public-domain/licensed music.

## Themes

Two themes ship, sharing the same solver:

* `boomwhacker_wall` — coloured tubes on thin metal brackets against a tiled
  wall (glissando-friendly).
* `floating_paddles` — kalimba-like paddles on thin posts in a bright void.

A theme is a JSON file in `themes/`: instrument shape, colours, wobble
animation parameters, background, plus optional `solver`/`audio` overrides.
Copy one, tweak it, and pass `--theme mytheme`.

## Rendering engines

* `--engine preview` (default): matplotlib → H.264. Real output, fast, zero
  extra dependencies. `marblegen preview` is the same engine at half
  resolution for iteration.
* `--engine blender`: exports the solved trajectory and drives
  `marblegen/blender/build_scene.py` inside headless Blender (Eevee) — 3D
  tubes/paddles, tiled wall, soft studio lighting, motion blur, per-hit
  wobble keyframes. The CLI muxes the audio either way.

Camera work is computed once and shared by both engines: a critically damped
spring follows the ball (no jitter), keeping it in the upper third of frame.

## Batch mode

`marblegen batch jobs.json` renders a queue and keeps going on failures:

```json
[
  {"song": "ode_to_joy",  "theme": "boomwhacker_wall", "caption": "Name this song 🤯"},
  {"song": "korobeiniki", "theme": "floating_paddles", "bank": "music_box", "seed": 3},
  {"song": "custom/riff.mid", "trim": "0:30", "transpose": 2}
]
```

Every job accepts the same fields as the CLI flags. Output lands in `out/`
(or per-job `"out"`). Runs are **deterministic**: the same song, theme and
seed always produce the identical video, and `--seed` varies the layout.

## Repository layout

```
marblegen/           the package
  notes.py           MIDI ingestion, melody extraction, transforms
  solver.py          forward physics solver (bounce search, rolls, pegs)
  trajectory.py      closed-form segments + events, JSON export
  validate.py        physics/layout validation report
  audio.py synth.py  sound banks, procedural percussion, mixing
  preview.py         matplotlib render engine
  blender/build_scene.py   headless Blender engine
  cli.py             render / preview / solve / batch / validate
songs/               bundled public-domain MIDI + manifests
banks/               sound bank definitions
themes/              visual theme definitions
tools/make_songs.py  regenerates songs/ from encoded melodies
```

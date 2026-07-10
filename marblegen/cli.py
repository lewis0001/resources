"""marblegen command line.

  marblegen render  --song ode_to_joy --theme boomwhacker_wall
  marblegen preview --song korobeiniki
  marblegen batch   jobs.json
  marblegen solve   --song fur_elise -o out/fur_elise.traj.json
  marblegen validate --all
  marblegen songs | marblegen banks | marblegen themes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from marblegen.config import REPO_ROOT, build_config, load_theme
from marblegen.notes import load_midi, resolve_song, to_hits, transform
from marblegen.solver import solve
from marblegen.validate import format_report, validate


# ---------------------------------------------------------------------------
# shared pipeline
# ---------------------------------------------------------------------------

def _parse_trim(s: str | None):
    if not s:
        return None
    a, b = s.split(":")
    return (float(a or 0), float(b) if b else 1e9)


def _prepare(args):
    """song + theme + flags -> (validated trajectory, cfg, theme, names)."""
    midi_path, manifest = resolve_song(args.song)
    theme_name = args.theme or manifest.get("suggested", {}).get(
        "theme", "boomwhacker_wall")
    theme = load_theme(theme_name)

    cli_overrides: dict = {"solver": {}, "audio": {}, "render": {}}
    if args.seed is not None:
        cli_overrides["solver"]["seed"] = args.seed
    if getattr(args, "bank", None):
        cli_overrides["audio"]["bank"] = args.bank
    if getattr(args, "caption", None):
        cli_overrides["render"]["caption"] = args.caption

    manifest_overrides = {k: v for k, v in manifest.items()
                          if k in ("solver", "audio", "render", "palette")}
    suggested_bank = manifest.get("suggested", {}).get("bank")
    if suggested_bank and not getattr(args, "bank", None):
        manifest_overrides.setdefault("audio", {})["bank"] = suggested_bank

    cfg = build_config(theme, manifest_overrides, cli_overrides)

    song = load_midi(midi_path, track=args.track)
    if manifest.get("title"):
        song.title = manifest["title"]
    song = transform(song, transpose=args.transpose,
                     tempo_scale=args.tempo_scale,
                     trim=_parse_trim(args.trim), loop=args.loop)
    hits = to_hits(song, cfg["solver"]["chord_epsilon"])

    traj = solve(hits, cfg["solver"],
                 meta={"song": song.title, "theme": theme["name"],
                       "bank": cfg["audio"]["bank"],
                       "seed": cfg["solver"]["seed"]})
    report = validate(traj, hits, cfg["solver"])
    print(format_report(f"{Path(midi_path).stem} / {theme['name']}", report))
    if not report["ok"] and not args.force:
        print("validation FAILED — refusing to render (use --force to override)",
              file=sys.stderr)
        sys.exit(2)
    traj.meta["video_t0"] = traj.time_range()[0] - float(cfg["render"]["lead_in_s"])
    return traj, cfg, theme, Path(midi_path).stem, hits


def _default_out(stem: str, theme: dict, suffix: str) -> Path:
    return REPO_ROOT / "out" / f"{stem}__{theme['name']}{suffix}"


def _render_video(traj, cfg, theme, stem, engine: str, scale: float,
                  out_path: Path) -> Path:
    from marblegen.audio import render_audio
    from marblegen.ffmpeg import mux

    with tempfile.TemporaryDirectory(prefix="marblegen_") as td:
        wav = Path(td) / "audio.wav"
        render_audio(traj, cfg["audio"], wav,
                     x_bounds=tuple(cfg["solver"]["x_bounds"]),
                     seed=int(cfg["solver"]["seed"]))
        silent = Path(td) / "video.mp4"
        if engine == "preview":
            from marblegen.preview import render_preview
            def progress(f, n):
                print(f"  frame {f}/{n} ({100 * f // max(1, n)}%)", end="\r")
            render_preview(traj, theme, cfg, silent, scale=scale,
                           progress=progress)
            print()
        elif engine == "blender":
            _run_blender(traj, cfg, theme, Path(td), silent)
        else:
            raise SystemExit(f"unknown engine '{engine}'")
        return mux(silent, wav, out_path)


def _run_blender(traj, cfg, theme, tmpdir: Path, out_mp4: Path) -> None:
    blender = os.environ.get("BLENDER") or shutil.which("blender")
    if not blender:
        raise SystemExit(
            "Blender not found on PATH (set $BLENDER or install it).\n"
            "Tip: `marblegen render --engine preview` renders without Blender.")
    from marblegen.palette import instrument_length, pitch_color

    traj_json = tmpdir / "traj.json"
    traj.to_json(traj_json, cfg["render"])
    palette = cfg.get("palette", "boomwhacker")
    visuals = []
    for e in traj.events:
        if e.kind == "note":
            visuals.append({
                "color": list(pitch_color(max(e.pitches), palette)),
                "length": instrument_length(max(e.pitches))
                + 0.05 * (len(e.pitches) - 1)})
        else:
            visuals.append(None)
    scene_cfg = tmpdir / "scene.json"
    with open(scene_cfg, "w") as f:
        json.dump({"theme": theme, "render": cfg["render"],
                   "instrument_visuals": visuals}, f)
    script = REPO_ROOT / "marblegen" / "blender" / "build_scene.py"
    cmd = [blender, "--background", "--factory-startup", "--python", str(script),
           "--", "--traj", str(traj_json), "--scene", str(scene_cfg),
           "--out", str(out_mp4)]
    print("  running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_render(args):
    traj, cfg, theme, stem, _ = _prepare(args)
    out = Path(args.out) if args.out else _default_out(stem, theme, ".mp4")
    final = _render_video(traj, cfg, theme, stem, args.engine, args.scale, out)
    print(f"wrote {final}")


def cmd_preview(args):
    args.engine = "preview"
    traj, cfg, theme, stem, _ = _prepare(args)
    out = Path(args.out) if args.out else _default_out(stem, theme,
                                                       ".preview.mp4")
    final = _render_video(traj, cfg, theme, stem, "preview", args.scale, out)
    print(f"wrote {final}")


def cmd_solve(args):
    traj, cfg, theme, stem, _ = _prepare(args)
    out = Path(args.out) if args.out else _default_out(stem, theme,
                                                       ".traj.json")
    traj.to_json(out, cfg["render"])
    print(f"wrote {out}")


def cmd_batch(args):
    with open(args.jobs) as f:
        jobs = json.load(f)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, job in enumerate(jobs):
        print(f"\n=== job {i + 1}/{len(jobs)}: {job.get('song')} ===")
        ns = _job_namespace(job, args)
        try:
            traj, cfg, theme, stem, _ = _prepare(ns)
            out = Path(job["out"]) if job.get("out") else \
                outdir / f"{stem}__{theme['name']}.mp4"
            _render_video(traj, cfg, theme, stem, ns.engine, ns.scale, out)
            print(f"wrote {out}")
        except SystemExit as ex:
            failed.append((job.get("song"), str(ex)))
        except Exception as ex:  # keep the batch going
            failed.append((job.get("song"), f"{type(ex).__name__}: {ex}"))
    print(f"\nbatch done: {len(jobs) - len(failed)}/{len(jobs)} succeeded")
    for song, err in failed:
        print(f"  FAILED {song}: {err}")
    if failed:
        sys.exit(1)


def _job_namespace(job: dict, args) -> argparse.Namespace:
    return argparse.Namespace(
        song=job["song"], theme=job.get("theme"), bank=job.get("bank"),
        caption=job.get("caption"), seed=job.get("seed"),
        transpose=int(job.get("transpose", 0)),
        tempo_scale=float(job.get("tempo_scale", 1.0)),
        trim=job.get("trim"), loop=int(job.get("loop", 1)),
        track=job.get("track"), force=bool(job.get("force", False)),
        engine=job.get("engine", args.engine), scale=float(job.get("scale", args.scale)),
        out=job.get("out"))


def cmd_validate(args):
    from marblegen.notes import songs_dir
    themes = [args.theme] if args.theme else ["boomwhacker_wall",
                                              "floating_paddles"]
    song_stems = [args.song] if args.song else \
        sorted(p.stem for p in songs_dir().glob("*.mid"))
    n_fail = 0
    for theme_name in themes:
        theme = load_theme(theme_name)
        cfg = build_config(theme)
        for stem in song_stems:
            midi_path, manifest = resolve_song(stem)
            song = load_midi(midi_path)
            hits = to_hits(song, cfg["solver"]["chord_epsilon"])
            traj = solve(hits, cfg["solver"], meta={"song": stem})
            rep = validate(traj, hits, cfg["solver"])
            s = rep["stats"]
            print(f"{'PASS' if rep['ok'] else 'FAIL'} "
                  f"{stem:28s} {theme_name:18s} notes={s['notes']:3d} "
                  f"pegs={s['pegs']:3d} rev={s['reversals']:2d} "
                  f"speeds {s['speed_min']}-{s['speed_max']} m/s "
                  f"warn={len(rep['warnings'])}")
            if not rep["ok"]:
                n_fail += 1
                for e in rep["errors"][:4]:
                    print(f"     ERROR {e}")
    print(f"\n{n_fail} failing combination(s)")
    sys.exit(1 if n_fail else 0)


def cmd_songs(_args):
    from marblegen.notes import songs_dir
    for p in sorted(songs_dir().glob("*.mid")):
        mf = p.with_suffix(".json")
        info = json.load(open(mf)) if mf.exists() else {}
        print(f"{p.stem:28s} {info.get('title', ''):38s} "
              f"{info.get('composer', '')}")


def cmd_banks(_args):
    from marblegen.audio import list_banks
    for b in list_banks():
        spec = json.load(open(REPO_ROOT / "banks" / b / "bank.json"))
        print(f"{b:14s} {spec.get('description', '')}")


def cmd_themes(_args):
    for p in sorted((REPO_ROOT / "themes").glob("*.json")):
        theme = json.load(open(p))
        print(f"{p.stem:20s} {theme.get('description', '')}")


# ---------------------------------------------------------------------------

def _add_song_flags(sp, engine_default: str, scale_default: float):
    sp.add_argument("--song", required=True,
                    help="bundled song name or path to any .mid file")
    sp.add_argument("--theme", help="theme name (see `marblegen themes`)")
    sp.add_argument("--bank", help="sound bank (see `marblegen banks`)")
    sp.add_argument("--caption", help="text overlay burned into the video")
    sp.add_argument("--seed", type=int, help="layout seed (deterministic)")
    sp.add_argument("--transpose", type=int, default=0, help="semitones")
    sp.add_argument("--tempo-scale", type=float, default=1.0, dest="tempo_scale")
    sp.add_argument("--trim", help="time range in seconds, e.g. 0:30")
    sp.add_argument("--loop", type=int, default=1, help="repeat melody N times")
    sp.add_argument("--track", type=int, help="MIDI track override")
    sp.add_argument("--force", action="store_true",
                    help="render even if validation fails")
    sp.add_argument("--engine", default=engine_default,
                    choices=["preview", "blender"])
    sp.add_argument("--scale", type=float, default=scale_default,
                    help="preview resolution scale (1.0 = 1080x1920)")
    sp.add_argument("-o", "--out")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="marblegen",
                                 description="Marble Music video generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("render", help="song -> finished MP4 with audio")
    _add_song_flags(sp, engine_default="preview", scale_default=1.0)
    sp.set_defaults(fn=cmd_render)

    sp = sub.add_parser("preview", help="fast low-res render for iteration")
    _add_song_flags(sp, engine_default="preview", scale_default=0.5)
    sp.set_defaults(fn=cmd_preview)

    sp = sub.add_parser("solve", help="solve + validate, write trajectory JSON")
    _add_song_flags(sp, engine_default="preview", scale_default=1.0)
    sp.set_defaults(fn=cmd_solve)

    sp = sub.add_parser("batch", help="render a queue of jobs from a JSON file")
    sp.add_argument("jobs", help="JSON list of job dicts (see README)")
    sp.add_argument("--outdir", default=str(REPO_ROOT / "out"))
    sp.add_argument("--engine", default="preview",
                    choices=["preview", "blender"])
    sp.add_argument("--scale", type=float, default=1.0)
    sp.set_defaults(fn=cmd_batch)

    sp = sub.add_parser("validate",
                        help="solve+validate songs without rendering")
    sp.add_argument("--song", help="one song (default: all bundled)")
    sp.add_argument("--theme", help="one theme (default: both)")
    sp.set_defaults(fn=cmd_validate)

    sub.add_parser("songs", help="list bundled songs").set_defaults(fn=cmd_songs)
    sub.add_parser("banks", help="list sound banks").set_defaults(fn=cmd_banks)
    sub.add_parser("themes", help="list themes").set_defaults(fn=cmd_themes)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

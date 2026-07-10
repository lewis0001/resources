"""Default configuration and config merging.

Precedence (low -> high): DEFAULTS < theme JSON "solver"/"render" sections
< song manifest overrides < CLI flags.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "solver": {
        # --- physics (never violated) ---
        "gravity": 9.81,            # m/s^2, ONE constant for the whole video
        "restitution": 0.88,        # nominal bounce energy retention (speed ratio)
        "restitution_min": 0.55,    # solver may use a *deader* bounce when needed
        "roll_friction": 0.97,      # tangential speed kept when landing on a ramp
        "roll_accel_factor": 0.7142857142857143,  # 5/7: solid sphere rolling
        # --- layout / search ---
        # horizontal play area; sized so a 9:16 camera at view_height_m fully
        # covers it (view width = 2.3 * 9/16 ≈ 1.29 m around the camera x)
        "x_bounds": [-0.95, 0.95],
        "impact_speed0": 3.0,       # m/s, target speed of the very first hit
        "min_air_dt": 0.18,         # s, gaps shorter than this switch to roll mode
        "max_air_dt": 0.85,         # s, gaps longer than this insert silent pegs
        "min_out_elev_deg": 12.0,   # bounce departure elevation range (from horizontal)
        "max_out_elev_deg": 84.0,
        "n_dir_candidates": 181,    # direction samples per bounce solve
        "instrument_clearance": 0.17,   # m, min distance between hit points
        "arc_clearance": 0.11,      # m, min distance arc <-> earlier instruments
        "clearance_lookback": 16,   # how many recent instruments to check against
        "target_drop_per_hit": 0.22,    # m, preferred net descent per bounce
        "max_rise_per_hit": 0.06,   # m, a bounce may end at most this much higher
        "ramp_angle_deg": [14.0, 44.0],  # roll-mode slope search range
        "ramp_min_spacing": 0.055,  # m, min spacing of instruments along a ramp
        "min_speed": 0.35,          # m/s, sanity floor for |v| at any hit
        "max_speed": 9.0,           # m/s, sanity ceiling
        "chord_epsilon": 0.012,     # s, notes closer than this = one chord hit
        "seed": 7,
    },
    "audio": {
        "sample_rate": 44100,
        "bank": "boomwhacker",
        "master_gain_db": -1.0,
        "pan_spread": 0.6,          # how strongly instrument x position pans audio
        "humanize_db": 1.5,         # random per-hit gain variation (seeded)
        "tail_s": 2.5,              # silence after last note
    },
    "render": {
        "width": 1080,
        "height": 1920,
        "fps": 60,
        "engine": "preview",        # "preview" (matplotlib) or "blender"
        "view_height_m": 2.3,       # world metres visible vertically
        "camera_zeta": 1.0,         # critically damped follow
        "camera_omega": 5.5,        # rad/s, camera spring stiffness
        "camera_lead": 0.28,        # frame the ball this fraction above centre
        "caption": "",
        "show_title": True,
        "lead_in_s": 0.9,           # video time before the first drop starts
    },
    "palette": "boomwhacker",
}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_theme(theme: str | Path) -> dict:
    """Load a theme by name (from themes/) or by explicit path."""
    p = Path(theme)
    if not p.suffix:
        p = REPO_ROOT / "themes" / f"{theme}.json"
    if not p.exists():
        available = sorted(t.stem for t in (REPO_ROOT / "themes").glob("*.json"))
        raise FileNotFoundError(
            f"theme '{theme}' not found; available: {', '.join(available)}")
    with open(p) as f:
        data = json.load(f)
    data["_path"] = str(p)
    return data


def build_config(theme_data: dict | None = None,
                 song_manifest: dict | None = None,
                 cli_overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for layer in (theme_data, song_manifest, cli_overrides):
        if not layer:
            continue
        cfg = _deep_merge(cfg, {k: v for k, v in layer.items()
                                if k in ("solver", "audio", "render", "palette")})
    return cfg

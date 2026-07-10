"""Preview renderer: draws the solved trajectory to a real MP4 with
matplotlib + ffmpeg. It samples the same closed-form segments the validator
checks, so what you watch IS the verified physics. Used both for fast
iteration (`marblegen preview`) and as the fallback engine when Blender is
not installed (`marblegen render --engine preview`).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from marblegen.ffmpeg import ffmpeg_exe
from marblegen.palette import instrument_length, pitch_color
from marblegen.trajectory import Trajectory, unit


def _perp(v):
    return np.array([-v[1], v[0]])


class _Instrument:
    """One drawable strike target with an optional wobble animation."""

    def __init__(self, ax, ev, theme_style: dict, palette):
        self.pos = np.array(ev.pos)
        self.hit_times = [ev.time]
        n = np.array(ev.normal) if ev.normal is not None else np.array([0.0, 1.0])
        is_roll_note = ev.mode == "roll" and ev.kind == "note"

        if ev.kind == "note":
            length = instrument_length(max(ev.pitches))
            length += 0.05 * (len(ev.pitches) - 1)          # chords are wider
            color = pitch_color(max(ev.pitches), palette)
            thick = 2 * float(theme_style.get("tube_radius_m",
                              theme_style.get("paddle_thickness_m", 0.03)))
        else:  # peg / roll_land pad
            length = 0.07 if ev.kind == "roll_land" else 0.05
            color = theme_style.get("peg_color", "#b9bec4")
            thick = 0.026

        if is_roll_note:
            # tubes in a glissando row hang below the rolling line, so
            # neighbours never overlap however tight the ramp spacing is
            axis = unit(-n)
            d0 = 0.006
        else:
            axis = unit(_perp(n))
            d0 = -length / 2
        self.angle_deg = math.degrees(math.atan2(axis[1], axis[0]))

        self.length, self.thick = length, thick
        self.wobble_deg = float(theme_style.get("wobble_deg", 10.0))
        self.wobble_decay = float(theme_style.get("wobble_decay_s", 0.3))

        # bracket: a thin post behind the instrument
        b0 = self.pos + axis * (d0 + length * (0.9 if is_roll_note else 0.5))
        bx, by = b0 - n * 0.055
        self.bracket, = ax.plot([b0[0], bx], [b0[1], by],
                                color=theme_style.get("bracket_color", "#9aa0a6"),
                                lw=2.2, solid_capstyle="round", zorder=2)
        # base rect laid along +x from d0, then rotated about self.pos so the
        # contact point stays the rotation anchor (pendulum-like wobble)
        self.patch = FancyBboxPatch(
            (self.pos[0] + d0, self.pos[1] - thick / 2), length, thick,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.8, edgecolor=(0, 0, 0, 0.18), facecolor=color, zorder=4)
        ax.add_patch(self.patch)
        self._ax = ax
        self._base_angle = self.angle_deg

    def update(self, t: float) -> None:
        wobble = 0.0
        for th in self.hit_times:
            dt = t - th
            if 0.0 <= dt < 4 * self.wobble_decay:
                wobble += (self.wobble_deg * math.exp(-dt / self.wobble_decay)
                           * math.sin(2 * math.pi * 5.2 * dt))
        tr = (transforms.Affine2D()
              .rotate_deg_around(self.pos[0], self.pos[1],
                                 self._base_angle + wobble))
        self.patch.set_transform(tr + self._ax.transData)

    def set_visible(self, vis: bool) -> None:
        self.patch.set_visible(vis)
        self.bracket.set_visible(vis)


def render_preview(traj: Trajectory, theme: dict, cfg: dict,
                   out_mp4: str | Path, scale: float = 1.0,
                   progress=None) -> Path:
    rc = cfg["render"]
    style = theme.get("style", {})
    palette = cfg.get("palette", "boomwhacker")
    fps = int(rc["fps"])
    W = max(2, int(rc["width"] * scale) // 2 * 2)
    H = max(2, int(rc["height"] * scale) // 2 * 2)
    view_h = float(rc["view_height_m"])
    view_w = view_h * W / H

    t0_seg, t_end = traj.time_range()
    video_t0 = t0_seg - float(rc["lead_in_s"])
    n_frames = int((t_end - video_t0) * fps)
    ball_path = traj.sample(fps, video_t0, t_end)
    cam_path = traj.camera_path(fps, video_t0, t_end,
                                omega=float(rc["camera_omega"]),
                                zeta=float(rc["camera_zeta"]),
                                lead=float(rc["camera_lead"]),
                                view_h=view_h)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    bg = style.get("background", "#efeee9")
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- static wall grid (tiled-wall look) -------------------------------
    grid_col = style.get("background_grid")
    if grid_col:
        sp = float(style.get("grid_spacing_m", 0.42))
        y_top = float(np.max(ball_path[:, 1])) + view_h
        y_bot = float(np.min(ball_path[:, 1])) - view_h
        xs = np.arange(-3.0, 3.0, sp)
        ys = np.arange(y_bot - (y_bot % sp), y_top, sp)
        for x in xs:
            ax.plot([x, x], [y_bot, y_top], color=grid_col, lw=1.1, zorder=1)
        for y in ys:
            ax.plot([-3, 3], [y, y], color=grid_col, lw=1.1, zorder=1)

    # --- instruments -------------------------------------------------------
    instruments: list[_Instrument] = []
    for ev in traj.events:
        if ev.kind in ("note", "peg", "roll_land"):
            instruments.append(_Instrument(ax, ev, style, palette))
        elif ev.kind == "reversal" and instruments:
            # a reversal re-strikes the nearest existing instrument
            near = min(instruments,
                       key=lambda i: float(np.linalg.norm(i.pos - np.array(ev.pos))))
            near.hit_times.append(ev.time)

    # --- hit flashes --------------------------------------------------------
    note_events = [e for e in traj.events if e.kind == "note"]
    flash_pool = [Circle((0, 0), 0.01, fill=False, lw=2.0, visible=False,
                         zorder=6) for _ in range(6)]
    for c in flash_pool:
        ax.add_patch(c)

    # --- ball + trail -------------------------------------------------------
    ball_r = float(style.get("ball_radius_m", 0.032))
    ball_col = style.get("ball_color", "#dfe8f0")
    trail_n = 9
    trail_lines = []
    for k in range(trail_n):
        ln, = ax.plot([], [], color=ball_col, lw=3.2 * (1 - k / trail_n),
                      alpha=0.30 * (1 - k / trail_n),
                      solid_capstyle="round", zorder=5)
        trail_lines.append(ln)
    ball = Circle((0, 0), ball_r, facecolor=ball_col,
                  edgecolor=(0, 0, 0, 0.25), lw=1.0, zorder=7)
    shine = Circle((0, 0), ball_r * 0.3, facecolor="white", alpha=0.75, zorder=8)
    ax.add_patch(ball)
    ax.add_patch(shine)

    # --- text overlays ------------------------------------------------------
    caption = rc.get("caption") or ""
    if caption:
        ax.text(0.5, 0.90, caption, transform=ax.transAxes,
                ha="center", va="center", fontsize=15 * scale + 7,
                fontweight="bold", color="#222222", wrap=True, zorder=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="none",
                          alpha=0.92))
    if rc.get("show_title", True):
        title = traj.meta.get("song", "")
        ax.text(0.04, 0.035, title, transform=ax.transAxes, ha="left",
                fontsize=10 * scale + 5, color="#555555", zorder=10)

    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe()
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "21",
                                      "-preset", "veryfast"])
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out_mp4), dpi=100):
        for f in range(n_frames):
            t = video_t0 + f / fps
            cx, cy = cam_path[min(f, len(cam_path) - 1)]
            ax.set_xlim(cx - view_w / 2, cx + view_w / 2)
            ax.set_ylim(cy - view_h / 2, cy + view_h / 2)

            bx, by = ball_path[min(f, len(ball_path) - 1)]
            ball.center = (bx, by)
            shine.center = (bx - ball_r * 0.3, by + ball_r * 0.35)
            for k, ln in enumerate(trail_lines):
                a = f - k - 1
                b = f - k
                if a >= 0:
                    ln.set_data(ball_path[a:b + 1, 0], ball_path[a:b + 1, 1])

            y_lo, y_hi = cy - view_h * 0.75, cy + view_h * 0.75
            for ins in instruments:
                vis = y_lo <= ins.pos[1] <= y_hi
                ins.set_visible(vis)
                if vis:
                    ins.update(t)

            fi = 0
            for ev in note_events:
                dt = t - ev.time
                if 0.0 <= dt < 0.28 and fi < len(flash_pool):
                    c = flash_pool[fi]
                    col = pitch_color(max(ev.pitches), palette)
                    c.set_center(ev.pos)
                    c.set_radius(0.03 + 0.22 * dt)
                    c.set_edgecolor((*col, max(0.0, 0.8 * (1 - dt / 0.28))))
                    c.set_visible(True)
                    fi += 1
            for k in range(fi, len(flash_pool)):
                flash_pool[k].set_visible(False)

            writer.grab_frame()
            if progress and f % (fps * 2) == 0:
                progress(f, n_frames)
    plt.close(fig)
    return out_mp4

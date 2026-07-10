"""Preview renderer: draws the solved trajectory to a real MP4 with
matplotlib + ffmpeg. It samples the same closed-form segments the validator
checks, so what you watch IS the verified physics.

The look is pseudo-3D: soft wall shadows, shaded tube bodies with visible
mouths, glass marbles with floating inner beads, embossed tiles and a gentle
vignette. For the fully 3D look use the Blender engine; this one needs no
extra installs and renders fast, so it is the default and the iteration tool.
"""

from __future__ import annotations

import colorsys
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

from marblegen.ffmpeg import ffmpeg_exe
from marblegen.palette import instrument_length, pitch_color
from marblegen.trajectory import Trajectory, unit


def _perp(v):
    return np.array([-v[1], v[0]])


def _shade(rgb, f):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb(h, min(1, s * (1.15 if f < 1 else 0.9)),
                               max(0, min(1, v * f)))


def _rot_about(angle_deg, cx, cy, ax):
    return (transforms.Affine2D().rotate_deg_around(cx, cy, angle_deg)
            + ax.transData)


BEAD_SETS = [["#e56b6f", "#6d9dc5", "#f2c14e"],
             ["#7fb069", "#b56dc4", "#f28e2b"],
             ["#5bc0be", "#ef767a", "#f7b32b"]]


class _Instrument:
    """One drawable strike target: shadow + shaded body + mouth + mount,
    with a springy wobble on hit."""

    def __init__(self, ax, ev, theme_style: dict, palette, x_center: float):
        self.pos = np.array(ev.pos)
        self.hit_times = [ev.time]
        self.wobble_deg = float(theme_style.get("wobble_deg", 10.0))
        self.wobble_decay = float(theme_style.get("wobble_decay_s", 0.3))
        self._ax = ax
        n = np.array(ev.normal) if ev.normal is not None else np.array([0.0, 1.0])
        n = unit(n)
        if n[1] < 0:
            n = -n
        is_tube = theme_style.get("instrument", "tube") == "tube"
        is_roll_note = ev.mode == "roll" and ev.kind == "note"
        r = float(theme_style.get("tube_radius_m", 0.030))
        pad_t = float(theme_style.get("paddle_thickness_m", 0.028))
        bracket_col = theme_style.get("bracket_color", "#9aa0a6")
        p = self.pos

        self.body_patches: list = []     # wobble with the instrument
        self.static_patches: list = []   # shadows / mounts, no wobble

        def add(patch, wobbles):
            ax.add_patch(patch)
            (self.body_patches if wobbles else self.static_patches).append(patch)

        if ev.kind in ("peg", "roll_land"):
            self._build_peg(ax, add, ev, theme_style)
            self.anchor, self.base_angle = tuple(p), 0.0
            return

        pitch = max(ev.pitches)
        base = pitch_color(pitch, palette)
        L = instrument_length(pitch) + 0.05 * (len(ev.pitches) - 1)

        if is_tube:
            if is_roll_note:
                axis = -n                       # hang below the rolling line
                c = p + axis * (L / 2 + 0.004)
                mouth_at = p + axis * 0.004     # mouth up, ball rolls across it
            else:
                sign = 1.0 if (x_center - p[0]) >= 0 else -1.0
                axis = unit(_perp(n)) * sign    # mouth points toward the lane
                c = p - n * r
                mouth_at = c + axis * (L / 2)
            ang = math.degrees(math.atan2(axis[1], axis[0]))
            th = 2 * r

            # soft wall shadow (two stacked for a blurred edge)
            for grow, alpha in ((1.9, 0.07), (1.25, 0.10)):
                sh = FancyBboxPatch((-L / 2, -th * grow / 2), L, th * grow,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    fc=(0, 0, 0, alpha), ec="none", zorder=2)
                sh.set_transform(transforms.Affine2D().rotate_deg(ang)
                                 .translate(c[0] + 0.028, c[1] - 0.042)
                                 + ax.transData)
                add(sh, wobbles=False)

            def body_piece(y0, height, color, z):
                pc = FancyBboxPatch((-L / 2, y0), L, height,
                                    boxstyle="round,pad=0.002,rounding_size=0.012",
                                    fc=color, ec="none", zorder=z)
                self._local.append((pc, ang, c))
                add(pc, wobbles=True)

            self._local: list = []
            body_piece(-th / 2, th, base, 4)                       # base body
            body_piece(-th / 2, th * 0.30, (*_shade(base, 0.55), 1.0), 5)
            body_piece(th * 0.20, th * 0.30, (*_shade(base, 0.75), 0.9), 5)
            body_piece(th * 0.02, th * 0.22, (1, 1, 1, 0.30), 6)   # gloss

            # tube mouth: rim + dark bore, foreshortened ellipse
            rim = Ellipse((0, 0), th * 0.62, th * 1.06,
                          fc=_shade(base, 1.12), ec=(0, 0, 0, 0.18),
                          lw=0.8, zorder=7)
            bore = Ellipse((0, 0), th * 0.40, th * 0.78,
                           fc=_shade(base, 0.28), ec="none", zorder=8)
            for e_ in (rim, bore):
                self._local.append((e_, ang, mouth_at))
                add(e_, wobbles=True)

            # metal mount behind the far end
            m0 = c - axis * (L * 0.30)
            mount, = ax.plot([m0[0], m0[0] + 0.02], [m0[1], m0[1] - 0.055],
                             color=bracket_col, lw=2.4,
                             solid_capstyle="round", zorder=3)
            self.static_patches.append(mount)
            self.anchor = tuple(p)
            self.base_angle = 0.0
        else:
            axis = unit(_perp(n))
            ang = math.degrees(math.atan2(axis[1], axis[0]))
            c = p - n * (pad_t / 2)
            th = 2.2 * pad_t
            for grow, alpha in ((1.8, 0.07), (1.2, 0.10)):
                sh = FancyBboxPatch((-L / 2, -th * grow / 2), L, th * grow,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    fc=(0, 0, 0, alpha), ec="none", zorder=2)
                sh.set_transform(transforms.Affine2D().rotate_deg(ang)
                                 .translate(c[0] + 0.026, c[1] - 0.04)
                                 + ax.transData)
                add(sh, wobbles=False)
            self._local = []
            pieces = [(-th / 2, th, base, 4),
                      (-th / 2, th * 0.26, (*_shade(base, 0.6), 1.0), 5),
                      (th * 0.10, th * 0.24, (1, 1, 1, 0.32), 6)]
            for y0, hgt, col, z in pieces:
                pc = FancyBboxPatch((-L / 2, y0), L, hgt,
                                    boxstyle="round,pad=0.003,rounding_size=0.016",
                                    fc=col, ec=(0, 0, 0, 0.15), lw=0.6, zorder=z)
                self._local.append((pc, ang, c))
                add(pc, wobbles=True)
            # thin post with ball cap
            post0 = c - n * 0.012
            post1 = c - n * 0.075
            post, = ax.plot([post0[0], post1[0]], [post0[1], post1[1]],
                            color=bracket_col, lw=2.2,
                            solid_capstyle="round", zorder=3)
            cap = Circle(tuple(post1), 0.009, fc=bracket_col, ec="none",
                         zorder=3)
            ax.add_patch(cap)
            self.static_patches += [post, cap]
            self.anchor = tuple(p)
            self.base_angle = 0.0
        self._apply(0.0)

    def _build_peg(self, ax, add, ev, style):
        p = self.pos
        col = style.get("peg_color", "#b9bec4")
        sh = Ellipse((p[0] + 0.018, p[1] - 0.03), 0.055, 0.03,
                     fc=(0, 0, 0, 0.10), ec="none", zorder=2)
        add(sh, wobbles=False)
        knob = Circle(tuple(p), 0.020, fc=col, ec=(0, 0, 0, 0.2), lw=0.7,
                      zorder=4)
        hi = Circle((p[0] - 0.006, p[1] + 0.006), 0.006, fc="white",
                    alpha=0.7, zorder=5)
        stem, = ax.plot([p[0], p[0]], [p[1], p[1] - 0.05],
                        color=style.get("bracket_color", "#9aa0a6"),
                        lw=2.0, solid_capstyle="round", zorder=3)
        add(knob, wobbles=True)
        add(hi, wobbles=True)
        self.static_patches.append(stem)
        self._local = []

    # ------------------------------------------------------------------
    def _apply(self, wobble_deg: float):
        for pc, ang, c in self._local:
            tr = (transforms.Affine2D().rotate_deg(ang).translate(c[0], c[1])
                  .rotate_deg_around(self.anchor[0], self.anchor[1],
                                     wobble_deg))
            pc.set_transform(tr + self._ax.transData)

    def update(self, t: float) -> None:
        if not self._local:
            return
        wob = 0.0
        for th in self.hit_times:
            dt = t - th
            if 0.0 <= dt < 4 * self.wobble_decay:
                wob += (self.wobble_deg * math.exp(-dt / self.wobble_decay)
                        * math.sin(2 * math.pi * 5.2 * dt))
        self._apply(wob)

    def set_visible(self, vis: bool) -> None:
        for p in self.body_patches:
            p.set_visible(vis)
        for p in self.static_patches:
            p.set_visible(vis)


class _Marble:
    """Glass marble with floating coloured beads, specular highlights, a wall
    shadow and a motion trail."""

    def __init__(self, ax, track: int, radius: float, path: np.ndarray,
                 start_t: float, video_t0: float, fps: int):
        self.r = radius
        self.path = path
        self.start_frame = max(0, int((start_t - video_t0) * fps) - int(0.1 * fps))
        d = np.diff(path, axis=0)
        self.dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])
        r = radius
        self.shadow = Ellipse((0, 0), 2.6 * r, 1.5 * r, fc=(0, 0, 0, 0.12),
                              ec="none", zorder=9)
        self.glass = Circle((0, 0), r, fc=(1, 1, 1, 0.44),
                            ec=(0.25, 0.28, 0.33, 0.55), lw=1.4, zorder=11)
        self.rimlight = Circle((0, 0), r * 0.86, fc="none",
                               ec=(1, 1, 1, 0.55), lw=1.6, zorder=12)
        beads = BEAD_SETS[track % len(BEAD_SETS)]
        self.beads = [Circle((0, 0), r * 0.34, fc=b, ec=(0, 0, 0, 0.15),
                             lw=0.5, alpha=0.95, zorder=13) for b in beads]
        self.spec = Circle((0, 0), r * 0.22, fc="white", alpha=0.95, zorder=14)
        self.spec2 = Circle((0, 0), r * 0.10, fc="white", alpha=0.6, zorder=14)
        self.trail = []
        for k in range(8):
            ln, = ax.plot([], [], color="white", lw=3.4 * (1 - k / 8),
                          alpha=0.16 * (1 - k / 8), solid_capstyle="round",
                          zorder=8)
            self.trail.append(ln)
        for art in [self.shadow, self.glass, self.rimlight, *self.beads,
                    self.spec, self.spec2]:
            ax.add_patch(art)
        self.artists = [self.shadow, self.glass, self.rimlight, *self.beads,
                        self.spec, self.spec2, *self.trail]

    def update(self, f: int) -> None:
        vis = f >= self.start_frame
        for a in self.artists:
            a.set_visible(vis)
        if not vis:
            return
        i = min(f, len(self.path) - 1)
        bx, by = self.path[i]
        r = self.r
        self.shadow.set_center((bx + 0.045, by - 0.06))
        self.glass.center = (bx, by)
        self.rimlight.center = (bx, by)
        roll = self.dist[i] / max(r, 1e-6)
        for k, bead in enumerate(self.beads):
            a = roll * 0.6 + k * 2.1
            bead.center = (bx + 0.42 * r * math.cos(a),
                           by + 0.42 * r * math.sin(a))
        self.spec.center = (bx - 0.34 * r, by + 0.38 * r)
        self.spec2.center = (bx + 0.28 * r, by - 0.20 * r)
        for k, ln in enumerate(self.trail):
            a, b = f - k - 1, f - k
            if a >= self.start_frame and a >= 0:
                ln.set_data(self.path[a:b + 1, 0], self.path[a:b + 1, 1])
            else:
                ln.set_data([], [])


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
    x_lo, x_hi = cfg["solver"]["x_bounds"]

    t0_seg, t_end = traj.time_range()
    video_t0 = t0_seg - float(rc["lead_in_s"])
    n_frames = int((t_end - video_t0) * fps)
    tracks = traj.tracks()
    ball_paths = {tr: traj.sample(fps, video_t0, t_end, tr) for tr in tracks}
    cam_path = traj.camera_path(fps, video_t0, t_end,
                                omega=float(rc["camera_omega"]),
                                zeta=float(rc["camera_zeta"]),
                                lead=float(rc["camera_lead"]),
                                view_h=view_h)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    bg = style.get("background", "#efeee9")
    bg_rgb = tuple(int(bg.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.axis("off")

    all_y = np.concatenate([p[:, 1] for p in ball_paths.values()])
    y_top = float(all_y.max()) + view_h
    y_bot = float(all_y.min()) - view_h

    # --- background: vertical gradient + embossed tile grid ----------------
    grad = np.zeros((256, 1, 3))
    top = np.clip(np.array(bg_rgb) * 1.05, 0, 1)
    bot = np.clip(np.array(bg_rgb) * 0.92, 0, 1)
    for i in range(256):
        grad[i, 0] = top + (bot - top) * (i / 255)
    ax.imshow(grad, extent=(-3.5, 3.5, y_bot, y_top), aspect="auto",
              zorder=0, interpolation="bilinear")
    grid_col = style.get("background_grid")
    if grid_col:
        sp = float(style.get("grid_spacing_m", 0.42))
        for x in np.arange(-3.0, 3.0, sp):
            ax.plot([x, x], [y_bot, y_top], color=grid_col, lw=1.3,
                    alpha=0.9, zorder=1)
            ax.plot([x + 0.006, x + 0.006], [y_bot, y_top], color="white",
                    lw=1.0, alpha=0.5, zorder=1)
        for y in np.arange(y_bot - (y_bot % sp), y_top, sp):
            ax.plot([-3.5, 3.5], [y, y], color=grid_col, lw=1.3,
                    alpha=0.9, zorder=1)
            ax.plot([-3.5, 3.5], [y - 0.006, y - 0.006], color="white",
                    lw=1.0, alpha=0.5, zorder=1)

    # --- instruments ---------------------------------------------------------
    n_lanes = max(1, len(tracks))
    lane_w = (x_hi - x_lo) / n_lanes
    instruments: list[_Instrument] = []
    for ev in traj.events:
        if ev.kind in ("note", "peg", "roll_land"):
            xc = x_lo + (ev.track + 0.5) * lane_w
            instruments.append(_Instrument(ax, ev, style, palette, xc))
        elif ev.kind == "reversal" and instruments:
            near = min(instruments,
                       key=lambda i: float(np.linalg.norm(i.pos - np.array(ev.pos))))
            near.hit_times.append(ev.time)

    # --- hit flashes -----------------------------------------------------------
    note_events = [e for e in traj.events if e.kind == "note"]
    flash_pool = [Circle((0, 0), 0.01, fill=False, lw=1.6, visible=False,
                         zorder=16) for _ in range(8)]
    for c in flash_pool:
        ax.add_patch(c)

    # --- marbles ------------------------------------------------------------------
    ball_r = float(style.get("ball_radius_m", 0.032))
    marbles = [_Marble(ax, tr, ball_r, ball_paths[tr],
                       traj.track_start(tr), video_t0, fps)
               for tr in tracks]

    # --- vignette -------------------------------------------------------------------
    vg = np.zeros((96, 96, 4))
    yy, xx = np.mgrid[0:96, 0:96]
    rad = np.sqrt(((xx - 47.5) / 47.5) ** 2 + ((yy - 47.5) / 47.5) ** 2)
    vg[..., 3] = np.clip((rad - 0.55) / 0.9, 0, 1) ** 2 * 0.30
    ax.imshow(vg, extent=(0, 1, 0, 1), transform=ax.transAxes, zorder=25,
              interpolation="bilinear", aspect="auto")

    # --- text overlays ------------------------------------------------------------
    caption = rc.get("caption") or ""
    if caption:
        ax.text(0.5, 0.90, caption, transform=ax.transAxes,
                ha="center", va="center", fontsize=15 * scale + 7,
                fontweight="bold", color="#1c1c1e", wrap=True, zorder=30,
                bbox=dict(boxstyle="round,pad=0.55", fc="white", ec="none",
                          alpha=0.94))
    if rc.get("show_title", True):
        title = traj.meta.get("song", "")
        ax.text(0.045, 0.032, title, transform=ax.transAxes, ha="left",
                fontsize=10 * scale + 5, color=(0.35, 0.35, 0.37),
                fontweight="bold", zorder=30)

    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe()
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20",
                                      "-preset", "veryfast",
                                      "-movflags", "+faststart"])
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out_mp4), dpi=100):
        for f in range(n_frames):
            t = video_t0 + f / fps
            cx, cy = cam_path[min(f, len(cam_path) - 1)]
            ax.set_xlim(cx - view_w / 2, cx + view_w / 2)
            ax.set_ylim(cy - view_h / 2, cy + view_h / 2)

            for m in marbles:
                m.update(f)

            y_lo_v, y_hi_v = cy - view_h * 0.8, cy + view_h * 0.8
            for ins in instruments:
                vis = y_lo_v <= ins.pos[1] <= y_hi_v
                ins.set_visible(vis)
                if vis:
                    ins.update(t)

            fi = 0
            for ev in note_events:
                dt = t - ev.time
                if 0.0 <= dt < 0.26 and fi < len(flash_pool):
                    c = flash_pool[fi]
                    col = pitch_color(max(ev.pitches), palette)
                    c.set_center(ev.pos)
                    c.set_radius(0.035 + 0.20 * dt)
                    c.set_edgecolor((*col, max(0.0, 0.65 * (1 - dt / 0.26))))
                    c.set_visible(True)
                    fi += 1
            for k in range(fi, len(flash_pool)):
                flash_pool[k].set_visible(False)

            writer.grab_frame()
            if progress and f % (fps * 2) == 0:
                progress(f, n_frames)
    plt.close(fig)
    return out_mp4

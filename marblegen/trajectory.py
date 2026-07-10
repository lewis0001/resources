"""Trajectory data model.

A solved video is a sequence of analytic motion segments plus a list of hit
events. Physics is stored in closed form (parabolas / rolling arcs), so both
renderers and the validator sample the exact same motion — there is no
separate "animation" that could drift from the physics.

Coordinates: 2D vertical plane. x = horizontal (metres, + right),
y = vertical (+ up). The marble descends over the song; the camera follows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Segment:
    """Closed-form motion between two consecutive contact events."""
    kind: str            # 'air' | 'roll'
    t0: float
    t1: float
    p0: tuple[float, float]
    v0: tuple[float, float]          # velocity at t0
    gravity: float = 9.81
    # roll-only fields:
    tangent: tuple[float, float] | None = None   # unit vector, down-slope
    accel: float = 0.0                           # signed accel along tangent

    def pos(self, t: float) -> np.ndarray:
        tau = t - self.t0
        if self.kind == "air":
            return np.array([
                self.p0[0] + self.v0[0] * tau,
                self.p0[1] + self.v0[1] * tau - 0.5 * self.gravity * tau * tau,
            ])
        u = np.array(self.tangent)
        speed0 = float(np.dot(self.v0, u))
        s = speed0 * tau + 0.5 * self.accel * tau * tau
        return np.array(self.p0) + u * s

    def vel(self, t: float) -> np.ndarray:
        tau = t - self.t0
        if self.kind == "air":
            return np.array([self.v0[0], self.v0[1] - self.gravity * tau])
        u = np.array(self.tangent)
        speed0 = float(np.dot(self.v0, u))
        return u * (speed0 + self.accel * tau)


@dataclass
class Event:
    """A contact: an instrument strike, a silent peg, a ramp landing, or the
    spawn/release of the marble."""
    time: float
    kind: str                        # 'spawn'|'note'|'peg'|'roll_land'|'reversal'
    pos: tuple[float, float] = (0.0, 0.0)
    pitches: list[int] = field(default_factory=list)
    velocity: int = 96
    v_in: tuple[float, float] = (0.0, 0.0)
    v_out: tuple[float, float] = (0.0, 0.0)
    normal: tuple[float, float] | None = None    # contact normal (bounces)
    mode: str = "bounce"             # 'bounce' | 'roll' (instrument on a ramp)
    tangent: tuple[float, float] | None = None   # ramp direction for roll hits
    e_used: float | None = None      # restitution actually applied (bounces)


@dataclass
class Trajectory:
    events: list[Event]
    segments: list[Segment]
    meta: dict

    # ------------------------------------------------------------------
    def time_range(self) -> tuple[float, float]:
        return self.segments[0].t0, self.segments[-1].t1

    def pos_at(self, t: float) -> np.ndarray:
        segs = self.segments
        if t <= segs[0].t0:
            return segs[0].pos(segs[0].t0)
        for s in segs:
            if t <= s.t1:
                return s.pos(t)
        return segs[-1].pos(segs[-1].t1)

    def sample(self, fps: int, t_start: float, t_end: float) -> np.ndarray:
        n = max(2, int(round((t_end - t_start) * fps)) + 1)
        ts = t_start + np.arange(n) / fps
        return np.array([self.pos_at(t) for t in ts])

    # ------------------------------------------------------------------
    def camera_path(self, fps: int, t_start: float, t_end: float,
                    omega: float = 5.5, zeta: float = 1.0,
                    lead: float = 0.0, view_h: float = 2.3) -> np.ndarray:
        """Critically damped 2nd-order follow of the ball. Returns [n,2]
        camera-centre positions. `lead` shifts the target down so the ball sits
        above centre-frame (lead is a fraction of view height)."""
        ball = self.sample(fps, t_start, t_end)
        target = ball.copy()
        target[:, 0] *= 0.35              # follow x only gently — keeps frame calm
        target[:, 1] -= lead * view_h     # ball rides above centre-frame
        cam = np.zeros_like(target)
        cam[0] = target[0]
        vel = np.zeros(2)
        dt = 1.0 / fps
        for i in range(1, len(target)):
            acc = omega * omega * (target[i] - cam[i - 1]) - 2.0 * zeta * omega * vel
            vel = vel + acc * dt
            cam[i] = cam[i - 1] + vel * dt
        return cam

    # ------------------------------------------------------------------
    def to_json(self, path: str | Path, render_cfg: dict | None = None) -> None:
        """Export everything a renderer needs (analytic segments + baked
        samples + camera path) as one JSON file."""
        rc = render_cfg or {}
        fps = int(rc.get("fps", 60))
        t0, t1 = self.time_range()
        lead_in = float(rc.get("lead_in_s", 0.9))
        video_t0 = t0 - lead_in
        ball = self.sample(fps, video_t0, t1)
        cam = self.camera_path(fps, video_t0, t1,
                               omega=float(rc.get("camera_omega", 5.5)),
                               zeta=float(rc.get("camera_zeta", 1.0)),
                               lead=float(rc.get("camera_lead", 0.28)),
                               view_h=float(rc.get("view_height_m", 2.3)))

        def r(x):
            return round(float(x), 5)

        data = {
            "meta": {**self.meta, "video_t0": r(video_t0), "fps": fps,
                     "t_end": r(t1)},
            "events": [{
                "time": r(e.time), "kind": e.kind, "pos": [r(e.pos[0]), r(e.pos[1])],
                "pitches": e.pitches, "velocity": e.velocity,
                "v_in": [r(e.v_in[0]), r(e.v_in[1])],
                "v_out": [r(e.v_out[0]), r(e.v_out[1])],
                "normal": [r(e.normal[0]), r(e.normal[1])] if e.normal else None,
                "mode": e.mode,
                "tangent": [r(e.tangent[0]), r(e.tangent[1])] if e.tangent else None,
                "e_used": r(e.e_used) if e.e_used is not None else None,
            } for e in self.events],
            "segments": [{
                "kind": s.kind, "t0": r(s.t0), "t1": r(s.t1),
                "p0": [r(s.p0[0]), r(s.p0[1])], "v0": [r(s.v0[0]), r(s.v0[1])],
                "gravity": r(s.gravity),
                "tangent": [r(s.tangent[0]), r(s.tangent[1])] if s.tangent else None,
                "accel": r(s.accel),
            } for s in self.segments],
            "ball": {"fps": fps, "t0": r(video_t0),
                     "positions": [[r(p[0]), r(p[1])] for p in ball]},
            "camera": {"fps": fps, "t0": r(video_t0),
                       "positions": [[r(p[0]), r(p[1])] for p in cam]},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def from_json(path: str | Path) -> "Trajectory":
        with open(path) as f:
            d = json.load(f)
        events = [Event(time=e["time"], kind=e["kind"], pos=tuple(e["pos"]),
                        pitches=e["pitches"], velocity=e["velocity"],
                        v_in=tuple(e["v_in"]), v_out=tuple(e["v_out"]),
                        normal=tuple(e["normal"]) if e["normal"] else None,
                        mode=e["mode"],
                        tangent=tuple(e["tangent"]) if e["tangent"] else None,
                        e_used=e["e_used"]) for e in d["events"]]
        segments = [Segment(kind=s["kind"], t0=s["t0"], t1=s["t1"],
                            p0=tuple(s["p0"]), v0=tuple(s["v0"]),
                            gravity=s["gravity"],
                            tangent=tuple(s["tangent"]) if s["tangent"] else None,
                            accel=s["accel"]) for s in d["segments"]]
        return Trajectory(events=events, segments=segments, meta=d["meta"])


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 1.0])


def bounce_normal(v_in, v_out) -> np.ndarray:
    """Contact normal that turns v_in into v_out: the unit bisector between the
    reversed incoming direction and the outgoing direction. This is exactly the
    geometry of a real bounce, so orienting the instrument face along this
    normal makes every bounce read as physically caused. Signed so the normal
    opposes the incoming velocity (points out of the struck surface)."""
    n = unit(unit(v_out) - unit(v_in))
    if float(np.dot(n, np.asarray(v_in, dtype=float))) > 0:
        n = -n
    return n

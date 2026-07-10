"""Forward physics solver.

The layout is derived FROM the physics, never the other way around:

* One gravity constant for the whole video.
* At every bounce the departure speed is exactly ``e_used * |v_in|`` with
  ``e_used <= restitution`` (a bounce can only lose energy). The only free
  parameter is the departure *direction* — equivalently, the orientation of
  the instrument face, whose normal is the bisector of the in/out directions.
* The next instrument is placed wherever the ball genuinely is at the next
  note's timestamp (closed-form parabola), so hits are sample-exact and the
  motion can never speed up or slow down without a visible physical cause.

Awkward timing is absorbed by physical fallbacks, not by cheating:
long gaps insert silent bounce pegs; fast runs switch to roll mode (a ramp of
instruments, tangential acceleration = 5/7 * g * sin(theta) for a solid
sphere, with the ramp zig-zagging at the play-area edges); chords become one
wide instrument struck once. The ball leaves a ramp via a normal restitution
bounce off the last tube's lip, so ramp exits are ordinary, physical bounces.

Solver state machine: we always stand at a *pending contact* (position,
incoming velocity, timestamp, and what kind of thing was struck). Looking at
the next planned target tells us the exact flight time; we then search the
departure direction, emit the pending contact's event with its now-known
v_out, and advance. Roll runs emit all their events internally and leave
their final note as the new pending contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from marblegen.notes import Hit
from marblegen.trajectory import Event, Segment, Trajectory, bounce_normal, unit

G_DOWN = np.array([0.0, -1.0])


class SolveError(RuntimeError):
    pass


@dataclass
class _Ctx:
    cfg: dict
    rng: np.random.Generator
    # placed instruments: (position, mode) where mode is 'bounce' | 'roll'
    instruments: list[tuple[np.ndarray, str]] = field(default_factory=list)
    prev_dx_sign: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def required_gap(self, mode_a: str, mode_b: str) -> float:
        if mode_a == "bounce" and mode_b == "bounce":
            return float(self.cfg["instrument_clearance"])
        return max(1.5 * float(self.cfg["ramp_min_spacing"]), 0.09)

    def near_obstacles(self, p: np.ndarray, dy: float = 0.5) -> list[np.ndarray]:
        return [q for q, _ in self.instruments if abs(q[1] - p[1]) < dy]

    def placement_ok(self, p: np.ndarray, mode: str) -> bool:
        for q, m in self.instruments:
            if abs(q[1] - p[1]) > 0.5:
                continue
            if float(np.linalg.norm(p - q)) < self.required_gap(mode, m):
                return False
        return True

    def add(self, p: np.ndarray, mode: str) -> None:
        self.instruments.append((np.asarray(p, dtype=float).copy(), mode))


@dataclass
class _Pending:
    """A contact whose departure is not yet decided."""
    hit: Hit | None          # None for silent pegs
    kind: str                # 'note' | 'peg'
    mode: str                # 'bounce' | 'roll' (roll = sits on a ramp)
    tangent: tuple | None = None
    ramp_normal: tuple | None = None


# ---------------------------------------------------------------------------
# Plan: notes -> steps (air notes, silent pegs, roll runs)
# ---------------------------------------------------------------------------

def _build_plan(hits: list[Hit], cfg: dict) -> list[dict]:
    min_air = float(cfg["min_air_dt"])
    max_air = float(cfg["max_air_dt"])

    steps: list[dict] = []
    i = 0
    while i < len(hits):
        j = i
        while j + 1 < len(hits) and (hits[j + 1].time - hits[j].time) < min_air:
            j += 1
        if j > i:
            steps.append({"kind": "run", "hits": hits[i:j + 1]})
            i = j + 1
        else:
            steps.append({"kind": "note", "hit": hits[i]})
            i += 1

    out: list[dict] = []
    target = 0.75 * max_air
    for k, st in enumerate(steps):
        if k > 0:
            t_prev = _step_end_time(out[-1])
            t_next = _step_start_time(st)
            gap = t_next - t_prev
            if gap > max_air:
                n_pegs = int(math.ceil(gap / target)) - 1
                for m in range(1, n_pegs + 1):
                    out.append({"kind": "peg",
                                "time": t_prev + gap * m / (n_pegs + 1)})
        out.append(st)
    return out


def _step_start_time(st: dict) -> float:
    if st["kind"] == "run":
        return st["hits"][0].time
    if st["kind"] == "note":
        return st["hit"].time
    return st["time"]


def _step_end_time(st: dict) -> float:
    if st["kind"] == "run":
        return st["hits"][-1].time
    return _step_start_time(st)


# ---------------------------------------------------------------------------
# Bounce solve: pick a physical departure direction
# ---------------------------------------------------------------------------

def _candidate_dirs(cfg: dict, relaxed: bool) -> list[np.ndarray]:
    lo = 5.0 if relaxed else float(cfg["min_out_elev_deg"])
    hi = 88.0 if relaxed else float(cfg["max_out_elev_deg"])
    n = int(cfg["n_dir_candidates"])
    per_side = max(8, n // 2)
    elevs = np.linspace(math.radians(lo), math.radians(hi), per_side)
    dirs = []
    for side in (1.0, -1.0):
        for el in elevs:
            dirs.append(np.array([side * math.cos(el), math.sin(el)]))
    return dirs


def _arc_clear(p0: np.ndarray, v_out: np.ndarray, dt: float, g: float,
               obstacles: list[np.ndarray], clearance: float) -> bool:
    if not obstacles:
        return True
    for tau in np.linspace(0.15 * dt, 0.85 * dt, 9):
        q = p0 + v_out * tau + 0.5 * G_DOWN * g * tau * tau
        for ob in obstacles:
            if float(np.linalg.norm(q - ob)) < clearance:
                return False
    return True


def _bounce_step(ctx: _Ctx, p: np.ndarray, v_in: np.ndarray, dt: float,
                 step_idx: int, next_mode: str = "bounce"
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Choose v_out at a contact so that after exactly dt of ballistic flight
    the ball arrives somewhere usable. Returns (v_out, p_next, v_in_next,
    e_used). Physics guarantee: |v_out| = e_used * |v_in|, e_used <= e."""
    cfg = ctx.cfg
    g = float(cfg["gravity"])
    e_nom = float(cfg["restitution"])
    e_min = float(cfg["restitution_min"])
    x_lo, x_hi = cfg["x_bounds"]
    speed_in = float(np.linalg.norm(v_in))
    jitter = ctx.rng.normal(0.0, 1.0, size=2048)

    def attempt(relaxed: bool):
        best, best_score = None, float("inf")
        dirs = _candidate_dirs(cfg, relaxed)
        max_rise = 0.35 if relaxed else float(cfg["max_rise_per_hit"])
        arc_cl = 0.06 if relaxed else float(cfg["arc_clearance"])
        obstacles = ctx.near_obstacles(p, dy=0.9)
        ji = 0
        for e_used in np.linspace(e_nom, e_min, 5):
            s_out = min(e_used * speed_in, float(cfg["max_speed"]))
            for d in dirs:
                ji += 1
                v_out = s_out * d
                p_next = p + v_out * dt + 0.5 * G_DOWN * g * dt * dt
                v_next = v_out + G_DOWN * g * dt
                # --- hard constraints ---
                if not (x_lo + 0.05 <= p_next[0] <= x_hi - 0.05):
                    continue
                if p_next[1] - p[1] > max_rise:
                    continue
                if v_next[1] > -0.25:          # must arrive descending
                    continue
                if next_mode == "roll" and abs(v_next[0]) > 2.4:
                    continue                    # ramps need steady arrivals
                if float(np.linalg.norm(p_next - p)) < \
                        0.8 * float(cfg["instrument_clearance"]):
                    continue
                if not ctx.placement_ok(p_next, next_mode):
                    continue
                if not _arc_clear(p, v_out, dt, g, obstacles, arc_cl):
                    continue
                # --- scoring (lower is better) ---
                drop = p[1] - p_next[1]
                x_pull = 0.5 if abs(p[0]) < 0.8 * x_hi else 1.6
                dx = p_next[0] - p[0]
                altern = 0.0
                if not relaxed and ctx.prev_dx_sign != 0 and dx != 0 and \
                        math.copysign(1, dx) == ctx.prev_dx_sign:
                    altern = 0.25
                elev = math.degrees(math.asin(max(-1.0, min(1.0, float(d[1])))))
                score = (0.9 * x_pull * abs(p_next[0])
                         + 1.4 * abs(drop - float(cfg["target_drop_per_hit"]))
                         + altern
                         + 0.004 * abs(elev - 50.0)
                         + 0.8 * (e_nom - e_used)      # prefer lively bounces
                         + 0.02 * abs(float(jitter[ji % jitter.size])))
                if next_mode == "roll":
                    score += 0.6 * abs(float(v_next[0]))  # gentler roll entry
                if score < best_score:
                    best_score = score
                    best = (v_out, p_next, v_next, float(e_used))
        return best

    got = attempt(relaxed=False) or attempt(relaxed=True)
    if got is None:
        # Guaranteed-physical last resort: nearly straight up, nudged to centre.
        s_out = e_min * speed_in
        side = -1.0 if p[0] > 0 else 1.0
        d = unit(np.array([0.12 * side, 1.0]))
        v_out = s_out * d
        p_next = p + v_out * dt + 0.5 * G_DOWN * g * dt * dt
        v_next = v_out + G_DOWN * g * dt
        ctx.warnings.append(f"contact {step_idx}: forced fallback bounce "
                            f"(layout constraints could not all be met)")
        got = (v_out, p_next, v_next, e_min)
    v_out, p_next, v_next, e_used = got
    if p_next[0] != p[0]:
        ctx.prev_dx_sign = math.copysign(1, p_next[0] - p[0])
    return v_out, p_next, v_next, e_used


# ---------------------------------------------------------------------------
# Roll solve: fast note runs on a ramp
# ---------------------------------------------------------------------------

def _solve_run(ctx: _Ctx, p_land: np.ndarray, v_land: np.ndarray, t_land: float,
               run_hits: list[Hit], events: list[Event],
               segments: list[Segment]
               ) -> tuple[_Pending, np.ndarray, np.ndarray, float]:
    """Roll the ball along ramp(s) so every run note is struck exactly on
    time. Emits the landing and all notes except the last, which is returned
    as the new pending contact (its departure is a normal bounce later).
    Returns (pending, pos, rolling velocity, time) at the last run note."""
    cfg = ctx.cfg
    g = float(cfg["gravity"])
    x_lo, x_hi = cfg["x_bounds"]
    roll_k = float(cfg["roll_accel_factor"])
    fric = float(cfg["roll_friction"])
    th_lo, th_hi = (math.radians(a) for a in cfg["ramp_angle_deg"])
    e_nom = float(cfg["restitution"])
    min_sp = float(cfg["ramp_min_spacing"])
    lands_on_first = abs(run_hits[0].time - t_land) < 1e-9
    remaining = run_hits[1:] if lands_on_first else run_hits
    if not remaining:  # degenerate: single-hit run at the landing
        u = np.array([1.0, -0.3])
        pend = _Pending(hit=run_hits[-1], kind="note", mode="roll",
                        tangent=tuple(unit(u)), ramp_normal=(0.0, 1.0))
        return pend, p_land, v_land * 0.3, t_land

    room_right = x_hi - p_land[0]
    room_left = p_land[0] - x_lo
    prefer = 1.0 if room_right >= room_left else -1.0

    def in_bounds(pos: np.ndarray) -> bool:
        return x_lo + 0.04 <= pos[0] <= x_hi - 0.04

    def simulate(direction: float, theta: float, forced: bool = False):
        """Walk the whole run (including zig-zag reversals) for one candidate
        ramp. Returns (actions, conflicts, vt0, u0, a) or None if physically
        impossible. Conflicts count layout problems (tight spacing, proximity
        to existing instruments, out-of-frame notes) — never physics ones."""
        u = np.array([direction * math.cos(theta), -math.sin(theta)])
        n = np.array([direction * math.sin(theta), math.cos(theta)])
        vt0 = float(np.dot(v_land, u)) * fric
        if forced:
            vt0 = max(0.3, vt0)
        elif vt0 < 0.25 or float(np.dot(v_land, n)) > -0.05:
            return None                     # not actually landing onto it
        a = roll_k * g * math.sin(theta)
        u0 = u.copy()
        actions: list[tuple] = []
        conflicts = 0
        placed_local = [p_land.copy()]
        t_ref, p_ref, vt, dd = t_land, p_land.copy(), vt0, direction
        last_pos, last_v, last_t = p_land.copy(), u * vt0, t_land
        for k, h in enumerate(remaining):
            tau = h.time - t_ref
            s = vt * tau + 0.5 * a * tau * tau
            if k == 0 and not lands_on_first and s < 0.02 and not forced:
                return None
            pos = p_ref + u * s
            v_here = u * (vt + a * tau)
            if not in_bounds(pos):
                if last_t - t_ref <= 1e-9:
                    if not forced:
                        return None         # would reverse twice in one spot
                    conflicts += 5
                else:
                    dd = -dd
                    u = np.array([dd * math.cos(theta), -math.sin(theta)])
                    vt = e_nom * float(np.linalg.norm(last_v))
                    actions.append(("rev", last_t, last_pos.copy(),
                                    last_v.copy(), u * vt, u.copy(), dd))
                    t_ref, p_ref = last_t, last_pos.copy()
                    tau = h.time - t_ref
                    s = vt * tau + 0.5 * a * tau * tau
                    pos = p_ref + u * s
                    v_here = u * (vt + a * tau)
                    if not in_bounds(pos):
                        conflicts += 3
            if float(np.linalg.norm(pos - last_pos)) < min_sp:
                conflicts += 1
            if any(float(np.linalg.norm(pos - q)) < min_sp
                   for q in placed_local[:-1]):
                conflicts += 1
            if not ctx.placement_ok(pos, "roll"):
                conflicts += 1
            actions.append(("note", h, pos.copy(), v_here.copy(), u.copy(), dd))
            placed_local.append(pos.copy())
            last_pos, last_v, last_t = pos, v_here, h.time
        return actions, conflicts, vt0, u0, a

    best = None
    for direction in (prefer, -prefer):
        for theta in np.linspace(th_lo, th_hi, 9):
            r = simulate(direction, theta)
            if r is None:
                continue
            actions, conflicts, vt0, u0, a = r
            score = (10.0 * conflicts + abs(theta - math.radians(24))
                     + (0.0 if direction == prefer else 0.4) + 0.12 * vt0)
            if best is None or score < best[0]:
                best = (score, direction, theta, actions, conflicts, vt0, u0, a)
    if best is None:
        direction, theta = prefer, th_hi
        actions, conflicts, vt0, u0, a = simulate(direction, theta, forced=True)
        best = (0.0, direction, theta, actions, conflicts, vt0, u0, a)
    _, direction0, theta, actions, conflicts, vt0, u0, a = best
    if conflicts:
        ctx.warnings.append(f"run at t={t_land:.2f}: {conflicts} layout "
                            f"conflict(s) — tight spacing possible")

    def normal_for(dd: float):
        return (float(dd * math.sin(theta)), float(math.cos(theta)))

    # --- landing event -----------------------------------------------------
    if lands_on_first:
        first = run_hits[0]
        events.append(Event(time=t_land, kind="note", pos=tuple(p_land),
                            pitches=list(first.pitches), velocity=first.velocity,
                            v_in=tuple(v_land), v_out=tuple(u0 * vt0),
                            normal=normal_for(direction0), mode="roll",
                            tangent=tuple(u0)))
    else:
        events.append(Event(time=t_land, kind="roll_land", pos=tuple(p_land),
                            v_in=tuple(v_land), v_out=tuple(u0 * vt0),
                            normal=normal_for(direction0), mode="roll",
                            tangent=tuple(u0)))
    ctx.add(p_land, "roll")

    # --- replay the winning walk: events + roll segments --------------------
    seg_t0, seg_p0, seg_v0, seg_u = t_land, p_land.copy(), u0 * vt0, u0
    final_pos, final_v, final_t, final_u, final_dd = \
        p_land.copy(), u0 * vt0, t_land, u0, direction0
    n_notes = sum(1 for act in actions if act[0] == "note")
    seen_notes = 0
    for act in actions:
        if act[0] == "rev":
            _, t_rev, p_rev, v_in_rev, v_out_rev, u_new, dd = act
            if t_rev - seg_t0 > 1e-9:
                segments.append(Segment(kind="roll", t0=seg_t0, t1=t_rev,
                                        p0=tuple(seg_p0), v0=tuple(seg_v0),
                                        gravity=g, tangent=tuple(seg_u),
                                        accel=a))
            events.append(Event(time=t_rev, kind="reversal", pos=tuple(p_rev),
                                v_in=tuple(v_in_rev), v_out=tuple(v_out_rev),
                                normal=tuple(bounce_normal(v_in_rev, v_out_rev)),
                                mode="roll", tangent=tuple(u_new),
                                e_used=e_nom))
            seg_t0, seg_p0, seg_v0, seg_u = t_rev, p_rev, v_out_rev, u_new
        else:
            _, h, pos, v_here, u_here, dd = act
            seen_notes += 1
            if seen_notes < n_notes:        # final note stays pending
                events.append(Event(time=h.time, kind="note", pos=tuple(pos),
                                    pitches=list(h.pitches),
                                    velocity=h.velocity,
                                    v_in=tuple(v_here), v_out=tuple(v_here),
                                    normal=normal_for(dd), mode="roll",
                                    tangent=tuple(u_here)))
            ctx.add(pos, "roll")
            final_pos, final_v, final_t = pos, v_here, h.time
            final_u, final_dd = u_here, dd
    segments.append(Segment(kind="roll", t0=seg_t0, t1=final_t,
                            p0=tuple(seg_p0), v0=tuple(seg_v0),
                            gravity=g, tangent=tuple(seg_u), accel=a))
    pend = _Pending(hit=remaining[-1], kind="note", mode="roll",
                    tangent=tuple(final_u), ramp_normal=normal_for(final_dd))
    return pend, final_pos, final_v, final_t


# ---------------------------------------------------------------------------
# Main solve
# ---------------------------------------------------------------------------

def solve(hits: list[Hit], cfg_solver: dict, meta: dict | None = None) -> Trajectory:
    if not hits:
        raise SolveError("song has no notes")
    cfg = cfg_solver
    g = float(cfg["gravity"])
    ctx = _Ctx(cfg=cfg, rng=np.random.default_rng(int(cfg["seed"])))

    plan = _build_plan(hits, cfg)
    events: list[Event] = []
    segments: list[Segment] = []

    # ---- initial free-fall onto the first contact -----------------------
    v0 = float(cfg["impact_speed0"])
    t_first = _step_start_time(plan[0])
    h0 = v0 * v0 / (2.0 * g)
    spawn_t = t_first - v0 / g
    events.append(Event(time=spawn_t, kind="spawn", pos=(0.0, h0),
                        v_in=(0.0, 0.0), v_out=(0.0, 0.0), mode="bounce"))
    segments.append(Segment(kind="air", t0=spawn_t, t1=t_first,
                            p0=(0.0, h0), v0=(0.0, 0.0), gravity=g))
    p = np.array([0.0, 0.0])
    v_in = np.array([0.0, -v0])
    t = t_first

    # ---- establish the first pending contact ----------------------------
    first = plan[0]
    if first["kind"] == "run":
        pending, p, v_in, t = _solve_run(ctx, p, v_in, t, first["hits"],
                                         events, segments)
    else:
        pending = _Pending(hit=first.get("hit"),
                           kind="note" if first["kind"] == "note" else "peg",
                           mode="bounce")

    # ---- walk the remaining plan ----------------------------------------
    for idx in range(1, len(plan)):
        nxt = plan[idx]
        if nxt["kind"] == "run":
            gap = _step_start_time(nxt) - t
            preroll = min(0.4, max(0.06, 0.3 * gap))
            dt_air = gap - preroll
            v_out, p_land, v_land, e_used = _bounce_step(ctx, p, v_in, dt_air,
                                                         idx, next_mode="roll")
            _emit_pending(ctx, events, pending, t, p, v_in, v_out, e_used)
            segments.append(Segment(kind="air", t0=t, t1=t + dt_air,
                                    p0=tuple(p), v0=tuple(v_out), gravity=g))
            pending, p, v_in, t = _solve_run(ctx, p_land, v_land, t + dt_air,
                                             nxt["hits"], events, segments)
        else:
            t_next = _step_start_time(nxt)
            dt = t_next - t
            v_out, p_next, v_next, e_used = _bounce_step(ctx, p, v_in, dt, idx)
            _emit_pending(ctx, events, pending, t, p, v_in, v_out, e_used)
            segments.append(Segment(kind="air", t0=t, t1=t_next,
                                    p0=tuple(p), v0=tuple(v_out), gravity=g))
            pending = _Pending(hit=nxt.get("hit"),
                               kind="note" if nxt["kind"] == "note" else "peg",
                               mode="bounce")
            p, v_in, t = p_next, v_next, t_next

    # ---- final contact + flourish tail -----------------------------------
    e_nom = float(cfg["restitution"])
    s_out = e_nom * float(np.linalg.norm(v_in))
    side = -1.0 if p[0] > 0 else 1.0
    v_out = s_out * unit(np.array([0.35 * side, 1.0]))
    _emit_pending(ctx, events, pending, t, p, v_in, v_out, e_nom)
    segments.append(Segment(kind="air", t0=t, t1=t + 1.6,
                            p0=tuple(p), v0=tuple(v_out), gravity=g))

    meta = dict(meta or {})
    meta.update({
        "solver": {k: cfg[k] for k in ("gravity", "restitution",
                                       "restitution_min", "seed",
                                       "impact_speed0")},
        "n_notes": sum(1 for e in events if e.kind == "note"),
        "n_pegs": sum(1 for e in events if e.kind == "peg"),
        "n_reversals": sum(1 for e in events if e.kind == "reversal"),
        "warnings": ctx.warnings,
    })
    return Trajectory(events=events, segments=segments, meta=meta)


def _emit_pending(ctx: _Ctx, events: list[Event], pending: _Pending, t: float,
                  p: np.ndarray, v_in, v_out, e_used: float) -> None:
    v_in = np.asarray(v_in, dtype=float)
    v_out = np.asarray(v_out, dtype=float)
    events.append(Event(
        time=t, kind=pending.kind, pos=tuple(p),
        pitches=list(pending.hit.pitches) if pending.hit else [],
        velocity=pending.hit.velocity if pending.hit else 80,
        v_in=tuple(v_in), v_out=tuple(v_out),
        normal=tuple(bounce_normal(v_in, v_out)),
        mode=pending.mode,
        tangent=pending.tangent,
        e_used=float(e_used)))
    if pending.mode == "bounce":
        ctx.add(p, "bounce")
    # roll-mode pending contacts were already registered by _solve_run

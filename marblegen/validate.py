"""Automatic physics/layout validation of a solved trajectory.

Errors (hard failures — the CLI exits non-zero):
  * acceleration inside any segment differs from gravity (air) or from
    5/7 * g * sin(theta) along the ramp (roll)
  * position discontinuity between consecutive segments
  * a bounce GAINS energy (|v_out| > restitution * |v_in|, beyond tolerance)
  * a velocity jump at a segment boundary with no contact event to explain it
  * a note hit further than 5 ms from its MIDI onset
  * the ball leaves the horizontal play area by more than 10 cm
  * two instruments interpenetrate (closer than 60% of required spacing)

Warnings (reported, non-fatal): soft layout issues such as tighter-than-ideal
spacing, deader-than-nominal bounces from relaxed solving, out-of-band speeds.
"""

from __future__ import annotations

import numpy as np

from marblegen.notes import Hit
from marblegen.trajectory import Trajectory


def validate(traj: Trajectory, hits: list[Hit], cfg_solver: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = list(traj.meta.get("warnings", []))
    g = float(cfg_solver["gravity"])
    e_nom = float(cfg_solver["restitution"])
    e_min = float(cfg_solver["restitution_min"])
    roll_k = float(cfg_solver["roll_accel_factor"])
    x_lo, x_hi = cfg_solver["x_bounds"]

    segs = traj.segments
    events = traj.events

    # --- 1. physics inside every segment (finite differences vs closed form)
    for si, s in enumerate(segs):
        dur = s.t1 - s.t0
        if dur <= 0:
            errors.append(f"segment {si}: non-positive duration {dur:.4f}s")
            continue
        n = max(6, int(dur * 240))
        ts = np.linspace(s.t0, s.t1, n)
        pos = np.array([s.pos(t) for t in ts])
        dt = ts[1] - ts[0]
        acc = np.diff(pos, 2, axis=0) / (dt * dt)
        if s.kind == "air":
            want = np.array([0.0, -g])
            err = np.max(np.linalg.norm(acc - want, axis=1))
            if err > 1e-3 * max(1.0, g):
                errors.append(f"segment {si} (air): accel deviates from gravity "
                              f"by {err:.4f} m/s^2")
        else:
            u = np.array(s.tangent)
            sin_th = max(0.0, -float(u[1]))
            want_a = roll_k * g * sin_th
            a_tan = acc @ u
            a_norm = acc @ np.array([-u[1], u[0]])
            if np.max(np.abs(a_tan - want_a)) > 1e-3 * max(1.0, g):
                errors.append(f"segment {si} (roll): tangential accel "
                              f"{np.mean(a_tan):.3f} != 5/7*g*sin= {want_a:.3f}")
            if np.max(np.abs(a_norm)) > 1e-3 * max(1.0, g):
                errors.append(f"segment {si} (roll): off-ramp acceleration")

    # --- 2. continuity + explained velocity changes at boundaries ----------
    def events_at(t: float):
        return [e for e in events if abs(e.time - t) < 1e-6]

    for si in range(len(segs) - 1):
        a, b = segs[si], segs[si + 1]
        if abs(a.t1 - b.t0) > 1e-6:
            errors.append(f"segments {si}->{si+1}: time gap {b.t0 - a.t1:.6f}s")
            continue
        p_end, p_start = a.pos(a.t1), b.pos(b.t0)
        if float(np.linalg.norm(p_end - p_start)) > 1e-4:
            errors.append(f"segments {si}->{si+1}: position jump "
                          f"{np.linalg.norm(p_end - p_start) * 1000:.2f} mm")
        v_end, v_start = a.vel(a.t1), b.vel(b.t0)
        evs = events_at(a.t1)
        if not evs:
            if float(np.linalg.norm(v_end - v_start)) > 1e-6:
                errors.append(f"segments {si}->{si+1}: unexplained velocity "
                              f"jump (no contact event at t={a.t1:.3f})")
            continue
        ev = evs[-1]
        s_in, s_out = float(np.linalg.norm(v_end)), float(np.linalg.norm(v_start))
        if ev.e_used is not None:                     # a bounce
            if s_out > e_nom * s_in * 1.01 + 1e-9:
                errors.append(f"bounce at t={ev.time:.3f}: energy gained "
                              f"(|v_out|/|v_in| = {s_out / max(s_in, 1e-9):.3f} "
                              f"> e = {e_nom:.2f})")
            if s_out < e_min * s_in * 0.9:
                warnings.append(f"bounce at t={ev.time:.3f}: very dead bounce "
                                f"(ratio {s_out / max(s_in, 1e-9):.2f})")
            if ev.normal is not None:
                n = np.array(ev.normal)
                if float(np.dot(v_end, n)) > 1e-6:
                    warnings.append(f"bounce at t={ev.time:.3f}: incoming "
                                    f"velocity not into the surface")
                if float(np.dot(v_start, n)) < -1e-6:
                    warnings.append(f"bounce at t={ev.time:.3f}: outgoing "
                                    f"velocity not away from the surface")
        else:                                         # a landing (roll capture)
            if s_out > s_in * (1.0 + 1e-6):
                errors.append(f"landing at t={ev.time:.3f}: energy gained "
                              f"({s_out:.3f} > {s_in:.3f} m/s)")

    # --- 3. every MIDI hit has an exactly-timed strike ----------------------
    note_times = sorted(e.time for e in events if e.kind == "note")
    for h in hits:
        if not any(abs(t - h.time) <= 0.005 for t in note_times):
            errors.append(f"note at t={h.time:.3f} has no strike within 5 ms")
    n_extra = len(note_times) - len(hits)
    if n_extra != 0:
        errors.append(f"strike count mismatch: {len(note_times)} strikes for "
                      f"{len(hits)} hits")

    # --- 4. bounds -----------------------------------------------------------
    for e in events:
        if e.kind == "spawn":
            continue
        if e.pos[0] < x_lo - 0.10 or e.pos[0] > x_hi + 0.10:
            errors.append(f"{e.kind} at t={e.time:.3f}: x={e.pos[0]:.2f} "
                          f"outside play area")
        elif e.pos[0] < x_lo - 0.005 or e.pos[0] > x_hi + 0.005:
            warnings.append(f"{e.kind} at t={e.time:.3f}: x={e.pos[0]:.2f} "
                            f"grazes play-area edge")

    # --- 5. interpenetration --------------------------------------------------
    placed = [(np.array(e.pos), e.mode, e.time) for e in events
              if e.kind in ("note", "peg")]
    min_sp = float(cfg_solver["ramp_min_spacing"])
    instr_cl = float(cfg_solver["instrument_clearance"])
    worst = None
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            (pa, ma, ta), (pb, mb, tb) = placed[i], placed[j]
            if abs(pa[1] - pb[1]) > 0.6:
                continue
            need = instr_cl if (ma == "bounce" and mb == "bounce") else \
                0.98 * min_sp
            d = float(np.linalg.norm(pa - pb))
            if d < 0.6 * need:
                errors.append(f"instruments at t={ta:.2f} and t={tb:.2f} "
                              f"interpenetrate ({d * 100:.1f} cm apart)")
            elif d < need and (worst is None or d < worst):
                worst = d
                warnings.append(f"instruments at t={ta:.2f}/{tb:.2f} are close "
                                f"({d * 100:.1f} cm)")

    # --- 6. stats --------------------------------------------------------------
    speeds = [float(np.linalg.norm(e.v_in)) for e in events
              if e.kind in ("note", "peg", "reversal")]
    stats = {
        "notes": sum(1 for e in events if e.kind == "note"),
        "pegs": sum(1 for e in events if e.kind == "peg"),
        "reversals": sum(1 for e in events if e.kind == "reversal"),
        "duration_s": round(segs[-1].t1 - segs[0].t0, 2),
        "descent_m": round(float(segs[0].pos(segs[0].t0)[1] -
                                 segs[-1].pos(segs[-1].t1)[1]), 2),
        "speed_min": round(min(speeds), 2) if speeds else None,
        "speed_max": round(max(speeds), 2) if speeds else None,
    }
    if speeds and max(speeds) > float(cfg_solver["max_speed"]) * 1.05:
        warnings.append(f"peak speed {max(speeds):.1f} m/s above configured cap")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "stats": stats}


def format_report(name: str, report: dict) -> str:
    s = report["stats"]
    lines = [f"— solver report: {name} "
             f"[{'PASS' if report['ok'] else 'FAIL'}]",
             f"  notes={s['notes']} pegs={s['pegs']} reversals={s['reversals']} "
             f"duration={s['duration_s']}s descent={s['descent_m']}m "
             f"impact speeds {s['speed_min']}–{s['speed_max']} m/s"]
    for e in report["errors"]:
        lines.append(f"  ERROR   {e}")
    shown = report["warnings"][:8]
    for w in shown:
        lines.append(f"  warning {w}")
    if len(report["warnings"]) > len(shown):
        lines.append(f"  … and {len(report['warnings']) - len(shown)} more warnings")
    return "\n".join(lines)

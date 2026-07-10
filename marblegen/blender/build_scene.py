"""Blender headless scene builder + renderer for marblegen.

Runs INSIDE Blender (3.6+ or 4.x):

  blender --background --factory-startup --python build_scene.py -- \
      --traj traj.json --scene scene.json --out video.mp4

Reads the solved trajectory JSON (analytic segments already baked to
per-frame ball/camera samples by `marblegen`) plus a scene config carrying
the theme and precomputed per-event instrument visuals. Builds instruments,
wall, lights and camera, keyframes the ball and per-hit wobbles, and renders
a silent H.264 MP4 (the CLI muxes the audio afterwards).

Solver coordinates (x right, y up, 2D wall plane) map to Blender as
(x, 0, y); the camera sits at negative Y looking toward +Y.
"""

import argparse
import json
import math
import sys

import bpy
from mathutils import Euler, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb_to_linear(c):
    return tuple(((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
                 for v in c)


def make_material(name, rgb, roughness=0.35, metallic=0.0, transmission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(rgb), 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    for key in ("Transmission Weight", "Transmission"):  # 4.x vs 3.x name
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = transmission
            break
    return mat


def look_of(style, key, default):
    v = style.get(key, default)
    return hex_to_rgb(v) if isinstance(v, str) else tuple(v[:3])


def main():
    args = parse_args()
    with open(args.traj) as f:
        traj = json.load(f)
    with open(args.scene) as f:
        scene_cfg = json.load(f)
    theme = scene_cfg["theme"]
    style = theme.get("style", {})
    rc = scene_cfg["render"]
    visuals = scene_cfg["instrument_visuals"]

    fps = int(traj["ball"]["fps"])
    ball_pos = traj["ball"]["positions"]
    cam_pos = traj["camera"]["positions"]
    video_t0 = float(traj["ball"]["t0"])
    n_frames = len(ball_pos)
    events = traj["events"]

    # ---- fresh scene -----------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.resolution_x = int(rc.get("width", 1080))
    scn.render.resolution_y = int(rc.get("height", 1920))
    scn.render.fps = fps
    scn.frame_start = 1
    scn.frame_end = n_frames
    try:
        scn.render.engine = "BLENDER_EEVEE_NEXT"      # Blender 4.2+
    except TypeError:
        scn.render.engine = "BLENDER_EEVEE"
    if hasattr(scn, "eevee"):
        for attr, val in (("use_gtao", True), ("use_raytracing", True),
                          ("use_motion_blur", True), ("taa_render_samples", 32)):
            if hasattr(scn.eevee, attr):
                setattr(scn.eevee, attr, val)

    # ---- world / wall ----------------------------------------------------
    bg = look_of(style, "background", "#efeee9")
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = \
        (*srgb_to_linear(bg), 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    scn.world = world

    ys = [p[1] for p in ball_pos]
    y_top, y_bot = max(ys) + 3.0, min(ys) - 3.0
    if style.get("background_grid"):
        bpy.ops.mesh.primitive_plane_add(size=1)
        wall = bpy.context.object
        wall.scale = (8.0, (y_top - y_bot), 1.0)
        wall.rotation_euler = Euler((math.pi / 2, 0, 0))
        wall.location = (0, 0.09, (y_top + y_bot) / 2)
        mat = bpy.data.materials.new("wall")
        mat.use_nodes = True
        nt = mat.node_tree
        brick = nt.nodes.new("ShaderNodeTexBrick")
        brick.offset = 0.0
        sp = float(style.get("grid_spacing_m", 0.42))
        brick.inputs["Scale"].default_value = 1.0 / sp
        brick.inputs["Color1"].default_value = (*srgb_to_linear(bg), 1.0)
        brick.inputs["Color2"].default_value = (*srgb_to_linear(bg), 1.0)
        mortar = look_of(style, "background_grid", "#d8d5cf")
        brick.inputs["Mortar"].default_value = (*srgb_to_linear(mortar), 1.0)
        brick.inputs["Mortar Size"].default_value = 0.004
        tex = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tex.outputs["Object"], brick.inputs["Vector"])
        bsdf = nt.nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = 0.7
        nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
        wall.data.materials.append(mat)

    # ---- lights ------------------------------------------------------------
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(15)
    sun.rotation_euler = Euler((math.radians(55), math.radians(-15), 0))
    scn.collection.objects.link(sun)
    key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
    key.data.energy = 900.0
    key.data.size = 6.0
    key.location = (1.5, -3.5, (y_top + y_bot) / 2 + 1.0)
    key.rotation_euler = Euler((math.pi / 2, 0, math.radians(20)))
    scn.collection.objects.link(key)

    # ---- instruments --------------------------------------------------------
    is_tube = style.get("instrument", "tube") == "tube"
    tube_r = float(style.get("tube_radius_m", 0.03))
    pad_t = float(style.get("paddle_thickness_m", 0.028))
    bracket_rgb = look_of(style, "bracket_color", "#9aa0a6")
    peg_rgb = look_of(style, "peg_color", "#b9bec4")
    bracket_mat = make_material("bracket", bracket_rgb, roughness=0.25,
                                metallic=0.9)
    peg_mat = make_material("peg", peg_rgb, roughness=0.4, metallic=0.6)
    wobble_deg = float(style.get("wobble_deg", 10.0))
    wobble_decay = float(style.get("wobble_decay_s", 0.3))

    def add_bracket(pos3, into_wall=0.10):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=into_wall,
                                            vertices=12)
        b = bpy.context.object
        b.rotation_euler = Euler((math.pi / 2, 0, 0))
        b.location = (pos3[0], pos3[1] + into_wall / 2, pos3[2])
        b.data.materials.append(bracket_mat)
        return b

    def keyframe_wobble(obj, hit_frames, axis_index):
        base = list(obj.rotation_euler)
        for hf in hit_frames:
            amp = math.radians(wobble_deg)
            times = [0.0, 0.06, 0.14, 0.22, 0.34, 0.5]
            amps = [0.0, 1.0, -0.55, 0.3, -0.12, 0.0]
            for dt, k in zip(times, amps):
                rot = list(base)
                rot[axis_index] = base[axis_index] + amp * k * \
                    math.exp(-dt / max(wobble_decay, 1e-3))
                obj.rotation_euler = Euler(rot)
                obj.keyframe_insert("rotation_euler",
                                    frame=hf + int(dt * fps))
        obj.rotation_euler = Euler(base)

    for i, ev in enumerate(events):
        kind = ev["kind"]
        if kind not in ("note", "peg", "roll_land"):
            continue
        x, y = ev["pos"]
        nx, ny = ev["normal"] if ev["normal"] else (0.0, 1.0)
        hit_frame = 1 + int(round((ev["time"] - video_t0) * fps))
        vis = visuals[i] if i < len(visuals) else None

        if kind != "note":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.022, depth=0.05,
                                                vertices=24)
            peg = bpy.context.object
            peg.rotation_euler = Euler((math.pi / 2, 0, 0))
            peg.location = (x, -0.02, y - 0.012)
            peg.data.materials.append(peg_mat)
            add_bracket((x, 0.0, y - 0.012))
            continue

        rgb = vis["color"] if vis else (0.8, 0.2, 0.2)
        length = vis["length"] if vis else 0.2
        mat = make_material(f"instr_{i}", rgb, roughness=0.32)
        angle = math.atan2(ny, nx)          # normal direction in wall plane

        if is_tube:
            # tube pointing out of the wall, its rim tangent to the contact
            bpy.ops.mesh.primitive_cylinder_add(radius=tube_r, depth=length,
                                                vertices=36)
            obj = bpy.context.object
            # axis: mostly -Y (toward camera), drooping slightly downward
            droop = math.radians(18)
            obj.rotation_euler = Euler((math.pi / 2 - droop, 0, 0))
            cx = x - nx * tube_r
            cy = y - ny * tube_r
            obj.location = (cx, -length / 2 * math.cos(droop) + 0.02,
                            cy - length / 2 * math.sin(droop))
            add_bracket((cx, 0.0, cy - length * 0.45))
            wob_axis = 0
        else:
            # paddle: face normal along the contact normal, in the wall plane
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.object
            obj.scale = (length, 2.2 * pad_t, pad_t)
            mod = obj.modifiers.new("bevel", "BEVEL")
            mod.width = 0.008
            mod.segments = 3
            obj.rotation_euler = Euler((0, 0, angle - math.pi / 2))
            obj.location = (x - nx * pad_t / 2, 0, y - ny * pad_t / 2)
            add_bracket((x - nx * 0.03, 0.0, y - ny * 0.03))
            wob_axis = 1
        obj.data.materials.append(mat)
        keyframe_wobble(obj, [hit_frame], wob_axis)

    # ---- ball ---------------------------------------------------------------
    ball_r = float(style.get("ball_radius_m", 0.032))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_r, segments=48,
                                         ring_count=24)
    ball = bpy.context.object
    bpy.ops.object.shade_smooth()
    ball_rgb = look_of(style, "ball_color", "#dfe8f0")
    ball.data.materials.append(
        make_material("ball", ball_rgb, roughness=0.05, transmission=0.7))
    action_frames = range(1, n_frames + 1)
    for f, (bx, by) in zip(action_frames, ball_pos):
        ball.location = (bx, -0.0, by + ball_r)
        ball.keyframe_insert("location", frame=f)

    # ---- camera ---------------------------------------------------------------
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 40
    cam = bpy.data.objects.new("cam", cam_data)
    scn.collection.objects.link(cam)
    scn.camera = cam
    cam.rotation_euler = Euler((math.pi / 2, 0, 0))   # look toward +Y
    view_h = float(rc.get("view_height_m", 2.3))
    dist = view_h / (2 * math.tan(0.5 * 2 *
                                  math.atan(cam_data.sensor_width / (2 * cam_data.lens))))
    for f, (cx, cy) in zip(action_frames, cam_pos):
        cam.location = (cx, -dist - 0.4, cy)
        cam.keyframe_insert("location", frame=f)

    # linear interpolation for all animation (samples are already per-frame)
    for obj in (ball, cam):
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

    # ---- output -----------------------------------------------------------------
    scn.render.image_settings.file_format = "FFMPEG"
    scn.render.ffmpeg.format = "MPEG4"
    scn.render.ffmpeg.codec = "H264"
    scn.render.ffmpeg.constant_rate_factor = "HIGH"
    scn.render.ffmpeg.ffmpeg_preset = "GOOD"
    scn.render.filepath = args.out
    scn.render.use_motion_blur = True
    bpy.ops.render.render(animation=True)


main()

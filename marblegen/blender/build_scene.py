"""Blender headless scene builder + renderer for marblegen.

Runs inside Blender two ways:

  blender --background --factory-startup --python build_scene.py -- \
      --traj traj.json --scene scene.json --out video.mp4
  python3 build_scene.py --traj traj.json --scene scene.json --out video.mp4
      (requires `pip install bpy`; works headless via Mesa/llvmpipe)

Reads the solved trajectory JSON (analytic segments already baked to
per-frame ball/camera samples by `marblegen`) plus a scene config carrying
the theme and precomputed per-event instrument visuals. Builds instruments,
wall, lights and camera, keyframes every marble and per-hit wobbles, and
renders a silent H.264 MP4 (the CLI muxes the audio afterwards).

Solver coordinates (x right, y up, 2D wall plane) map to Blender as
(x, 0, y); the camera sits at negative Y looking toward +Y.
"""

import argparse
import json
import math
import sys

import bpy
from mathutils import Euler


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv \
        else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resolution scale (1.0 = full 1080x1920)")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--still", type=int, default=None,
                    help="render only this frame as PNG (look-dev)")
    ap.add_argument("--frame-end", type=int, default=None,
                    help="stop after this frame (debugging)")
    return ap.parse_args(argv)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb_to_linear(c):
    return tuple(((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
                 for v in c)


def set_input(bsdf, names, value):
    for name in names:
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value
            return


def make_material(name, rgb, roughness=0.35, metallic=0.0, transmission=0.0,
                  ior=1.45, clearcoat=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(rgb), 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    set_input(bsdf, ("Transmission Weight", "Transmission"), transmission)
    set_input(bsdf, ("IOR",), ior)
    if clearcoat:
        set_input(bsdf, ("Coat Weight", "Clearcoat"), clearcoat)
        set_input(bsdf, ("Coat Roughness", "Clearcoat Roughness"), 0.08)
    return mat


def _hash01(*ints):
    """Deterministic pseudo-random in [0,1) from integers (no RNG state)."""
    h = 2166136261
    for v in ints:
        h = (h ^ (int(v) & 0xFFFFFFFF)) * 16777619 & 0xFFFFFFFF
    return (h % 100000) / 100000.0


def look_of(style, key, default):
    v = style.get(key, default)
    return hex_to_rgb(v) if isinstance(v, str) else tuple(v[:3])


def shade_smooth(obj):
    for poly in obj.data.polygons:
        poly.use_smooth = True


def shade_smooth_sides(obj):
    """Smooth only the side faces of a cylinder — flat caps stay flat, so
    drum heads and tube mouths don't render as inflated domes."""
    for poly in obj.data.polygons:
        poly.use_smooth = abs(poly.normal.z) < 0.5


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

    balls = traj["balls"]
    fps = int(balls[0]["fps"])
    video_t0 = float(balls[0]["t0"])
    n_frames = len(balls[0]["positions"])
    cam_pos = traj["camera"]["positions"]
    events = traj["events"]

    def frame_of(t):
        return 1 + int(round((t - video_t0) * fps))

    # ---- fresh scene -----------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.resolution_x = max(2, int(int(rc.get("width", 1080))
                                         * args.scale) // 2 * 2)
    scn.render.resolution_y = max(2, int(int(rc.get("height", 1920))
                                         * args.scale) // 2 * 2)
    scn.render.fps = fps
    scn.frame_start = 1
    scn.frame_end = n_frames
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scn.render.engine = engine
            break
        except TypeError:
            continue
    ev = getattr(scn, "eevee", None)
    if ev is not None:
        for attr, val in (("taa_render_samples", args.samples),
                          ("use_gtao", True),
                          ("use_shadows", True),
                          ("use_raytracing", False)):
            if hasattr(ev, attr):
                setattr(ev, attr, val)
    # AgX (the default view transform) heavily desaturates the toy-bright
    # palette; Standard keeps the boomwhacker colours vivid.
    try:
        scn.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    scn.view_settings.exposure = float(rc.get("blender_exposure", -0.9))

    # ---- world -----------------------------------------------------------
    bg = look_of(style, "background", "#efeee9")
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = \
        (*srgb_to_linear(bg), 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
    scn.world = world

    all_y = [p[1] for b in balls for p in b["positions"]]
    y_top, y_bot = max(all_y) + 3.0, min(all_y) - 3.0

    # ---- wall with tile grout --------------------------------------------
    bpy.ops.mesh.primitive_plane_add(size=1)
    wall = bpy.context.object
    wall.scale = (9.0, (y_top - y_bot), 1.0)
    wall.rotation_euler = Euler((math.pi / 2, 0, 0))
    wall.location = (0, 0.10, (y_top + y_bot) / 2)
    mat = bpy.data.materials.new("wall")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.65
    if style.get("background_grid"):
        brick = nt.nodes.new("ShaderNodeTexBrick")
        sp = float(style.get("grid_spacing_m", 0.42))
        brick.offset = 0.0
        brick.inputs["Scale"].default_value = 1.0
        brick.inputs["Color1"].default_value = (*srgb_to_linear(bg), 1.0)
        brick.inputs["Color2"].default_value = (*srgb_to_linear(bg), 1.0)
        mortar = look_of(style, "background_grid", "#d8d5cf")
        brick.inputs["Mortar"].default_value = (*srgb_to_linear(mortar), 1.0)
        brick.inputs["Mortar Size"].default_value = 0.006 / sp
        brick.inputs["Brick Width"].default_value = 1.0
        brick.inputs["Row Height"].default_value = 1.0
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (9.0 / sp,
                                                 (y_top - y_bot) / sp, 1.0)
        tex = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tex.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], brick.inputs["Vector"])
        nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(bg), 1.0)
    wall.data.materials.append(mat)

    # ---- lights ------------------------------------------------------------
    # soft studio: a gentle directional key for shape + a big area wash. The
    # sun angle sets shadow softness (the reference look has large, soft,
    # down-left shadows).
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 1.0
    sun.data.angle = math.radians(9)
    sun.rotation_euler = Euler((math.radians(56), math.radians(-22), 0))
    scn.collection.objects.link(sun)
    key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
    key.data.energy = 300.0
    key.data.size = 12.0
    key.location = (1.4, -4.2, (y_top + y_bot) / 2 + 1.5)
    key.rotation_euler = Euler((math.pi / 2, 0, math.radians(16)))
    scn.collection.objects.link(key)

    # ---- instruments --------------------------------------------------------
    instr_kind = style.get("instrument", "tube")
    tube_r = float(style.get("tube_radius_m", 0.03))
    pad_t = float(style.get("paddle_thickness_m", 0.028))
    bracket_mat = make_material("bracket",
                                look_of(style, "bracket_color", "#9aa0a6"),
                                roughness=0.25, metallic=0.9)
    peg_mat = make_material("peg", look_of(style, "peg_color", "#b9bec4"),
                            roughness=0.35, metallic=0.7)
    wood_mat = make_material("wood", look_of(style, "wood_color", "#8a5a3b"),
                             roughness=0.5, clearcoat=0.3)
    head_mat = make_material("head", look_of(style, "head_color", "#f2efe8"),
                             roughness=0.25, clearcoat=0.5)
    wobble_deg = float(style.get("wobble_deg", 10.0))
    wobble_decay = float(style.get("wobble_decay_s", 0.3))

    def add_bracket(x, z, depth=0.10):
        """Metal rod anchored in the wall (y=0.09), reaching toward the
        instrument, with a ball joint on its free end."""
        bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=depth,
                                            vertices=12)
        b = bpy.context.object
        b.rotation_euler = Euler((math.pi / 2, 0, 0))
        b.location = (x, 0.088 - depth / 2, z)
        b.data.materials.append(bracket_mat)
        shade_smooth(b)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.011, segments=16,
                                             ring_count=8)
        joint = bpy.context.object
        joint.location = (x, 0.088 - depth, z)
        joint.data.materials.append(bracket_mat)
        shade_smooth(joint)
        return b

    def keyframe_wobble(obj, hit_frames, axis_index):
        base = list(obj.rotation_euler)
        obj.rotation_euler = Euler(base)
        obj.keyframe_insert("rotation_euler", frame=1)
        amp = math.radians(wobble_deg)
        for hf in hit_frames:
            for dt, k in ((0.0, 0.0), (0.05, 1.0), (0.13, -0.55),
                          (0.21, 0.30), (0.33, -0.12), (0.5, 0.0)):
                rot = list(base)
                rot[axis_index] = base[axis_index] + amp * k * \
                    math.exp(-dt / max(wobble_decay, 1e-3))
                obj.rotation_euler = Euler(rot)
                obj.keyframe_insert("rotation_euler",
                                    frame=hf + max(1, int(dt * fps)))
        obj.rotation_euler = Euler(base)

    for i, evd in enumerate(events):
        kind = evd["kind"]
        if kind not in ("note", "peg", "roll_land"):
            continue
        x, y = evd["pos"]
        nx, ny = evd["normal"] if evd["normal"] else (0.0, 1.0)
        hit_frame = frame_of(evd["time"])
        vis = visuals[i] if i < len(visuals) else None

        if kind != "note":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, segments=24,
                                                 ring_count=12)
            peg = bpy.context.object
            peg.location = (x, 0.0, y - 0.008)
            peg.data.materials.append(peg_mat)
            shade_smooth(peg)
            add_bracket(x, y - 0.008)
            continue

        rgb = vis["color"] if vis else (0.8, 0.2, 0.2)
        length = vis["length"] if vis else 0.2
        is_roll = evd["mode"] == "roll"
        # per-instrument variation makes the wall read as hand-placed 3D
        # objects instead of a flat pattern
        var_a = _hash01(i, 11)
        var_b = _hash01(i, 29)

        if instr_kind == "tube":
            mat = make_material(f"instr_{i}", rgb,
                                roughness=0.24 + 0.10 * var_b, clearcoat=0.6)
            bpy.ops.mesh.primitive_cylinder_add(radius=tube_r, depth=length,
                                                vertices=32)
            obj = bpy.context.object
            shade_smooth_sides(obj)
            bev = obj.modifiers.new("bevel", "BEVEL")
            bev.width = 0.004
            bev.segments = 2
            if is_roll:
                # hang along the negative contact normal (below the roll line)
                ang = math.atan2(ny, nx)
                obj.rotation_euler = Euler((0, ang + math.pi / 2, 0))
                cx = x - nx * (length / 2 + 0.004)
                cz = y - ny * (length / 2 + 0.004)
                obj.location = (cx, 0.0, cz)
                add_bracket(cx - nx * length * 0.3, cz - ny * length * 0.3)
                wob_axis = 1
            else:
                # lie in the wall plane, perpendicular to the contact normal,
                # tilted out of the wall so the mouth faces the camera; each
                # tube gets its own tilt so the wall has depth variety
                ang = math.atan2(-nx, ny)     # axis = perp(normal)
                out_tilt = math.radians(14 + 22 * var_a)
                obj.rotation_euler = Euler((out_tilt, ang + math.pi / 2, 0),
                                           "ZYX")
                obj.location = (x - nx * tube_r, length / 2 *
                                math.sin(out_tilt) * 0.4, y - ny * tube_r)
                add_bracket(x - nx * tube_r, y - ny * tube_r - 0.02,
                            depth=0.08)
                wob_axis = 0
            obj.data.materials.append(mat)
        elif instr_kind == "drum":
            # round drum/banjo head on a metal arm: wood body + bright skin,
            # tilted out of the wall by a per-instrument amount
            radius = max(0.055, length * 0.42)
            # tilt azimuth wanders around the contact normal so heads face
            # left/right/up with hand-placed variety, like the reference
            ang = math.atan2(ny, nx) + (var_b - 0.5) * 1.3
            out_tilt = math.radians(16 + 26 * var_a)
            # after this rotation the drum's local +Z cap faces the camera,
            # nodding `out_tilt` upward, twisted so the tilt follows the
            # contact normal direction
            rot = Euler((math.pi / 2 - out_tilt, 0, ang - math.pi / 2), "ZYX")
            bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=0.045,
                                                vertices=40)
            obj = bpy.context.object
            shade_smooth_sides(obj)
            bev = obj.modifiers.new("bevel", "BEVEL")
            bev.width = 0.006
            bev.segments = 2
            obj.rotation_euler = rot
            obj.location = (x - nx * radius * 0.9, 0.02, y - ny * radius * 0.9)
            obj.data.materials.append(wood_mat)
            # drum skin: thin bright disc sitting proud of the front cap,
            # placed in the body's local space so wobble carries it along
            bpy.ops.mesh.primitive_cylinder_add(radius=radius * 0.85,
                                                depth=0.012, vertices=40)
            head = bpy.context.object
            shade_smooth_sides(head)
            head.parent = obj
            head.location = (0.0, 0.0, 0.026)
            head.rotation_euler = Euler((0, 0, 0))
            head.data.materials.append(head_mat)
            add_bracket(x - nx * radius, y - ny * radius, depth=0.075)
            wob_axis = 0
        else:  # paddle
            mat = make_material(f"instr_{i}", rgb,
                                roughness=0.26 + 0.08 * var_b, clearcoat=0.5)
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.object
            obj.scale = (length, 2.4 * pad_t, pad_t)
            bev = obj.modifiers.new("bevel", "BEVEL")
            bev.width = 0.008
            bev.segments = 3
            ang = math.atan2(ny, nx)
            roll_var = math.radians((var_a - 0.5) * 14)
            obj.rotation_euler = Euler((roll_var, 0, ang - math.pi / 2), "ZYX")
            obj.location = (x - nx * pad_t / 2, 0, y - ny * pad_t / 2)
            add_bracket(x - nx * 0.04, y - ny * 0.04, depth=0.14)
            obj.data.materials.append(mat)
            wob_axis = 1
        keyframe_wobble(obj, [hit_frame], wob_axis)

    # ---- marbles -------------------------------------------------------------
    ball_r = float(style.get("ball_radius_m", 0.04))
    # alpha-blended glass: reads as transparent without needing raytraced
    # transmission (which software-GL Eevee can't do)
    tint = look_of(style, "ball_color", "#b3c4de")
    glass = make_material("marble", tint, roughness=0.04, clearcoat=0.8)
    gb = glass.node_tree.nodes["Principled BSDF"]
    set_input(gb, ("Alpha",), 0.55)
    for attr, val in (("surface_render_method", "BLENDED"),
                      ("blend_method", "BLEND")):
        if hasattr(glass, attr):
            try:
                setattr(glass, attr, val)
                break
            except TypeError:
                continue
    bead_sets = [[(0.90, 0.42, 0.44), (0.43, 0.62, 0.77), (0.95, 0.76, 0.31)],
                 [(0.50, 0.69, 0.41), (0.71, 0.43, 0.77), (0.95, 0.56, 0.17)],
                 [(0.36, 0.75, 0.75), (0.94, 0.46, 0.48), (0.97, 0.70, 0.17)]]
    for b in balls:
        track = int(b["track"])
        pos = b["positions"]
        start_f = frame_of(float(b["start"])) - int(0.1 * fps)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_r, segments=40,
                                             ring_count=20)
        ball = bpy.context.object
        shade_smooth(ball)
        ball.data.materials.append(glass)
        beads = []
        for k, col in enumerate(bead_sets[track % len(bead_sets)]):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_r * 0.30,
                                                 segments=20, ring_count=10)
            bead = bpy.context.object
            shade_smooth(bead)
            bead.data.materials.append(
                make_material(f"bead{track}_{k}", col, roughness=0.4))
            beads.append(bead)
        dist = 0.0
        prev = pos[0]
        for f in range(1, n_frames + 1):
            bx, by = pos[min(f - 1, len(pos) - 1)]
            dist += math.hypot(bx - prev[0], by - prev[1])
            prev = (bx, by)
            ball.location = (bx, 0.0, by)
            ball.keyframe_insert("location", frame=f)
            roll = dist / max(ball_r, 1e-6) * 0.6
            for k, bead in enumerate(beads):
                a = roll + k * 2.1
                bead.location = (bx + 0.40 * ball_r * math.cos(a),
                                 0.0,
                                 by + 0.40 * ball_r * math.sin(a))
                bead.keyframe_insert("location", frame=f)
        for obj in (ball, *beads):
            for prop in ("hide_render", "hide_viewport"):
                obj.hide_render = start_f > 1
                setattr(obj, prop, start_f > 1)
                obj.keyframe_insert(prop, frame=1)
                setattr(obj, prop, False)
                obj.keyframe_insert(prop, frame=max(1, start_f))

    # ---- camera ---------------------------------------------------------------
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 60                      # long-ish lens, flat perspective
    cam_data.sensor_fit = "HORIZONTAL"      # sensor_width maps to res_x even
    if hasattr(cam_data, "dof"):            # in portrait orientation
        cam_data.dof.use_dof = True         # subtle depth falloff on tilted
        cam_data.dof.aperture_fstop = 2.8   # instruments sells the 3D
    cam = bpy.data.objects.new("cam", cam_data)
    scn.collection.objects.link(cam)
    scn.camera = cam
    cam.rotation_euler = Euler((math.pi / 2, 0, 0))   # look toward +Y
    view_h = float(rc.get("view_height_m", 2.3))
    view_w = view_h * scn.render.resolution_x / scn.render.resolution_y
    # distance so the horizontal extent equals view_w:
    # d = view_w * focal / sensor_width (sensor units in mm)
    dist = view_w * cam_data.lens / cam_data.sensor_width
    if hasattr(cam_data, "dof"):
        cam_data.dof.focus_distance = dist
    for f in range(1, n_frames + 1):
        cx, cy = cam_pos[min(f - 1, len(cam_pos) - 1)]
        cam.location = (cx, -dist, cy)
        cam.keyframe_insert("location", frame=f)

    # linear interpolation everywhere (samples are already per-frame)
    def action_fcurves(action):
        if hasattr(action, "fcurves"):          # Blender <= 4.x
            return list(action.fcurves)
        curves = []                              # Blender 5.x layered actions
        for layer in getattr(action, "layers", []):
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", []):
                    curves.extend(bag.fcurves)
        return curves

    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fc in action_fcurves(obj.animation_data.action):
                if fc.data_path.endswith("location"):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"

    # ---- output -----------------------------------------------------------------
    if args.still is not None:
        scn.frame_set(args.still)
        scn.render.image_settings.file_format = "PNG"
        scn.render.filepath = args.out
        bpy.ops.render.render(write_still=True)
        return
    if args.frame_end:
        scn.frame_end = min(scn.frame_end, args.frame_end)
    # motion blur costs several sub-frame passes; at 60 fps crisp frames read
    # fine, so trade it for wall-clock (flip on for hero renders on a GPU)
    if hasattr(scn.render, "use_motion_blur"):
        scn.render.use_motion_blur = False
    try:
        # full Blender installs can encode the movie directly …
        scn.render.image_settings.file_format = "FFMPEG"
        scn.render.ffmpeg.format = "MPEG4"
        scn.render.ffmpeg.codec = "H264"
        scn.render.ffmpeg.constant_rate_factor = "HIGH"
        scn.render.ffmpeg.ffmpeg_preset = "GOOD"
        scn.render.filepath = args.out
    except TypeError:
        # … the PyPI bpy wheel ships without ffmpeg output: render a PNG
        # sequence instead; the marblegen CLI encodes it afterwards.
        scn.render.image_settings.file_format = "PNG"
        scn.render.filepath = args.out + "_frames/"
    bpy.ops.render.render(animation=True)


main()

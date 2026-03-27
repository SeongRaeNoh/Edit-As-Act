#!/usr/bin/env python3

import json
import copy
import math
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  ID / object lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_id_maps(scene: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build bidirectional maps between EMPTY names and semantic IDs.

    EMPTY name  "{index}_{category}"  <->  semantic ID  "{category}_{index}"
    e.g.        "000_wall_tvs"        <->  "wall_tvs_000"

    Returns:
        name_to_sem: EMPTY name  -> semantic ID
        sem_to_name: semantic ID -> EMPTY name
    """
    name_to_sem: Dict[str, str] = {}
    sem_to_name: Dict[str, str] = {}

    for obj in scene["objects"]:
        if obj["type"] != "EMPTY":
            continue
        name = obj["name"]
        if name == "RoomContainer":
            name_to_sem[name] = "RoomContainer"
            sem_to_name["RoomContainer"] = name
            continue
        parts = name.split("_")
        if len(parts) >= 2 and parts[0].isdigit():
            idx = parts[0]
            category = "_".join(parts[1:])
            sem_id = f"{category}_{idx}"
            name_to_sem[name] = sem_id
            sem_to_name[sem_id] = name

    return name_to_sem, sem_to_name


def _find_empty(scene: Dict[str, Any], name: str) -> Optional[Dict]:
    """Return the EMPTY object dict with the given name, or None."""
    for obj in scene["objects"]:
        if obj["type"] == "EMPTY" and obj["name"] == name:
            return obj
    return None


def _find_mesh_child(scene: Dict[str, Any], parent_name: str) -> Optional[Dict]:
    """Return the first MESH whose parent matches parent_name, or None."""
    for obj in scene["objects"]:
        if obj["type"] == "MESH" and obj.get("parent") == parent_name:
            return obj
    return None


def _resolve(sem_id: str, sem_to_name: Dict[str, str]) -> Optional[str]:
    """Resolve semantic ID to EMPTY name, or None."""
    return sem_to_name.get(sem_id)


def _next_name_index(scene: Dict[str, Any]) -> int:
    """Return the next available numeric index for a new EMPTY name."""
    max_idx = -1
    for obj in scene["objects"]:
        if obj["type"] == "EMPTY":
            parts = obj["name"].split("_")
            if parts and parts[0].isdigit():
                max_idx = max(max_idx, int(parts[0]))
    return max_idx + 1


# ─────────────────────────────────────────────────────────────────────────────
#  matrix_world / quaternion recomputation
# ─────────────────────────────────────────────────────────────────────────────

def _sync_quaternion(obj: Dict) -> None:
    """Keep rotation_quaternion consistent with rotation_euler[2] (yaw around Z).

    Uses the half-angle formula for a pure Z-axis rotation:
        q = [cos(θ/2), 0, 0, sin(θ/2)]  →  [w, x, y, z]
    Only applied when rotation_mode == "XYZ".
    """
    if obj.get("rotation_mode") != "XYZ":
        return
    yaw = obj["rotation_euler"][2]
    half = yaw / 2.0
    obj["rotation_quaternion"] = [math.cos(half), 0.0, 0.0, math.sin(half)]


def _recompute_matrix_world(obj: Dict) -> None:
    """Recompute matrix_world from location, rotation_euler[2], and scale.

    The matrix encodes a rotation around the Z-axis (yaw) followed by
    a non-uniform scale, with translation appended:

        | cos(yaw)*sx  -sin(yaw)*sy  0   tx |
        | sin(yaw)*sx   cos(yaw)*sy  0   ty |
        |     0              0       sz  tz |
        |     0              0       0   1  |
    """
    tx, ty, tz = obj["location"]
    yaw = obj["rotation_euler"][2]
    sx, sy, sz = obj["scale"]

    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    obj["matrix_world"] = [
        [cos_y * sx, -sin_y * sy, 0.0, tx],
        [sin_y * sx,  cos_y * sy, 0.0, ty],
        [0.0,         0.0,        sz,  tz],
        [0.0,         0.0,        0.0, 1.0],
    ]
    _sync_quaternion(obj)


def _sync_mesh_child(scene: Dict, empty_name: str) -> None:
    """Copy the EMPTY's matrix_world to its MESH child (child has identity local transform)."""
    empty = _find_empty(scene, empty_name)
    if empty is None:
        return
    mesh = _find_mesh_child(scene, empty_name)
    if mesh is not None:
        mesh["matrix_world"] = [row[:] for row in empty["matrix_world"]]


def _update_bbox(obj: Dict) -> None:
    """Recompute bbox from location and dim (if dim is present).

    After a translation, the AABB center moves but the size (dim) stays the same.
    After a scale, dim is already updated by the caller before this is invoked.
    Rotation changes the AABB shape non-trivially; for a pure yaw rotation around Y
    we recompute the axis-aligned half-extents from the rotated box corners.

    bbox layout: [min_x, max_x, min_y, max_y, min_z, max_z]
    """
    if "dim" not in obj:
        return

    cx, cy, cz = obj["location"]
    dx, dy, dz = obj["dim"]
    yaw = obj["rotation_euler"][2]

    # Rotated AABB half-extents in X and Z (Y / height axis is unaffected by yaw)
    half_dx = dx / 2.0
    half_dz = dz / 2.0
    cos_a = abs(math.cos(yaw))
    sin_a = abs(math.sin(yaw))
    hx = half_dx * cos_a + half_dz * sin_a   # AABB half-width after yaw
    hz = half_dx * sin_a + half_dz * cos_a   # AABB half-depth after yaw
    hy = dy / 2.0                              # height unaffected by yaw

    obj["bbox"] = [
        round(cx - hx, 6), round(cx + hx, 6),
        round(cy - hy, 6), round(cy + hy, 6),
        round(cz - hz, 6), round(cz + hz, 6),
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Action applicators
# ─────────────────────────────────────────────────────────────────────────────

def apply_action(
    scene: Dict,
    action: Dict,
    sem_to_name: Dict[str, str],
    name_to_sem: Dict[str, str],
) -> str:
    """Apply a single action to the scene (mutates in place). Returns a log line."""
    name = action["action"]
    args = action.get("args", {})

    dispatch = {
        "remove_object":   _apply_remove,
        "add_object":      _apply_add,
        "move_to":         _apply_move_to,
        "move_group":      _apply_move_to,   # same geometry as move_to
        "place_relative":  _apply_place_relative,
        "place_on":        _apply_place_on,
        "place_between":   _apply_place_between,
        "rotate_towards":  _apply_rotate_towards,
        "rotate_by":       _apply_rotate_by,
        "scale":           _apply_scale,
        "align_with":      _apply_align_with,
        "stylize":         _apply_stylize,
    }

    fn = dispatch.get(name)
    if fn is None:
        return f"  [SKIP] Unknown action: {name}"
    return fn(scene, args, sem_to_name, name_to_sem)


# ── remove_object ─────────────────────────────────────────────────────────────

def _apply_remove(scene, args, sem_to_name, name_to_sem):
    obj_id = args.get("obj", "")
    empty_name = _resolve(obj_id, sem_to_name)
    if empty_name is None:
        return f"  remove_object({obj_id}) → NOT FOUND (skipped)"

    removed = []
    keep = []
    for o in scene["objects"]:
        if o["name"] == empty_name and o["type"] == "EMPTY":
            removed.append(o["name"])
        elif o["type"] == "MESH" and o.get("parent") == empty_name:
            removed.append(o["name"])
        else:
            keep.append(o)
    scene["objects"] = keep

    # Clean up maps
    del sem_to_name[obj_id]
    if empty_name in name_to_sem:
        del name_to_sem[empty_name]

    return f"  remove_object({obj_id}) → removed {removed}"


# ── add_object ────────────────────────────────────────────────────────────────

def _apply_add(scene, args, sem_to_name, name_to_sem):
    obj_id  = args.get("obj", "")
    cat     = args.get("cat", "object")
    support = args.get("support", "")

    if obj_id in sem_to_name:
        return f"  add_object({obj_id}) → already exists (skipped)"

    # Determine placement position
    support_name = _resolve(support, sem_to_name)
    support_obj  = _find_empty(scene, support_name) if support_name else None

    if support_obj is not None:
        sloc   = support_obj["location"]
        s_sy   = support_obj["scale"][1]
        new_loc = [sloc[0], sloc[1] + s_sy + 0.05, sloc[2]]
    else:
        new_loc = [0.0, 0.05, 0.0]   # room centre, near floor

    # Build a new unique name
    idx        = _next_name_index(scene)
    cat_clean  = cat.replace(" ", "_").lower()
    empty_name = f"{idx:03d}_{cat_clean}"

    new_scale  = [0.05, 0.05, 0.05]
    new_euler  = [0.0, 0.0, 0.0]

    new_empty: Dict[str, Any] = {
        "name":               empty_name,
        "type":               "EMPTY",
        "location":           new_loc,
        "rotation_mode":      "XYZ",
        "rotation_euler":     new_euler,
        "rotation_quaternion":[1.0, 0.0, 0.0, 0.0],
        "scale":              new_scale,
        "matrix_world":       [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "parent":             None,
        "parent_type":        "OBJECT",
        "collections":        ["Collection"],
        "hide_viewport":      False,
        "hide_render":        False,
        "_added_by_plan":     True,
        "_semantic_id":       obj_id,
        "_category":          cat,
    }
    _recompute_matrix_world(new_empty)

    # Minimal MESH child (no real geometry, placeholder)
    mesh_name = f"model.{idx:03d}"
    new_mesh: Dict[str, Any] = {
        "name":               mesh_name,
        "type":               "MESH",
        "location":           [0.0, 0.0, 0.0],
        "rotation_mode":      "QUATERNION",
        "rotation_euler":     [0.0, 0.0, 0.0],
        "rotation_quaternion":[1.0, 0.0, 0.0, 0.0],
        "scale":              [1.0, 1.0, 1.0],
        "matrix_world":       [row[:] for row in new_empty["matrix_world"]],
        "parent":             empty_name,
        "parent_type":        "OBJECT",
        "collections":        ["Scene Collection"],
        "hide_viewport":      False,
        "hide_render":        False,
        "glb_file":           f"{mesh_name}.glb",
    }

    scene["objects"].extend([new_empty, new_mesh])
    sem_to_name[obj_id]    = empty_name
    name_to_sem[empty_name] = obj_id

    return f"  add_object({obj_id}, cat={cat}, support={support}) → {empty_name}"


# ── move_to ───────────────────────────────────────────────────────────────────

def _apply_move_to(scene, args, sem_to_name, name_to_sem):
    obj_id = args.get("obj", args.get("parent", ""))
    pos    = args.get("pos", "")
    name   = _resolve(obj_id, sem_to_name)
    empty  = _find_empty(scene, name) if name else None
    if empty is None:
        return f"  move_to({obj_id}) → NOT FOUND"

    try:
        if isinstance(pos, (list, tuple)):
            coords = [float(v) for v in pos]
        elif "," in str(pos):
            coords = [float(v) for v in str(pos).split(",")]
        else:
            return f"  move_to({obj_id}, pos={pos}) → ambiguous position (skipped)"

        empty["location"] = coords[:3]
        _recompute_matrix_world(empty)
        _sync_mesh_child(scene, name)
        return f"  move_to({obj_id}) → location={[round(c, 4) for c in coords[:3]]}"
    except (ValueError, IndexError):
        return f"  move_to({obj_id}, pos={pos}) → parse error (skipped)"


# ── place_relative ────────────────────────────────────────────────────────────

def _apply_place_relative(scene, args, sem_to_name, name_to_sem):
    obj_id   = args.get("obj", "")
    tgt_id   = args.get("target", "")
    relation = args.get("relation", "near")

    obj_name = _resolve(obj_id, sem_to_name)
    tgt_name = _resolve(tgt_id, sem_to_name)
    obj_e    = _find_empty(scene, obj_name) if obj_name else None
    tgt_e    = _find_empty(scene, tgt_name) if tgt_name else None

    if obj_e is None:
        return f"  place_relative({obj_id}) → obj NOT FOUND"
    if tgt_e is None:
        return f"  place_relative({obj_id}, target={tgt_id}) → target NOT FOUND"

    tc = tgt_e["location"]     # [tx, ty, tz]
    ts = tgt_e["scale"]        # approx half-extents
    os_ = obj_e["scale"]

    GAP = 0.05
    new_loc = list(tc)

    # Coordinate axes: X=left/right, Y=up/height, Z=front/back (Z increases away)
    if relation == "left_of":
        new_loc[0] = tc[0] - ts[0] - os_[0] - GAP
    elif relation == "right_of":
        new_loc[0] = tc[0] + ts[0] + os_[0] + GAP
    elif relation == "in_front_of":
        new_loc[2] = tc[2] + ts[2] + os_[2] + GAP
    elif relation == "behind":
        new_loc[2] = tc[2] - ts[2] - os_[2] - GAP
    else:  # "near" or default
        new_loc[0] = tc[0] + ts[0] + os_[0] + GAP

    obj_e["location"] = new_loc
    _recompute_matrix_world(obj_e)
    _sync_mesh_child(scene, obj_name)
    return (f"  place_relative({obj_id}, {tgt_id}, {relation}) "
            f"→ location={[round(c, 4) for c in new_loc]}")


# ── place_on ──────────────────────────────────────────────────────────────────

def _apply_place_on(scene, args, sem_to_name, name_to_sem):
    obj_id  = args.get("obj", "")
    srf_id  = args.get("surface", "")

    obj_name = _resolve(obj_id, sem_to_name)
    srf_name = _resolve(srf_id, sem_to_name)
    obj_e    = _find_empty(scene, obj_name) if obj_name else None
    srf_e    = _find_empty(scene, srf_name) if srf_name else None

    if obj_e is None:
        return f"  place_on({obj_id}) → NOT FOUND"
    if srf_e is None:
        return f"  place_on({obj_id}, surface={srf_id}) → surface NOT FOUND"

    # Top of surface (Y-up): surface_center_y + surface_half_height + object_half_height
    srf_top   = srf_e["location"][1] + srf_e["scale"][1]
    new_loc   = [srf_e["location"][0],
                 srf_top + obj_e["scale"][1],
                 srf_e["location"][2]]

    obj_e["location"] = new_loc
    _recompute_matrix_world(obj_e)
    _sync_mesh_child(scene, obj_name)
    return f"  place_on({obj_id}, {srf_id}) → location={[round(c, 4) for c in new_loc]}"


# ── place_between ─────────────────────────────────────────────────────────────

def _apply_place_between(scene, args, sem_to_name, name_to_sem):
    obj_id  = args.get("obj", "")
    ref1_id = args.get("obj1", "")
    ref2_id = args.get("obj2", "")

    ok  = _resolve(obj_id,  sem_to_name)
    k1  = _resolve(ref1_id, sem_to_name)
    k2  = _resolve(ref2_id, sem_to_name)
    obj_e = _find_empty(scene, ok) if ok else None
    e1    = _find_empty(scene, k1) if k1 else None
    e2    = _find_empty(scene, k2) if k2 else None

    if obj_e is None:
        return f"  place_between({obj_id}) → NOT FOUND"
    if e1 is None or e2 is None:
        return f"  place_between({obj_id}) → reference objects NOT FOUND"

    midpoint = [(e1["location"][i] + e2["location"][i]) / 2 for i in range(3)]
    obj_e["location"] = midpoint
    _recompute_matrix_world(obj_e)
    _sync_mesh_child(scene, ok)
    return (f"  place_between({obj_id}, {ref1_id}, {ref2_id}) "
            f"→ location={[round(c, 4) for c in midpoint]}")


# ── rotate_towards ────────────────────────────────────────────────────────────

def _apply_rotate_towards(scene, args, sem_to_name, name_to_sem):
    obj_id = args.get("obj", "")
    tgt_id = args.get("target", "")

    ok  = _resolve(obj_id, sem_to_name)
    tk  = _resolve(tgt_id, sem_to_name)
    obj_e = _find_empty(scene, ok) if ok else None
    tgt_e = _find_empty(scene, tk) if tk else None

    if obj_e is None:
        return f"  rotate_towards({obj_id}) → NOT FOUND"
    if tgt_e is None:
        return f"  rotate_towards({obj_id}, target={tgt_id}) → target NOT FOUND"

    # Yaw = angle in the XZ-plane (horizontal plane; Y is up).
    # Convention from scene data: yaw=0 → faces +Z, yaw=π/2 → faces +X,
    # so forward(θ) = (sin(θ), 0, cos(θ)) and θ = atan2(dx, dz).
    dx = tgt_e["location"][0] - obj_e["location"][0]
    dz = tgt_e["location"][2] - obj_e["location"][2]
    yaw = math.atan2(dx, dz)

    obj_e["rotation_euler"][2] = yaw
    _recompute_matrix_world(obj_e)
    _sync_mesh_child(scene, ok)
    return f"  rotate_towards({obj_id}, {tgt_id}) → yaw={round(yaw, 4)} rad"


# ── rotate_by ─────────────────────────────────────────────────────────────────

def _apply_rotate_by(scene, args, sem_to_name, name_to_sem):
    obj_id  = args.get("obj", "")
    degrees = float(args.get("degrees", 0))

    ok    = _resolve(obj_id, sem_to_name)
    obj_e = _find_empty(scene, ok) if ok else None
    if obj_e is None:
        return f"  rotate_by({obj_id}) → NOT FOUND"

    obj_e["rotation_euler"][2] += math.radians(degrees)
    _recompute_matrix_world(obj_e)
    _sync_mesh_child(scene, ok)
    return (f"  rotate_by({obj_id}, {degrees}°) "
            f"→ yaw={round(obj_e['rotation_euler'][2], 4)} rad")


# ── scale ─────────────────────────────────────────────────────────────────────

def _apply_scale(scene, args, sem_to_name, name_to_sem):
    obj_id = args.get("obj", "")
    sx = float(args.get("sx", 1))
    sy = float(args.get("sy", 1))
    sz = float(args.get("sz", 1))

    ok    = _resolve(obj_id, sem_to_name)
    obj_e = _find_empty(scene, ok) if ok else None
    if obj_e is None:
        return f"  scale({obj_id}) → NOT FOUND"

    obj_e["scale"] = [obj_e["scale"][0] * sx,
                      obj_e["scale"][1] * sy,
                      obj_e["scale"][2] * sz]
    _recompute_matrix_world(obj_e)
    _sync_mesh_child(scene, ok)
    return (f"  scale({obj_id}, [{sx},{sy},{sz}]) "
            f"→ scale={[round(s, 5) for s in obj_e['scale']]}")


# ── align_with ────────────────────────────────────────────────────────────────

def _apply_align_with(scene, args, sem_to_name, name_to_sem):
    obj_id = args.get("obj", "")
    tgt_id = args.get("target", "")
    axis   = args.get("axis", "x").lower()

    ok  = _resolve(obj_id, sem_to_name)
    tk  = _resolve(tgt_id, sem_to_name)
    obj_e = _find_empty(scene, ok) if ok else None
    tgt_e = _find_empty(scene, tk) if tk else None

    if obj_e is None or tgt_e is None:
        return f"  align_with({obj_id}, {tgt_id}) → NOT FOUND"

    axis_map = {"x": 0, "y": 1, "z": 2}
    ai = axis_map.get(axis, 0)
    obj_e["location"][ai] = tgt_e["location"][ai]
    _recompute_matrix_world(obj_e)
    _sync_mesh_child(scene, ok)
    return f"  align_with({obj_id}, {tgt_id}, axis={axis})"


# ── stylize ───────────────────────────────────────────────────────────────────

def _apply_stylize(scene, args, sem_to_name, name_to_sem):
    obj_id = args.get("obj", "")
    desc   = args.get("desc", "")

    ok    = _resolve(obj_id, sem_to_name)
    obj_e = _find_empty(scene, ok) if ok else None
    if obj_e is None:
        return f"  stylize({obj_id}) → NOT FOUND"

    obj_e["_style"] = desc
    return f"  stylize({obj_id}, '{desc}')"


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply EditLang plans to scene layout (new Blender format)"
    )
    parser.add_argument("--scene",  required=True, help="Path to scene layout JSON (new format)")
    parser.add_argument("--plans",  required=True, help="Path to plan results JSON")
    parser.add_argument("--outdir", required=True, help="Output directory for per-instruction scenes")
    cli = parser.parse_args()

    with open(cli.scene) as f:
        original_scene = json.load(f)
    with open(cli.plans) as f:
        results = json.load(f)

    os.makedirs(cli.outdir, exist_ok=True)

    n_obj = sum(1 for o in original_scene["objects"] if o["type"] == "EMPTY")
    print("╔═══════════════════════════════════════════╗")
    print("║  Apply Plans to Scene Layout (new format) ║")
    print("╚═══════════════════════════════════════════╝")
    print(f"  Scene : {cli.scene}  ({n_obj} EMPTY objects)")
    print(f"  Plans : {cli.plans}  ({len(results)} instructions)")
    print(f"  Output: {cli.outdir}")

    for result in results:
        idx     = result["index"]
        cmd     = result["command"]
        instr   = result["instruction"]
        plan    = result.get("plan", [])
        success = result.get("success", False)

        print(f"\n{'='*60}")
        print(f"  [{idx}] {cmd}: {instr[:70]}...")
        print(f"  Plan: {len(plan)} actions, success={success}")

        if not plan:
            print("  → SKIPPED (empty plan)")
            continue

        scene = copy.deepcopy(original_scene)
        name_to_sem, sem_to_name = build_id_maps(scene)

        for step_i, action in enumerate(plan):
            log = apply_action(scene, action, sem_to_name, name_to_sem)
            print(f"  Step {step_i + 1}: {log}")

        outfile = os.path.join(
            cli.outdir,
            f"scene_layout_instruction_{idx}_{cmd.lower()}.json",
        )
        with open(outfile, "w") as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)
        print(f"  → Saved: {outfile}")

    print(f"\n{'='*60}")
    print(f"  Done. Results in: {cli.outdir}")


if __name__ == "__main__":
    main()

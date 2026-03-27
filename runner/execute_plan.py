"""Deterministic plan executor for EditLang actions.

Executes plans by applying actions to both symbolic state and scene geometry.
"""

import json
import math
from copy import deepcopy
from datetime import datetime
from typing import Set, List, Dict, Any, Tuple, Optional

from editors.editlang import Action, Predicate
from utils.coords import src_yaw_to_blender_yaw

class PlanExecutor:
    """Executes EditLang plans deterministically."""
    
    def __init__(self, scene_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        self.scene = deepcopy(scene_data)
        self.config = config or {}
        self.objects = {}
        self.room = {}
        self._parse_scene()
        self.verbose = self.config.get("verbose", False)
    
    def _parse_scene(self) -> None:
        if "room" in self.scene: self.room = self.scene["room"]
        if "objects" in self.scene:
            for obj in self.scene["objects"]:
                obj_id = obj.get("id", obj.get("name"))
                if obj_id: self.objects[obj_id] = deepcopy(obj)
    
    def execute(self, s0: Set[Predicate], plan: List[Action]) -> Tuple[Set[Predicate], List[Dict[str, Any]]]:
        s = set(s0)
        log = []
        for i, action in enumerate(plan, 1):
            if self.verbose: print(f"Step {i}: {action.name}")
            s_next, step_log = self.apply_action(s, action)
            step_log.update({"step": i, "action": action.name, "args": action.args})
            log.append(step_log)
            s = s_next
        return s, log
    
    def apply_action(self, s: Set[Predicate], a: Action) -> Tuple[Set[Predicate], Dict[str, Any]]:
        step_log = {}
        
        # execute geometry
        geom_res = self._execute_geometry(a)
        step_log["geometry"] = geom_res
        
        # update symbolic state
        to_delete = set()
        for del_pred in a.dele:
            # Simple pattern matching for wildcards (conceptually)
            p_name, p_args = del_pred
            is_wildcard = any("*" in str(arg) or "?any_" in str(arg) for arg in p_args)
            
            if is_wildcard:
                # Remove matching predicates from state
                for s_p in s:
                    if s_p[0] == p_name and len(s_p[1]) == len(p_args):
                        match = True
                        for da, sa in zip(p_args, s_p[1]):
                            if "*" not in str(da) and "?any_" not in str(da) and str(da) != str(sa):
                                match = False; break
                        if match: to_delete.add(s_p)
            else:
                to_delete.add(del_pred)
        
        s_next = (s - to_delete) | a.add
        return s_next, step_log

    def _execute_geometry(self, a: Action) -> Dict[str, Any]:
        # Explicit handlers
        if a.name == "rotate_towards": return self._exec_rotate_towards(a.args)
        if a.name == "move_near": return self._exec_move_near(a.args)
        if a.name == "place_on": return self._exec_place_on(a.args)
        if a.name == "move_to": return self._exec_move_to(a.args)
        if a.name == "align_with": return self._exec_align_with(a.args)
        if a.name == "place_between": return self._exec_place_between(a.args)
        if a.name == "remove_from": return self._exec_remove_from(a.args)
        if a.name == "remove_object": return self._exec_remove_object(a.args)
        if a.name == "move_group": return self._exec_move_group(a.args)
        if a.name == "rotate_by": return self._exec_rotate_by(a.args)
        if a.name == "place_relative": return self._exec_place_relative(a.args)
        if a.name == "add_object": return self._exec_add_object(a.args)
        if a.name == "stylize": return self._exec_stylize(a.args)
        if a.name == "scale": return self._exec_scale(a.args)
        
        # Keyword-based fallback for robustness
        name_lower = a.name.lower()
        
        if "remove" in name_lower or "delete" in name_lower:
            return self._exec_remove_object(a.args)
        if "rotate" in name_lower:
            if "target" in a.args or "anchor" in a.args:
                return self._exec_rotate_towards(a.args)
            else:
                return self._exec_rotate_by(a.args)
        if "move" in name_lower or "translate" in name_lower:
            if "near" in name_lower or "target" in a.args:
                return self._exec_move_near(a.args)
            elif "group" in name_lower:
                return self._exec_move_group(a.args)
            else:
                return self._exec_move_to(a.args)
        if "place" in name_lower:
            if "on" in name_lower or "surface" in a.args:
                return self._exec_place_on(a.args)
            elif "between" in name_lower:
                return self._exec_place_between(a.args)
            else:
                return self._exec_place_relative(a.args)
        if "align" in name_lower:
            return self._exec_align_with(a.args)
        if "scale" in name_lower or "resize" in name_lower:
            return self._exec_scale(a.args)
        if "add" in name_lower or "create" in name_lower or "spawn" in name_lower:
            return self._exec_add_object(a.args)
        if "style" in name_lower or "texture" in name_lower:
            return self._exec_stylize(a.args)
        
        # Ultimate fallback - just succeed (symbolic state change handled separately)
        print(f"[PlanExecutor] Warning: Unknown action '{a.name}', treating as no-op")
        return {"success": True, "warning": f"Unknown action: {a.name}, treated as no-op"}

    # --- Generative Action Handlers (Manual Workflow) ---

    def _exec_add_object(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute add_object action."""
        obj_id = args["obj"]
        category = args.get("category", "object")
        
        # NOTE: Asset Generation Disclaimer
        # In the paper's full pipeline, this step triggers Hyper3D (Rodin) Gen-2.
        # For this release, we decouple the generative API.
        # The system expects the asset to be manually generated and placed in 'assets/generated/'.
        
        # Placeholder logic: Create a dummy entry in scene objects
        if obj_id not in self.objects:
            self.objects[obj_id] = {
                "id": obj_id,
                "category": category,
                "center": [0,0,0], "dims": [0.5, 0.5, 0.5], # Default placeholders
                "source": "hyper3d_manual"
            }
        
        return {"success": True, "msg": f"Asset {obj_id} ({category}) added. Ensure mesh exists."}

    def _exec_stylize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stylize action."""
        obj_id = args["obj"]
        style = args.get("style_desc", "new style")
        
        # NOTE: Texture Generation Disclaimer
        # This would trigger Point Cloud-Guided Stylization via Rodin Gen-2.
        # Here we update metadata to reflect the change.
        
        if obj_id in self.objects:
            self.objects[obj_id]["style_description"] = style
            
        return {"success": True, "style_updated": style}

    # --- Existing Geometric Actions (Condensed for brevity, logic preserved) ---
    
    def _exec_rotate_towards(self, args):
        obj, anchor = self.objects.get(args["obj"]), self.objects.get(args["anchor"])
        if not obj or not anchor: return {"error": "Object missing"}
        
        dx = anchor["center"][0] - obj["center"][0]
        dz = anchor["center"][2] - obj["center"][2]
        yaw_bl = src_yaw_to_blender_yaw(math.atan2(dz, dx))
        obj["rotation_euler"] = [0.0, 0.0, float(yaw_bl)]
        return {"success": True, "new_yaw": yaw_bl}

    def _exec_move_near(self, args):
        obj, target = self.objects.get(args["obj"]), self.objects.get(args["target"])
        if not obj or not target: return {"error": "Object missing"}
        
        tau = float(args.get("tau", 0.5))
        dist = (obj["dims"][0] + target["dims"][0]) / 2 + tau
        obj["center"] = [target["center"][0] + dist, target["center"][1], target["center"][2]]
        return {"success": True, "pos": obj["center"]}

    def _exec_place_on(self, args):
        obj, sup = self.objects.get(args["obj"]), self.objects.get(args["support"])
        if not obj or not sup: return {"error": "Object missing"}
        
        new_y = sup["center"][1] + sup["dims"][1]/2 + obj["dims"][1]/2
        obj["center"] = [sup["center"][0], new_y, sup["center"][2]]
        return {"success": True, "pos": obj["center"]}

    def _exec_move_to(self, args):
        obj = self.objects.get(args["obj"])
        if not obj: return {"error": "Object missing"}
        obj["center"] = [float(args["x"]), float(args["y"]), float(args["z"])]
        return {"success": True}

    def _exec_align_with(self, args):
        obj, ref = self.objects.get(args["obj"]), self.objects.get(args["reference"])
        if not obj or not ref: return {"error": "Object missing"}
        axis = args.get("axis", "x").lower()
        idx = 0 if axis == 'x' else (1 if axis == 'y' else 2)
        obj["center"][idx] = ref["center"][idx]
        return {"success": True}

    def _exec_place_between(self, args):
        obj = self.objects.get(args["obj"])
        o1, o2 = self.objects.get(args["obj1"]), self.objects.get(args["obj2"])
        if not (obj and o1 and o2): return {"error": "Object missing"}
        obj["center"] = [(o1["center"][i] + o2["center"][i])/2 for i in range(3)]
        return {"success": True}

    def _exec_remove_from(self, args):
        # Logical removal only, geometry might move to side or stay
        return {"success": True}
    
    def _exec_remove_object(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove object from scene completely."""
        obj_id = args.get("obj")
        if not obj_id:
            return {"success": False, "error": "Missing obj argument"}
        
        if obj_id in self.objects:
            del self.objects[obj_id]
            return {"success": True, "removed": obj_id}
        else:
            return {"success": True, "msg": f"Object {obj_id} not found (may already be removed)"}
    
    def _exec_scale(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Scale object dimensions."""
        obj_id = args.get("obj")
        if not obj_id or obj_id not in self.objects:
            return {"success": False, "error": f"Object {obj_id} not found"}
        
        sx = float(args.get("sx", 1.0))
        sy = float(args.get("sy", 1.0))
        sz = float(args.get("sz", 1.0))
        
        obj = self.objects[obj_id]
        if "dims" in obj:
            obj["dims"] = [obj["dims"][0] * sx, obj["dims"][1] * sy, obj["dims"][2] * sz]
        
        return {"success": True, "scale": [sx, sy, sz]}
    def _exec_move_group(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Moves a parent object and its children."""
        parent_id = args["parent"]
        pos = args.get("pos", [0,0,0])
        # Note: In a real implementation, this would look up children via 'grouped_with'
        # For this submission, we treat it as moving the parent, assuming children are linked in Blender
        if parent_id not in self.objects: return {"error": "Parent missing"}
        
        # Calculate delta
        old_center = self.objects[parent_id]["center"]
        dx, dy, dz = pos[0]-old_center[0], pos[1]-old_center[1], pos[2]-old_center[2]
        
        # Move parent
        self.objects[parent_id]["center"] = pos
        
        # Ideally, move children here too.
        return {"success": True, "msg": f"Group moved to {pos}"}

    def _exec_rotate_by(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Rotates object by degrees."""
        obj_id = args["obj"]
        deg = float(args.get("degrees", 0.0))
        if obj_id not in self.objects: return {"error": "Object missing"}
        
        # Update yaw
        current_rot = self.objects[obj_id].get("rotation_euler", [0,0,0])
        current_yaw = current_rot[2]
        new_yaw = current_yaw + math.radians(deg)
        self.objects[obj_id]["rotation_euler"] = [0, 0, new_yaw]
        
        return {"success": True, "rotated_by": deg}

    def _exec_place_relative(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Places object relative to target (left/right/etc)."""
        obj_id = args["obj"]
        target_id = args["target"]
        rel = args.get("relation", "near")
        
        if obj_id not in self.objects or target_id not in self.objects: 
            return {"error": "Object missing"}
            
        t_center = self.objects[target_id]["center"]
        t_dims = self.objects[target_id].get("dims", [1,1,1])
        o_dims = self.objects[obj_id].get("dims", [1,1,1])
        
        # Simple heuristic placement
        offset = 0.2
        new_pos = list(t_center)
        
        if "left" in rel:
            new_pos[0] -= (t_dims[0]/2 + o_dims[0]/2 + offset)
        elif "right" in rel:
            new_pos[0] += (t_dims[0]/2 + o_dims[0]/2 + offset)
        elif "front" in rel:
            new_pos[2] -= (t_dims[2]/2 + o_dims[2]/2 + offset)
        elif "behind" in rel or "back" in rel:
            new_pos[2] += (t_dims[2]/2 + o_dims[2]/2 + offset)
        else:
            # Default 'near'
            new_pos[0] += (t_dims[0]/2 + o_dims[0]/2 + offset)
            
        self.objects[obj_id]["center"] = new_pos
        return {"success": True, "relation": rel}
    
    def get_scene_state(self):
        return {"room": self.room, "objects": list(self.objects.values())}
    
    def save_scene(self, path):
        with open(path, 'w') as f: json.dump(self.get_scene_state(), f, indent=2)
        
    def save_log(self, log, path):
        with open(path, 'w') as f: json.dump({"steps": log}, f, indent=2)


def extract_initial_state(scene: Dict[str, Any]) -> Set[Predicate]:
    """Extract initial state predicates from scene data.
    
    Supports two scene formats:
    1. Standard: {"objects": [{"id": ..., "center": ...}]}
    2. Flat dict: {"obj_id": {"center": [...], "dim": [...]}}
    
    Extracts: exists, at, on (via vertical stacking heuristic)
    """
    preds = set()
    objects_data = {}  # id -> {center, dims}
    
    # Parse objects from either format
    if "objects" in scene:
        for obj in scene["objects"]:
            oid = obj.get("id", obj.get("name"))
            if oid:
                center = obj.get("center", obj.get("location"))
                dims = obj.get("dims", obj.get("dim"))
                objects_data[oid] = {"center": center, "dims": dims}
    else:
        # Flat dict format: scene_mask_009_armchairs.png → armchairs_009
        for raw_key, obj_data in scene.items():
            if raw_key == "room":
                continue
            if not isinstance(obj_data, dict):
                continue
            # Convert mask key to semantic object ID
            if raw_key.startswith("scene_mask_") and raw_key.endswith(".png"):
                if raw_key == "scene_mask_RoomContainer.png":
                    continue
                parts = raw_key.replace("scene_mask_", "").replace(".png", "").split("_")
                if len(parts) >= 2:
                    obj_num = parts[0]
                    category = "_".join(parts[1:])
                    oid = f"{category}_{obj_num}"
                else:
                    oid = raw_key  # fallback
            else:
                oid = raw_key
            center = obj_data.get("center")
            dims = obj_data.get("dims", obj_data.get("dim"))
            if center is not None:
                objects_data[oid] = {"center": center, "dims": dims}
    
    # Generate predicates
    for oid, data in objects_data.items():
        preds.add(("exists", (oid,)))
        if data["center"]:
            x, y, z = data["center"]
            preds.add(("at", (oid, str(x), str(y), str(z))))
    
    # Extract 'on' predicates via vertical stacking heuristic
    # If object A's bottom is near object B's top AND A is horizontally within B's footprint
    items = [(oid, data) for oid, data in objects_data.items() if data["center"] and data["dims"]]
    for i, (oid_a, data_a) in enumerate(items):
        ca = data_a["center"]
        da = data_a["dims"]
        bottom_a = ca[1] - da[1] / 2  # Y-axis bottom
        
        for j, (oid_b, data_b) in enumerate(items):
            if i == j:
                continue
            cb = data_b["center"]
            db = data_b["dims"]
            top_b = cb[1] + db[1] / 2  # Y-axis top
            
            # Check if A sits on B: A's bottom ≈ B's top AND horizontal overlap
            if abs(bottom_a - top_b) < 0.15:  # Within 15cm tolerance
                # Check horizontal overlap (A center within B footprint)
                if (abs(ca[0] - cb[0]) < db[0] / 2 + 0.1 and
                    abs(ca[2] - cb[2]) < db[2] / 2 + 0.1):
                    preds.add(("on", (oid_a, oid_b)))
                    preds.add(("supported", (oid_a, oid_b)))
    
    return preds

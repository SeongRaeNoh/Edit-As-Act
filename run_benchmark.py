#!/usr/bin/env python3
"""
Edit-As-Act: Benchmark Runner
Full pipeline: instruction → terminal conditions → regression planning → plan execution

Usage:
    python run_benchmark.py \
        --scene dataset/dataset/bedroom/scene_layout_edited.json \
        --instructions dataset/dataset/user_instructions_edit/instruction_bedroom.json \
        --output results/bedroom_results.json \
        --verbose

Environment:
    OPENAI_API_KEY    Required. Set via env or .env file.
"""

import sys, os, json, time, traceback, argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, List, Set
from editors.editlang import standard_domain
from runner.execute_plan import extract_initial_state, PlanExecutor
from planners.regression_planner import RegressionPlanner
from tools.llm_helpers import LLMHelper
from validators.llm_semantic_validator import LLMSemanticValidator
from validators.geom_checker import GeomChecker
from errors.planner_error import PlannerSchemaOrLogicError, PlannerMaxRetriesError, PlannerCycleError


# ──────────────────────────────────────────────
#  Data loaders
# ──────────────────────────────────────────────

def load_scene_layout(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_instructions(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("edit_instructions", [])


def count_objects(scene: Dict[str, Any]) -> int:
    """Count objects in scene, supporting both formats."""
    if "objects" in scene:
        return len(scene["objects"])
    # Flat dict format: scene_mask_* keys or plain object keys
    count = 0
    for k, v in scene.items():
        if k == "room" or not isinstance(v, dict):
            continue
        if k.startswith("scene_mask_") and k.endswith(".png"):
            if k == "scene_mask_RoomContainer.png":
                continue
        count += 1
    return count


def get_actionable_predicates(editlang_spec: Dict) -> List[str]:
    """Return only predicates that appear in at least one action's add effects."""
    actionable = set()
    for action_def in editlang_spec.get("actions", {}).values():
        for eff in action_def.get("add", []):
            pred_name = eff.get("pred") if isinstance(eff, dict) else (eff[0] if isinstance(eff, list) else None)
            if pred_name:
                actionable.add(pred_name)
    actionable.update({"exists", "removed"})
    return sorted(actionable)


# ──────────────────────────────────────────────
#  Single instruction runner
# ──────────────────────────────────────────────

def run_single(
    idx: int,
    total: int,
    instr: Dict[str, str],
    llm: LLMHelper,
    planner: RegressionPlanner,
    scene: Dict[str, Any],
    allowed_preds: List[str],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run full pipeline for one instruction."""
    command = instr["command"]
    instruction = instr["instruction"]

    header = f"  [{idx+1}/{total}] {command}: {instruction[:80]}{'...' if len(instruction) > 80 else ''}"
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")

    result = {
        "index": idx + 1,
        "command": command,
        "instruction": instruction,
    }

    t_start = time.time()

    # ── Phase 1: Extract terminal conditions ──
    print("[Phase 1] Extracting terminal conditions from instruction...")
    t1 = time.time()
    try:
        terminal = llm.extract_terminal_conditions(
            instruction=instruction,
            scene_context=scene,
            allowed_predicates=allowed_preds,
        )
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  ❌ Terminal extraction failed ({time.time()-t1:.1f}s): {e}")
        traceback.print_exc()
        result.update({"success": False, "phase": "terminal_extraction",
                        "error": str(e), "time_s": round(elapsed, 1),
                        "plan_length": 0, "plan": []})
        return result

    t1_elapsed = time.time() - t1
    print(f"  Terminal conditions ({t1_elapsed:.1f}s, {len(terminal)} predicates):")
    for pred in sorted(terminal, key=str):
        print(f"    - {pred[0]}({', '.join(str(a) for a in pred[1])})")

    if not terminal:
        elapsed = time.time() - t_start
        print(f"  ❌ No terminal conditions extracted")
        result.update({"success": False, "phase": "terminal_extraction",
                        "error": "empty_terminal", "time_s": round(elapsed, 1),
                        "plan_length": 0, "plan": [],
                        "terminal_conditions": []})
        return result

    result["terminal_conditions"] = [
        {"pred": p[0], "args": list(p[1])} for p in terminal
    ]

    # ── Phase 2: Planning ──
    print(f"\n[Phase 2] Planning (K={planner.K}, max_steps={planner.max_steps})...")
    t2 = time.time()
    s0 = extract_initial_state(scene)
    try:
        plan = planner.plan(
            s0=s0,
            G=terminal,
            instruction_raw=instruction,
            G_terminal=terminal,
        )
    except (PlannerMaxRetriesError, PlannerCycleError) as e:
        elapsed = time.time() - t_start
        print(f"  ⚠️ Planner stopped ({time.time()-t2:.1f}s): {e}")
        result.update({"success": False, "phase": "planning",
                        "error": str(e), "time_s": round(elapsed, 1),
                        "plan_length": 0, "plan": []})
        return result
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  ❌ Planning failed ({time.time()-t2:.1f}s): {e}")
        traceback.print_exc()
        result.update({"success": False, "phase": "planning",
                        "error": str(e), "time_s": round(elapsed, 1),
                        "plan_length": 0, "plan": []})
        return result

    t2_elapsed = time.time() - t2

    if not plan or len(plan) == 0:
        elapsed = time.time() - t_start
        print(f"  ❌ Empty plan ({t2_elapsed:.1f}s)")
        result.update({"success": False, "phase": "planning",
                        "error": "empty_plan", "time_s": round(elapsed, 1),
                        "plan_length": 0, "plan": []})
        return result

    # ── Phase 3: Plan execution (validation) ──
    print(f"\n[Phase 3] Executing plan ({len(plan)} actions)...")
    t3 = time.time()
    try:
        executor = PlanExecutor(scene_data=scene, config={"verbose": verbose})
        s_final, exec_log = executor.execute(s0, plan)

        # Check how many terminal conditions are satisfied
        satisfied = sum(1 for g in terminal if g in s_final)
        total_goals = len(terminal)
        goal_rate = satisfied / total_goals if total_goals > 0 else 0.0
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Plan execution error (non-fatal): {e}")
        s_final = None
        exec_log = []
        satisfied = 0
        total_goals = len(terminal)
        goal_rate = 0.0

    t3_elapsed = time.time() - t3

    # ── Results ──
    elapsed = time.time() - t_start
    print(f"\n  ✅ Plan generated ({len(plan)} actions, {elapsed:.1f}s)")
    if s_final is not None:
        print(f"  Goal satisfaction: {satisfied}/{total_goals} ({goal_rate:.0%})")

    plan_json = []
    for i, step in enumerate(plan):
        action_name = getattr(step, "name", step.get("name", "?") if isinstance(step, dict) else "?")
        args = getattr(step, "args", step.get("args", {}) if isinstance(step, dict) else {})
        print(f"    Step {i+1}: {action_name}({args})")
        plan_json.append({"action": action_name, "args": args})

    result.update({
        "success": True,
        "plan_length": len(plan),
        "time_s": round(elapsed, 1),
        "time_terminal_s": round(t1_elapsed, 1),
        "time_planning_s": round(t2_elapsed, 1),
        "time_execution_s": round(t3_elapsed, 1),
        "plan": plan_json,
        "goal_satisfaction": {
            "satisfied": satisfied,
            "total": total_goals,
            "rate": round(goal_rate, 4),
        },
    })
    return result


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Edit-As-Act: Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bedroom scene
  python run_benchmark.py \\
    --scene dataset/dataset/bedroom/scene_layout_edited.json \\
    --instructions dataset/dataset/user_instructions_edit/instruction_bedroom.json \\
    --output results/bedroom_results.json

  # Quick test (first 3 instructions)
  python run_benchmark.py \\
    --scene dataset/dataset/bedroom/scene_layout_edited.json \\
    --instructions dataset/dataset/user_instructions_edit/instruction_bedroom.json \\
    --max 3 --verbose
""",
    )
    parser.add_argument("--scene", required=True, help="Path to scene layout JSON")
    parser.add_argument("--instructions", required=True, help="Path to instruction JSON")
    parser.add_argument("--output", default="results/benchmark_results.json", help="Output results path")
    parser.add_argument("--model", default="gpt-5", help="LLM model name (default: gpt-5)")
    parser.add_argument("--max", type=int, default=None, help="Max number of instructions to run")
    parser.add_argument("--max-steps", type=int, default=64, help="Max planning depth per instruction (default: 64)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Derive scene name from scene path for display
    scene_name = os.path.basename(os.path.dirname(args.scene))

    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Edit-As-Act: Benchmark — {scene_name:<33s}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Load domain ──
    print("\n[Setup] Loading domain...")
    domain = standard_domain()
    editlang_spec = domain.to_dict()
    allowed_preds = get_actionable_predicates(editlang_spec)
    print(f"  Domain: {len(domain.actions)} actions, {len(allowed_preds)} actionable predicates")

    # ── Load scene ──
    print(f"[Setup] Loading scene: {args.scene}")
    scene = load_scene_layout(args.scene)
    obj_count = count_objects(scene)
    print(f"  Objects: {obj_count}")

    # ── Load instructions ──
    print(f"[Setup] Loading instructions: {args.instructions}")
    instructions = load_instructions(args.instructions)
    n = min(len(instructions), args.max) if args.max else len(instructions)
    print(f"  Instructions: {n}/{len(instructions)}")

    # ── Initialize LLM ──
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Export it or add to .env file.")
        sys.exit(1)

    print(f"[Setup] Initializing LLM ({args.model})...")
    llm = LLMHelper(
        api_key=api_key,
        model=args.model,
        verbose=args.verbose,
    )

    # ── Initialize Validator ──
    print("[Setup] Initializing LLM Semantic Validator...")
    validator = LLMSemanticValidator(
        editlang_spec=editlang_spec,
        model_client=llm,
        timeout_ms=400000,
        verbose=args.verbose,
    )

    # ── Initialize GeomChecker ──
    print("[Setup] Initializing GeomChecker...")
    geom_checker = GeomChecker(scene_data=scene)
    print(f"  GeomChecker: {len(geom_checker.objects)} objects loaded")

    # ── Initialize Planner ──
    print(f"[Setup] Initializing Planner (K=3, max_steps={args.max_steps})...")
    planner = RegressionPlanner(
        domain=domain,
        scene_data=scene,
        llm_helper=llm,
        llm_validator=validator,
        geom_checker=geom_checker,
        skip_schema_validation=False,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # ── Run ──
    results = []
    for i in range(n):
        r = run_single(i, n, instructions[i], llm, planner, scene, allowed_preds, args.verbose)
        results.append(r)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in results if r["success"])
    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        if r["success"]:
            gs = r.get("goal_satisfaction", {})
            info = f"{r['plan_length']} actions, {r['time_s']}s, goals={gs.get('satisfied',0)}/{gs.get('total',0)}"
        else:
            info = f"{r.get('phase','?')}: {r.get('error','?')[:50]}"
        tc_count = len(r.get("terminal_conditions", []))
        print(f"  {status} | [{r['command']}] {r['instruction'][:50]}... | TC={tc_count} | {info}")

    # Aggregate goal satisfaction
    total_satisfied = sum(r.get("goal_satisfaction", {}).get("satisfied", 0) for r in results if r["success"])
    total_goals = sum(r.get("goal_satisfaction", {}).get("total", 0) for r in results if r["success"])
    avg_goal_rate = total_satisfied / total_goals if total_goals > 0 else 0.0

    print(f"\n  Total: {passed}/{n} passed")
    if total_goals > 0:
        print(f"  Goal satisfaction (passed): {total_satisfied}/{total_goals} ({avg_goal_rate:.1%})")

    # ── Save ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()

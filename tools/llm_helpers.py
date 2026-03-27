"""LLM helpers with reduced role - terminal condition extraction and action ranking only.

The LLM now only:
1. Extracts terminal conditions from natural language instructions
2. Ranks action candidates for regression planning
"""

import os
import json
import re
from typing import List, Set, Dict, Any, Tuple, Optional, Union
import urllib.request
import urllib.error
import base64  # 이미지 인코딩을 위해 추가

from editors.editlang import Predicate
from errors.planner_error import PlannerSchemaOrLogicError


EVALUATION_SYSTEM_PROMPT_INSTRUCTION_FIDELITY = """You are an interior designer and 3D scene editing expert.

You are given:
1) An original rendering of a 3D indoor scene BEFORE editing.
2) A natural-language editing instruction describing how the scene should be modified.
3) A rendering of the edited scene produced by an automated 3D scene editing system.

Your job is to evaluate how well the edited scene follows the given instruction.

Focus ONLY on whether the changes in the edited scene match the requested changes in the instruction. Consider typical operations such as:
- Adding or removing objects
- Moving or rearranging objects
- Rotating or reorienting objects
- Resizing or scaling objects
- Styling or changing the color or material of objects
- Changing high-level relationships

Do NOT evaluate:
- Image style, rendering quality, background color, or photorealism
- Minor aesthetic differences that are not mentioned in the instruction

Evaluate the system as follows:

- Scoring Criteria for Instruction Fidelity (0–100):
  100–81: Excellent Fidelity – All requested changes are correctly reflected in the edited scene. No important instruction element is missing or misinterpreted.
  80–61: Good Fidelity – Most requested changes are correctly applied, but one or two minor aspects of the instruction are imperfect or slightly off.
  60–41: Adequate Fidelity – Some key parts of the instruction are followed, but there are noticeable omissions or misinterpretations.
  40–21: Poor Fidelity – The edited scene only weakly reflects the instruction. Many requested changes are missing or incorrect.
  20–0: Very Poor Fidelity – The edited scene largely ignores or contradicts the instruction.

Your response must be a JSON object with the following format:

{
  "score": <integer from 0 to 100>,
  "explanation": "<2–4 sentences explaining why you gave this score>"
}
"""

EVALUATION_SYSTEM_PROMPT_SEMANTIC_CONSISTENCY = """You are an interior designer and 3D scene editing expert.

You are given:
1) An original rendering of a 3D indoor scene BEFORE editing.
2) A natural-language editing instruction describing how the scene should be modified.
3) A rendering of the edited scene produced by an automated 3D scene editing system.

Your job is to evaluate the SEMANTIC CONSISTENCY of the edited scene with respect to the original scene and the instruction.

Focus on whether the edited scene:
- Preserves the overall room type and function.
- Keeps object roles and usage reasonable.
- Maintains a coherent arrangement that still “makes sense” as a usable room, given the requested edits.
- Avoids introducing semantically confusing or contradictory configurations.

Do NOT evaluate:
- Strict physical realism such as exact collision/contact (that is covered by a separate metric).
- Rendering quality, texture realism, or lighting.

Evaluate the system as follows:

- Scoring Criteria for Semantic Consistency (0–100):
  100–81: Excellent Consistency – The edited scene preserves the original room’s function and context. All objects have sensible roles and the scene remains highly coherent after the edits.
  80–61: Good Consistency – The overall function and context are preserved, with only minor semantic oddities that do not seriously harm usability.
  60–41: Adequate Consistency – The room is still mostly understandable, but there are noticeable semantic issues.
  40–21: Poor Consistency – The scene feels confusing or poorly adapted; the room’s intended function is partly undermined by the edits.
  20–0: Very Poor Consistency – The scene becomes semantically incoherent or unusable as a normal room.

Your response must be a JSON object with the following format:

{
  "score": <integer from 0 to 100>,
  "explanation": "<2–4 sentences explaining why you gave this score>"
}
"""

EVALUATION_SYSTEM_PROMPT_PHYSICAL_REALISM = """You are an interior designer and 3D spatial reasoning expert.

You are given:
1) An original rendering of a 3D indoor scene BEFORE editing.
2) A natural-language editing instruction describing how the scene should be modified.
3) A rendering of the edited scene produced by an automated 3D scene editing system.

Your job is to evaluate the PHYSICAL PLAUSIBILITY of the edited scene.

Focus on whether the edited scene:
- Avoids obvious collisions.
- Respects support and gravity.
- Maintains accessibility and basic ergonomics.
- Uses plausible scales and positions for objects.

Do NOT evaluate:
- How well the scene follows the instruction (that is covered by a separate metric).
- Aesthetic style, color schemes, or rendering quality.

Evaluate the system as follows:

- Scoring Criteria for Physical Plausibility (0–100):
  100–81: Excellent Plausibility – No noticeable collisions or support issues. Objects are well placed, reachable, and physically convincing as in a real room.
  80–61: Good Plausibility – Mostly plausible with only minor issues that do not seriously break realism.
  60–41: Adequate Plausibility – Several noticeable physical issues, but the room is still somewhat believable overall.
  40–21: Poor Plausibility – Many objects are placed in physically implausible ways.
  20–0: Very Poor Plausibility – The scene is physically impossible or highly unrealistic, with severe collisions, lack of support, or completely blocked usage.

Your response must be a JSON object with the following format:

{
  "score": <integer from 0 to 100>,
  "explanation": "<2–4 sentences explaining why you gave this score>"
}
"""
class LLMHelper:
    """Helper class for LLM interactions with minimal responsibility."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5", verbose: bool = False):
        """Initialize LLM helper.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use
            verbose: Whether to print debug information
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided or set in environment")
        
        self.model = model
        self.verbose = verbose
    
    def evaluate_scene_edit(
            self,
            instruction: str,
            source_image_bytes: bytes,
            edited_image_bytes: bytes
        ) -> Dict[str, Any]:
        """
        Evaluates a scene edit using a vision-language model via the GPT-5 style API.
        """
        def encode_image(image_bytes: bytes) -> str:
            return base64.b64encode(image_bytes).decode('utf-8')

        base64_source = encode_image(source_image_bytes)
        base64_edited = encode_image(edited_image_bytes)

        user_prompt_content = [
            {"type": "input_text", "text": "This is the editing instruction:"},
            {"type": "input_text", "text": instruction},
            {"type": "input_text", "text": "This is the original scene BEFORE editing:"},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_source}",
            },
            {"type": "input_text", "text": "This is the edited scene AFTER the instruction was applied:"},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_edited}",
            },
            {"type": "input_text", "text": "Please provide your evaluation in the specified JSON format."},
        ]
        
        response_str = self._call_gpt5_api(
            EVALUATION_SYSTEM_PROMPT_PHYSICAL_REALISM,
            user_prompt_content,
            timeout=400000.0,
        )
        
        try:
            if not response_str:
                raise json.JSONDecodeError("API returned empty response", "", 0)

            if response_str.startswith("```json"):
                response_str = response_str[7:].strip()
            if response_str.endswith("```"):
                response_str = response_str[:-3].strip()
            
            result = json.loads(response_str)
            return result
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Failed to parse LVLM evaluation response: {e}")
                print(f"Raw response: {response_str}")
        raise PlannerSchemaOrLogicError(f"LVLM returned non-JSON response: {response_str}")


    def extract_terminal_conditions(
        self,
        instruction: str,
        scene_context: Optional[Dict[str, Any]] = None,
        allowed_predicates: Optional[List[str]] = None
    ) -> Set[Predicate]:
        """Extract terminal conditions from natural language instruction.
        
        Args:
            instruction: Natural language instruction
            scene_context: Optional scene information for context
            allowed_predicates: Optional list of allowed predicate names to constrain output
            
        Returns:
            Set of predicates representing terminal conditions
        """
        system_prompt = """You are a terminal condition extractor for scene editing.
Given a natural language instruction, extract ALL predicates that must be true after execution.

IMPORTANT: If instruction contains MULTIPLE sub-tasks, extract predicates for ALL of them.
Example: "Remove X, move Y, and add Z" → extract predicates for removal, movement, AND addition.

Output ONLY a JSON array of predicates. Each predicate has:
- "pred": predicate name
- "args": list of arguments

Available predicates:
- exists(obj_id): Object exists in scene
- removed(obj_id): Object is removed from scene (use for removal goals)
- is_facing(obj, anchor): Object faces anchor
- near(obj, target, distance): Object is near target (use for approximate positioning)
- on(obj, target): Object is on target
- at(obj, x, y, z): Object is at absolute position
- aligned_with(obj, ref, axis): Objects aligned on axis
- between(obj, obj1, obj2): Object is between two others
- has_style(obj, style_desc): Object has style/color/material
- has_scale(obj, sx, sy, sz): Object has specific scale

Rules for multi-step instructions:
1. REMOVE tasks → Add "removed(obj_id)" predicate (use actual scene object ID below)
2. MOVE/TRANSLATE tasks → near(obj, landmark, distance_estimate)
3. ADD tasks → exists(new_descriptive_id), placement (on/near scene object)
4. ROTATE tasks → is_facing(obj_id, target_id)
5. STYLIZE tasks → has_style(obj_id, style_description)
6. SCALE tasks → has_scale(obj_id, scale_factor, scale_factor, scale_factor)
7. CRITICAL: Use EXACT object IDs from the SCENE OBJECTS list below for existing objects
8. For new objects (ADD), use descriptive snake_case IDs (e.g., "wall_mounted_television")
9. Extract ALL sub-goals, not just the last one

No explanations, just the JSON array of ALL goal predicates.
"""
        # Constrain to allowed predicates if provided
        if allowed_predicates:
            allowed_block = "\nALLOWED PREDICATES (use ONLY these):\n" + "\n".join(f"- {p}" for p in sorted(allowed_predicates)) + "\n"
            system_prompt = system_prompt + allowed_block

        user_prompt = f'Instruction: "{instruction}"'
        
        # Build scene object list for grounding
        scene_obj_ids = set()
        if scene_context:
            obj_list = []
            
            if "objects" in scene_context:
                for obj in scene_context["objects"]:
                    obj_id = obj.get("id", obj.get("Variable Name", obj.get("name")))
                    if obj_id:
                        scene_obj_ids.add(obj_id)
                        cat = obj.get('Category', obj.get('category', 'unknown'))
                        center = obj.get('center', obj.get('location', []))
                        obj_list.append(f"- {obj_id} (category: {cat}, position: {center})")
            else:
                # Flat dict format (scene_mask_XXX_category.png)
                for key in sorted(scene_context.keys()):
                    if key.startswith("scene_mask_") and key.endswith(".png"):
                        if key == "scene_mask_RoomContainer.png":
                            continue
                        parts = key.replace("scene_mask_", "").replace(".png", "").split("_")
                        if len(parts) >= 2:
                            obj_num = parts[0]
                            category = "_".join(parts[1:])
                            obj_id = f"{category}_{obj_num}"
                            scene_obj_ids.add(obj_id)
                            center = scene_context[key].get('center', [])
                            center_str = [f"{c:.2f}" for c in center] if center else []
                            obj_list.append(f"- {obj_id} (category: {category}, position: [{', '.join(center_str)}])")
            
            if obj_list:
                user_prompt += "\n\nSCENE OBJECTS (you MUST use ONLY these IDs for existing objects):\n" + "\n".join(obj_list)
            else:
                user_prompt += "\n\nWARNING: Could not extract object list from scene."
        
        # Call LLM
        response = self._call_api(system_prompt, user_prompt)
        
        # Parse response
        try:
            pred_list = json.loads(response)
            predicates = set()
            
            for pred_data in pred_list:
                pred_name = pred_data["pred"]
                pred_args = tuple(str(a) for a in pred_data.get("args", []))
                
                # Post-filter 1: Only allowed predicates
                if allowed_predicates and pred_name not in allowed_predicates:
                    if self.verbose:
                        print(f"  [TC Filter] Skipping unsupported predicate: {pred_name}")
                    continue
                
                # Post-filter 2: Validate object IDs against scene
                # For predicates that reference EXISTING objects (not ADD targets)
                if scene_obj_ids and pred_name not in ("exists",):
                    # Check that at least one arg references a known object
                    has_known_obj = any(a in scene_obj_ids for a in pred_args)
                    if not has_known_obj and pred_name != "removed":
                        # All args are unknown — likely hallucinated
                        if self.verbose:
                            print(f"  [TC Filter] Skipping predicate with no known scene objects: "
                                  f"{pred_name}({', '.join(pred_args)})")
                        continue
                
                predicates.add((pred_name, pred_args))
            
            if self.verbose:
                print(f"Extracted terminal conditions: {predicates}")
            
            return predicates
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse terminal conditions: {e}")
            print(f"Raw response: {response}")
            return set()
    
    def propose_final_step(
        self,
        goal: Set[Predicate],
        available_actions: List[str]
    ) -> List[str]:
        """Propose ranking of actions for achieving goal as final step.
        
        Args:
            goal: Current goal predicates
            available_actions: List of available action names
            
        Returns:
            Ranked list of action names (best first)
        """
        # Format goal for prompt
        goal_str = self._format_predicates(goal)
        
        system_prompt = """You are an action ranker for regression planning.
Given a goal state and available actions, rank the actions by how well they would achieve the goal as the LAST step in a plan.

Output ONLY a JSON array of action names in order of preference (best first).
Include only actions from the available list.

Action effects:
- rotate_towards: adds is_facing(obj, anchor)
- move_near: adds near(obj, target, distance)
- place_on: adds on(obj, target) 
- move_to: adds at(obj, x, y, z)
- align_with: adds aligned_with(obj, ref, axis)
- place_between: adds between(obj, obj1, obj2)
- remove_from: removes on(obj, target) 

No explanations, just the JSON array of action names.
"""

        user_prompt = f"""Goal to achieve: {goal_str}

Available actions: {json.dumps(available_actions)}

Rank these actions for achieving the goal as the FINAL step."""

        # Call LLM
        response = self._call_api(system_prompt, user_prompt)
        
        # Parse response
        try:
            ranked_actions = json.loads(response)
            
            # Filter to only available actions
            valid_ranked = [a for a in ranked_actions if a in available_actions]
            
            if self.verbose:
                print(f"Action ranking for goal {goal_str}: {valid_ranked}")
            
            return valid_ranked
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse action ranking: {e}")
            print(f"Raw response: {response}")
            # Return original order as fallback
            return available_actions
    
    def chat(self, system: str, user: str, temperature: float = 0.0, timeout: float = 4.0) -> str:
        """Chat interface for compatibility with validators.
        
        Args:
            system: System prompt
            user: User prompt
            temperature: Temperature (overrides instance setting)
            timeout: Timeout in seconds
            
        Returns:
            API response text
        """
        # Use instance API with overridden temperature if needed
        return self._call_api(system, user, timeout)
    
    def _call_api(self, system_prompt: str, user_prompt: str, timeout: float = 300.0) -> str:
        """Call OpenAI API.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            timeout: Request timeout in seconds
            
        Returns:
            API response text
        """
        if self.model.startswith("gpt-5"):
            # GPT-5 style API (from original code)
            return self._call_gpt5_api(system_prompt, user_prompt, timeout)
        else:
            # Standard OpenAI API
            return self._call_openai_api(system_prompt, user_prompt, timeout)
    
    def _call_openai_api(self, system_prompt: str, user_prompt: Any, timeout: float) -> str:
        """Call standard OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200000
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                obj = json.loads(body)
                
                # Extract response
                if "choices" in obj and len(obj["choices"]) > 0:
                    return obj["choices"][0]["message"]["content"].strip()
                
                print(f"Unexpected API response format: {body[:500]}")
                return ""
                
        except Exception as e:
            print(f"API call failed: {e}")
            return ""
    
    def _call_gpt5_api(self, system_prompt: str, user_prompt: Union[str, List[Dict]], timeout: float) -> str:
        """
        Call GPT-5 style API. Handles both text-only and multi-modal (text+image) prompts.
        """
        url = "https://api.openai.com/v1/responses"

        # Build input blocks
        if isinstance(user_prompt, str):
            # Text-only
            input_blocks = [{"type": "input_text", "text": user_prompt}]
        elif isinstance(user_prompt, list):
            # Already a list of content blocks (we trust caller to format these correctly)
            input_blocks = user_prompt
        else:
            raise TypeError(f"Unsupported user_prompt type: {type(user_prompt)}")

        payload = {
            "model": self.model,
            "instructions": system_prompt,  # use proper field for system prompt
            "input": [
                {
                    "role": "user",
                    "content": input_blocks,
                }
            ],
            "max_output_tokens": 25000,
            "reasoning": {"effort": "high"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                obj = json.loads(body)

                if obj.get("status") == "incomplete":
                    reason = obj.get("incomplete_details", {}).get("reason", "unknown")
                    print(f"Response incomplete: {reason}")
                    return ""

                # Fast path if output_text is present (as in official SDK)
                if isinstance(obj, dict) and obj.get("output_text"):
                    return str(obj["output_text"]).strip()

                found_texts = []
                if isinstance(obj, dict) and "output" in obj:
                    output_list = obj.get("output", [])
                    if isinstance(output_list, list):
                        for part in output_list:
                            if isinstance(part, dict) and "content" in part:
                                content_list = part.get("content", [])
                                if isinstance(content_list, list):
                                    for block in content_list:
                                        if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                                            text = block.get("text")
                                            if isinstance(text, str) and text.strip():
                                                found_texts.append(text.strip())

                if found_texts:
                    return "\n".join(found_texts)

                print(f"Could not extract text from response: {body[:500]}")
                return ""

        except urllib.error.HTTPError as e:
            # Debug: print server error body, not just status code
            try:
                err_body = e.read().decode("utf-8")
                print(f"GPT-5 API call failed ({e.code}): {err_body[:500]}")
            except Exception:
                print(f"GPT-5 API call failed ({e.code}): {e}")
            return ""
        except Exception as e:
            print(f"GPT-5 API call failed: {e}")
            return ""

    # --- END: MODIFIED GPT-5 API CALLER ---
    def propose_transition_actions(
        self,
        instruction_raw: str,
        G_terminal: List,
        G_t: List,
        backward_history: List[Dict[str, Any]],
        S0_full: List,
        editlang_spec: Dict[str, Any],
        K: int = 5,
        rejection_feedback: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Propose Top-K grounded actions for current planning step.
        
        Returns ordered Top-K grounded actions (list of dicts):
          {action, args, pre, add, del, predicted_unmet_pre, rationale}
        On any schema/logic issue, raises PlannerSchemaOrLogicError (no retries here).
        
        Args:
            instruction_raw: Original natural language instruction
            G_terminal: Terminal goal predicates (list format)
            G_t: Current subgoal predicates (list format)
            backward_history: History of chosen actions so far
            S0_full: Complete initial state predicates (list format)
            editlang_spec: Full EditLang domain specification
            K: Number of actions to propose
            
        Returns:
            List of K action dictionaries with keys:
            [action, args, pre, add, del, predicted_unmet_pre, rationale]
            
        Raises:
            PlannerSchemaOrLogicError: On LLM failure or schema violation
        """
        # Extract valid action names from spec for prompt
        valid_action_names = list(editlang_spec.get("actions", {}).keys())
        if not valid_action_names:
            valid_action_names = ["rotate_towards", "move_near", "place_on", "move_to", 
                                 "align_with", "place_between", "remove_from"]
        
        action_list_str = ", ".join(valid_action_names)
        
        system_prompt = f"""You are the Planner LLM for an Edit-As-Act backward-planning loop.
ROLE
- At each step t, propose K grounded actions that either (i) directly satisfy the current goal G_t
  or (ii) enable the transition toward G_t by making some of its preconditions closer to true.
- You DO NOT perform geometric/physics checks.
- Use the provided EditLang specification as the authoritative source.
- Consider the full scene S0 (entire predicate set), not a summary.

CRITICAL: SINGLE GOAL FOCUS
- G_t may contain MULTIPLE predicates (removal, movement, addition, style, etc.)
- For THIS regression step, pick ONE target predicate from G_t to satisfy

AVAILABLE ACTIONS (use ONLY these):
{action_list_str}

IMPORTANT: The "action" field must be one of the above action names (e.g., "place_between", "rotate_towards").
These are NOT the same as predicates (e.g., "on" is a predicate, not an action).
Actions have specific schemas defined in editlang_spec.

CRITICAL GROUNDING RULES:
1. ALL arguments must be CONCRETE object IDs from S0 or new objects (e.g., "armchairs_009", "tall_decorative_vase")
2. NEVER use variables like ?obj, ?any_target, ?any_anchor - these are FORBIDDEN
3. Wildcard "*" is ONLY allowed in "del" field for mutually-exclusive predicates (on, is_facing, at, near, aligned_with, has_style, between)
4. Example VALID: {{"del": [["on", ["book_01", "*"]]]}}  (removes book from any surface)
5. Example INVALID: {{"del": [["on", ["book_01", "?any_target"]]]}}  (? is forbidden)

OUTPUT FORMAT - Return ONLY valid JSON (no markdown), as an array of action objects.
Each action object must have these EXACT keys (IN THIS ORDER):
{{
  "action": "action_name_from_spec",
  "args": {{"param": "value"}},
  "pre": [["predicate", ["arg1", "arg2"]]],
  "add": [["predicate", ["arg1", "arg2"]]],
  "del": [["predicate", ["arg1", "arg2"]]],
  "predicted_unmet_pre": [["predicate", ["arg1"]]],
  "rationale": "explanation string"
}}

CRITICAL: predicted_unmet_pre field
- Check EACH precondition in "pre" against S0_full
- If precondition NOT in S0, add to predicted_unmet_pre
- Example: pre=[exists(obj), clear(table)]
  If S0 has exists(obj) but NOT clear(table) → predicted_unmet_pre=[clear(table)]
- If ALL preconditions in S0 → predicted_unmet_pre=[] (empty)

PREDICATE FORMAT: ["predicate_name", ["arg1", "arg2", ...]]
Example: ["on", ["book", "table"]], ["is_facing", ["chair", "window"]]
"""
        
        # Build user payload with explicit action list reminder
        payload = {
            "INSTRUCTION": "Use ONLY the action names from editlang_spec.actions. These are the valid EditLang actions (e.g., place_between, rotate_towards), NOT predicates (e.g., on, is_facing).",
            "valid_action_names": valid_action_names,
            "instruction_raw": instruction_raw,
            "K": K,
            "G_terminal": G_terminal,
            "G_t": G_t,
            "backward_history": backward_history,
            "S0_full": S0_full,
            "editlang_spec": editlang_spec,
        }
        # Add rejection feedback if available (helps LLM avoid repeated mistakes)
        if rejection_feedback:
            payload["PREVIOUS_REJECTION_REASONS"] = rejection_feedback[-5:]  # Last 5 reasons
        user_payload = json.dumps(payload, ensure_ascii=False)
        
        # Call LLM
        resp_text = self._call_api(system_prompt, user_payload)
        
        if not resp_text:
            raise PlannerSchemaOrLogicError("LLM returned empty response")
        
        # Parse response
        try:
            data = json.loads(resp_text)
        except Exception as e:
            # Log raw response for debugging
            if self.verbose:
                print(f"[LLM Response] Parse failed. Raw text (first 500 chars):")
                print(resp_text[:500])
            raise PlannerSchemaOrLogicError(f"LLM returned non-JSON: {e}. Response: {resp_text[:200]}")
        
        if isinstance(data, dict) and "error" in data:
            # planner failure contract (already-structured)
            raise PlannerSchemaOrLogicError(
                f"LLM error: {data['error'].get('message','unknown')}"
            )
        
        if not isinstance(data, list) or len(data) == 0:
            raise PlannerSchemaOrLogicError(f"LLM returned empty list. Response: {resp_text[:200]}")
        
        if self.verbose:
            print(f"LLM proposed {len(data)} actions for G_t")
        
        return data
    
    def _format_predicates(self, preds: Set[Predicate]) -> str:
        """Format predicates for display.
        
        Args:
            preds: Set of predicates
            
        Returns:
            Formatted string
        """
        if not preds:
            return "{}"
        
        formatted = []
        for pred_name, args in sorted(preds):
            if args:
                formatted.append(f"{pred_name}({', '.join(str(a) for a in args)})")
            else:
                formatted.append(pred_name)
        
        return "{ " + ", ".join(formatted) + " }"


def extract_terminal_conditions_from_file(
    instruction: str,
    scene_file: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-5",
    allowed_predicates: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extract terminal conditions from instruction and optional scene file.
    
    Args:
        instruction: Natural language instruction
        scene_file: Optional path to scene JSON
        api_key: Optional API key
        model: Model to use
        allowed_predicates: Optional list of allowed predicate names to constrain output
        
    Returns:
        Dictionary with terminal_condition and optionally final_step_candidates
    """
    # Load scene if provided
    scene_context = None
    if scene_file:
        with open(scene_file, 'r') as f:
            scene_context = json.load(f)
    
    # Create helper
    helper = LLMHelper(api_key=api_key, model=model, verbose=True)
    
    # Extract terminal conditions
    terminal_preds = helper.extract_terminal_conditions(instruction, scene_context, allowed_predicates)
    
    # Format result
    result = {
        "terminal_condition": [
            {"pred": pred[0], "args": list(pred[1])}
            for pred in terminal_preds
        ],
        "final_step_candidates": []  # Will be filled by planner
    }
    
    return result
def evaluate_scene_edit_from_files(
    instruction: str,
    source_image_path: str,
    edited_image_path: str,
    api_key: Optional[str] = None,
    model: str = "gpt-5"  # 기본 모델을 gpt-5로 변경
) -> Dict[str, Any]:
    """
    Reads image files and evaluates the scene edit using the specified model.
    """
    try:
        with open(source_image_path, "rb") as f_source:
            source_bytes = f_source.read()
        
        with open(edited_image_path, "rb") as f_edited:
            edited_bytes = f_edited.read()
    except FileNotFoundError as e:
        print(f"Error reading image file: {e}")
        raise

    # benchmark.py에서 받은 model 인자를 그대로 전달합니다.
    helper = LLMHelper(api_key=api_key, model=model, verbose=True)
    
    return helper.evaluate_scene_edit(
        instruction=instruction,
        source_image_bytes=source_bytes,
        edited_image_bytes=edited_bytes
    )
# --- END: MODIFIED TOP-LEVEL HELPER FUNCTION ---
def main():
    """Command-line interface for LLM helpers."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract terminal conditions from instruction")
    parser.add_argument("--instruction", required=True, help="Natural language instruction")
    parser.add_argument("--scene", help="Optional scene JSON file for context")
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--model", default="gpt-5", help="Model to use")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Extract terminal conditions
    result = extract_terminal_conditions_from_file(
        args.instruction,
        args.scene,
        model=args.model
    )
    
    # Save result
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Terminal conditions saved to: {args.out}")
    
    if args.verbose:
        print(f"Extracted conditions: {result}")


if __name__ == "__main__":
    main()

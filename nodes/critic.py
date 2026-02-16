"""
Critic Node: Peer review step that checks if generated images match the original
scientific facts. Sets is_accurate and failing_scenes. Can be run in isolation for testing.
"""
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


def _image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critic node: for each (scene_id, image_path), compare with scene_blueprints and
    extracted_facts. Use a vision-capable LLM to judge if the image accurately
    represents the intended fact/scene. Return critic_feedback with is_accurate,
    failing_scenes, and feedback.
    """
    image_paths = state.get("image_paths") or {}
    blueprints = state.get("scene_blueprints") or []
    facts = state.get("extracted_facts") or []
    scene_by_id = {b.get("scene_id"): b for b in blueprints if b.get("scene_id")}

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for critic. pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

    failing_scenes: List[str] = []
    feedback_parts: List[str] = []

    for scene_id, path in image_paths.items():
        if not Path(path).exists():
            failing_scenes.append(scene_id)
            feedback_parts.append(f"{scene_id}: image file missing.")
            continue

        blueprint = scene_by_id.get(scene_id, {})
        intended = blueprint.get("description", "") + "\n" + blueprint.get("visual_prompt", "")
        facts_text = json.dumps(facts[:15], indent=2) if facts else "No structured facts."

        b64 = _image_to_base64(path)
        prompt = f"""You are a peer reviewer for an educational video. Check if this image ACCURATELY represents the intended content and has NO visual hallucinations.

Intended scene description and visual prompt:
{intended[:800]}

Relevant extracted facts (for consistency):
{facts_text[:1500]}

CRITIC CHECKLIST — You MUST check for:
1. Visual Hallucinations: extra limbs on objects (e.g., extra arms on a protein model), impossible anatomy, duplicate or nonsensical elements.
2. Nonsensical text: any text in the image that is garbled, wrong language, or unrelated to the intended content.
3. Scientific accuracy: the image must match the intended concept and facts (correct subject, no misleading visuals).

If the asset is a video frame, also consider: jittery or unstable motion would be a failure (for single images you cannot assess motion).

Reply with JSON only: {{"accurate": true or false, "reason": "one sentence", "hallucinations_found": ["list", "of", "issues"] or []}}."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=300,
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                lines = content.split("\n")
                lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                content = "\n".join(lines)
            verdict = json.loads(content)
            if not verdict.get("accurate", True):
                failing_scenes.append(scene_id)
                reason = verdict.get("reason", "Inaccurate")
                hallucinations = verdict.get("hallucinations_found") or []
                if hallucinations:
                    feedback_parts.append(f"{scene_id}: {reason} [Hallucinations: {', '.join(hallucinations)}]")
                else:
                    feedback_parts.append(f"{scene_id}: {reason}")
        except Exception as e:
            failing_scenes.append(scene_id)
            feedback_parts.append(f"{scene_id}: Review failed ({e})")

    is_accurate = len(failing_scenes) == 0
    critic_feedback = {
        "is_accurate": is_accurate,
        "failing_scenes": failing_scenes,
        "feedback": "; ".join(feedback_parts) if feedback_parts else "All scenes accurate.",
    }
    loop_count = (state.get("critic_loop_count") or 0) + 1
    return {"critic_feedback": critic_feedback, "critic_loop_count": loop_count}

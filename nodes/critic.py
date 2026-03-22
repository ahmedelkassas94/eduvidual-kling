"""
Critic Node: Scientific peer review that evaluates if generated images accurately
represent the facts extracted by the Retriever. Uses Gemini 2.0 Flash for multi-modal
vision analysis. Can be run in isolation for testing.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import env_loader  # noqa: F401 - load .env from project root first

def _get_critic_model() -> str:
    """Gemini model for vision-based scientific critique (supports multi-modal)."""
    return (os.getenv("CRITIC_MODEL") or "gemini-2.0-flash").strip()


def _extract_json_from_response(content: str) -> Dict[str, Any]:
    """Strip markdown code blocks and parse JSON from LLM response."""
    text = (content or "").strip()
    if not text:
        return {}
    # Strip ```json ... ``` or ``` ... ```
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _evaluate_single_image_with_gemini(
    *,
    image_path: Path,
    extracted_facts: List[Dict[str, Any]],
    scene_description: str,
) -> Tuple[bool, str, List[str]]:
    """
    Use Gemini 2.0 Flash (or Pro) to analyze an image against extracted facts.
    Returns (is_accurate, critique, revisions).
    """
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "google-genai and Pillow required for scientific critic. "
            "pip install google-genai Pillow"
        ) from e

    api_key = env_loader.require_env("GEMINI_API_KEY", "Scientific critic requires Gemini.")
    client = genai.Client(api_key=api_key)
    model = _get_critic_model()

    img = Image.open(image_path)

    facts_text = json.dumps(extracted_facts[:20], indent=2) if extracted_facts else "No structured facts provided."

    system_instruction = """You are a subject matter expert and scientific reviewer for educational content.
Your task is to evaluate whether a generated image ACCURATELY represents the scientific facts provided.

You must compare visual elements in the image against the extracted facts. Check for:
1. **Anatomical/Structural accuracy**: Correct number of parts (e.g., lobes in a lung, chambers in a heart)
2. **Process correctness**: Direction of flow (e.g., electron flow, blood flow), sequence of steps
3. **Label accuracy**: Any labels or annotations must match the facts
4. **Concept representation**: Diagrams, schematics, and illustrations must not contradict the facts
5. **Visual hallucinations**: Extra limbs, impossible anatomy, nonsensical elements, wrong counts

Be strict but fair. Minor stylistic differences are OK; factual errors are not.

You MUST respond with ONLY a JSON object (no markdown, no extra text):
{
  "is_accurate": true or false,
  "critique": "Detailed explanation of what is correct or incorrect. If inaccurate, specify which facts are violated and how the image differs.",
  "revisions": ["List of specific prompt adjustments to fix errors. Each item should be an imperative instruction, e.g. 'Show exactly 5 lobes in the right lung', 'Reverse the direction of the electron flow arrow'. Empty list if is_accurate is true."]
}"""

    user_content = f"""SCENE DESCRIPTION (intended visual content):
{scene_description[:1500]}

EXTRACTED SCIENTIFIC FACTS (must be accurately represented in the image):
{facts_text[:4000]}

Analyze the attached image. Compare every relevant visual element against these facts.
Return ONLY the JSON object with keys: is_accurate, critique, revisions."""

    response = client.models.generate_content(
        model=model,
        contents=[img, user_content],
        config=GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=1024,
        ),
    )

    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None) and response.candidates:
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)

    if not text or not text.strip():
        return False, "Critic returned empty response.", []

    out = _extract_json_from_response(text)
    is_accurate = bool(out.get("is_accurate", True))
    critique = out.get("critique", "")
    revisions = out.get("revisions", [])
    if not isinstance(revisions, list):
        revisions = [revisions] if revisions else []
    revisions = [str(r).strip() for r in revisions if r]

    return is_accurate, critique or "No critique provided.", revisions


def scientific_critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scientific critic: for each (scene_id, image_path), use Gemini vision to compare
    the image against extracted_facts and scene_description. Returns structured
    feedback: is_accurate, critique, revisions (prompt adjustments per failing scene).

    Handles file-not-found gracefully: missing images are marked inaccurate with
    appropriate feedback.
    """
    image_paths = state.get("image_paths") or {}
    blueprints = state.get("scene_blueprints") or []
    facts = state.get("extracted_facts") or []
    scene_by_id = {b.get("scene_id"): b for b in blueprints if b.get("scene_id")}

    failing_scenes: List[str] = []
    feedback_parts: List[str] = []
    revisions_by_scene: Dict[str, List[str]] = {}

    for scene_id, path in image_paths.items():
        image_path = Path(path)

        # DATA SAFETY: Handle file-not-found gracefully
        if not image_path.exists():
            failing_scenes.append(scene_id)
            feedback_parts.append(f"{scene_id}: Image file not found at {path}.")
            revisions_by_scene[scene_id] = [
                f"Regenerate the image; the previous output file was missing at {path}."
            ]
            continue

        blueprint = scene_by_id.get(scene_id, {})
        scene_description = (
            (blueprint.get("description") or "")
            + "\n"
            + (blueprint.get("visual_prompt") or "")
        ).strip() or f"Scene {scene_id}"

        try:
            is_accurate, critique, revisions = _evaluate_single_image_with_gemini(
                image_path=image_path,
                extracted_facts=facts,
                scene_description=scene_description,
            )
        except Exception as e:
            failing_scenes.append(scene_id)
            feedback_parts.append(f"{scene_id}: Review failed ({e})")
            revisions_by_scene[scene_id] = [f"Retry generation after reviewer error: {e}"]
            continue

        if not is_accurate:
            failing_scenes.append(scene_id)
            feedback_parts.append(f"{scene_id}: {critique[:200]}{'...' if len(critique) > 200 else ''}")
            revisions_by_scene[scene_id] = revisions if revisions else [
                "Apply corrections based on the scientific facts; ensure visual elements match the extracted facts exactly."
            ]

    is_accurate = len(failing_scenes) == 0
    critic_feedback = {
        "is_accurate": is_accurate,
        "failing_scenes": failing_scenes,
        "feedback": "; ".join(feedback_parts) if feedback_parts else "All scenes scientifically accurate.",
        "revisions": revisions_by_scene,
    }

    loop_count = (state.get("critic_loop_count") or 0) + 1
    return {"critic_feedback": critic_feedback, "critic_loop_count": loop_count}


# Alias for backward compatibility: critic_node now uses scientific_critic logic
critic_node = scientific_critic_node

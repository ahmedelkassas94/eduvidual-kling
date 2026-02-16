"""
Planner Node: Convert extracted facts into a scene-by-scene visual script (JSON).
Can be run in isolation for testing.
"""
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner node: convert extracted_facts (and optional raw_text) into scene_blueprints.
    Each blueprint has: scene_id, description, visual_prompt (for image generation), order.
    """
    facts = state.get("extracted_facts") or []
    raw_text = (state.get("raw_text") or "")[:8000]

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for planner. pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

    facts_str = json.dumps(facts, indent=2) if facts else "No structured facts (use raw text only)."

    prompt = f"""You are a visual scriptwriter for an educational video. Given the following extracted facts and source text, produce a scene-by-scene visual script.

Extracted facts:
{facts_str}

Source text (excerpt):
{raw_text[:4000]}

Output a JSON array of scene blueprints. Each scene must have:
- "scene_id": unique string id (e.g. "scene_1", "scene_2")
- "description": one-sentence summary of what this scene explains
- "visual_prompt": a detailed text-to-image prompt (2-4 sentences) describing the visual for this scene: subject, setting, lighting, style (photorealistic for real-world, clear for diagrams). No people or faces. Academic, professional look.
- "order": integer (1-based) for sequence

Use 3 to 8 scenes. Order must be logical for teaching. Return ONLY the JSON array, no markdown."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        blueprints = json.loads(content)
    except json.JSONDecodeError:
        blueprints = [{"scene_id": "scene_1", "description": "Intro", "visual_prompt": content[:500], "order": 1}]

    if not isinstance(blueprints, list):
        blueprints = [blueprints]
    # Ensure order and scene_id
    for i, b in enumerate(blueprints):
        b.setdefault("order", i + 1)
        b.setdefault("scene_id", f"scene_{i + 1}")

    return {"scene_blueprints": sorted(blueprints, key=lambda x: x.get("order", 0))}

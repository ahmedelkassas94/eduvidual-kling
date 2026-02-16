"""
Stylist Node: Generate a global visual style guide (colors, lighting, framing)
to ensure consistency across scenes. Can be run in isolation for testing.
"""
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


def stylist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stylist node: produce style_config from scene_blueprints (and optionally raw_text).
    style_config includes: colors, lighting, framing, visual_style, camera_rules.
    """
    blueprints = state.get("scene_blueprints") or []
    raw_text = (state.get("raw_text") or "")[:3000]

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for stylist. pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

    summaries = [b.get("description", b.get("visual_prompt", ""))[:200] for b in blueprints[:5]]
    context = "\n".join(summaries) if summaries else raw_text[:2000]

    prompt = f"""You are a visual style director. Given the following scene descriptions for an educational video, define a GLOBAL style guide so all scenes look consistent.

Scene context:
{context}

Produce a JSON object with exactly these keys (all strings):
- "visual_style": e.g. "Professional, documentary, photorealistic for real-world topics; clean diagrams for abstract concepts."
- "color_palette": e.g. "Neutral backgrounds, accent color for highlights; avoid saturated red/green for colorblind clarity."
- "lighting": e.g. "Soft, even key light; minimal harsh shadows; well-lit for readability."
- "framing": e.g. "Rule of thirds; subject centered or slightly off-center; consistent aspect 16:9."
- "camera_rules": e.g. "Static or slow push-in; no handheld; academic presentation."
- "continuity_rules": e.g. "Same color grading and lighting style across scenes; consistent typography if text appears."

Return ONLY the JSON object, no markdown."""

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
        style_config = json.loads(content)
    except json.JSONDecodeError:
        style_config = {
            "visual_style": "Professional, documentary.",
            "color_palette": "Neutral, clear.",
            "lighting": "Soft, even.",
            "framing": "16:9, rule of thirds.",
            "camera_rules": "Static or slow movement.",
            "continuity_rules": "Consistent grading and style.",
        }

    return {"style_config": style_config}

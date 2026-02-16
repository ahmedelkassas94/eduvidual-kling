"""
Visualizer Node: Call external Image APIs (Gemini/Imagen) to generate static images
for each scene blueprint. Wraps existing image_client. Can be run in isolation for testing.
"""
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


def _build_prompt(blueprint: Dict[str, Any], style_config: Dict[str, Any]) -> str:
    """Combine scene visual_prompt with global style_config."""
    prompt = blueprint.get("visual_prompt", "")
    if not prompt:
        return "Professional educational illustration, clean, well-lit, no people."
    style = style_config or {}
    parts = [prompt]
    if style.get("visual_style"):
        parts.append(f"Style: {style['visual_style']}")
    if style.get("lighting"):
        parts.append(f"Lighting: {style['lighting']}")
    return " ".join(parts)


def visualizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Visualizer node: for each scene in scene_blueprints, generate an image via Gemini (image_client).
    Optionally re-generate only failing_scenes if critic_feedback.failing_scenes is set.
    Writes images to output_dir and returns image_paths: { scene_id: path }.
    """
    blueprints = state.get("scene_blueprints") or []
    style_config = state.get("style_config") or {}
    output_dir = state.get("output_dir", "workflow_output")
    critic_feedback = state.get("critic_feedback") or {}
    failing_scenes = critic_feedback.get("failing_scenes") or []

    # If critic requested re-generation, only redo those scenes; keep existing image_paths
    image_paths = dict(state.get("image_paths") or {})

    if failing_scenes:
        to_generate = [b for b in blueprints if b.get("scene_id") in failing_scenes]
    else:
        to_generate = list(blueprints)

    if not to_generate:
        return {"image_paths": image_paths}

    try:
        from image_client import generate_image
    except ImportError:
        # Allow running from project root
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from image_client import generate_image

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    for blueprint in to_generate:
        scene_id = blueprint.get("scene_id", "scene_unknown")
        full_prompt = _build_prompt(blueprint, style_config)
        out_path = root / f"{scene_id}.png"
        generate_image(full_prompt, out_path)
        image_paths[scene_id] = str(out_path.resolve())

    return {"image_paths": image_paths}

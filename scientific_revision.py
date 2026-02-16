"""
Scientific accuracy revision: send first frame and last frame to OpenAI (Vision)
and get an assessment of scientific accuracy plus suggested changes for the last frame
if it is not accurate (e.g. angle of incidence vs reflection in optics).
"""

import base64
import json
import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are a scientific accuracy reviewer for educational video frames.
You are shown two images: the FIRST FRAME and the LAST FRAME of a short video segment.
Your task is to assess whether the LAST FRAME is scientifically accurate given the topic and the transition from the first frame.

Consider domain-specific rules. For example:
- Reflection of light: angle of incidence must equal angle of reflection; normal, incident ray, and reflected ray must be in the same plane.
- Physics/chemistry: proportions, directions, and labels must be correct.
- Biology: structures and relative positions must be accurate.

If the last frame is scientifically accurate, respond with JSON: {"scientifically_accurate": true, "suggested_changes": null}.
If it is NOT accurate, respond with JSON: {"scientifically_accurate": false, "suggested_changes": "A clear, detailed description of what must be changed in the last frame to make it scientifically correct. This will be used as an I2I prompt to regenerate the last frame. Be specific: angles, positions, labels, proportions."}
Return ONLY valid JSON, no markdown, no extra text."""


def _image_to_base64_data_url(image_path: Path) -> str:
    """Read image and return data URL for OpenAI Vision."""
    data = image_path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    # Assume PNG; could detect from suffix
    return f"data:image/png;base64,{b64}"


def revise_frames_for_scientific_accuracy(
    first_frame_path: Path,
    last_frame_path: Path,
    shot_context: str,
    *,
    topic_hint: str = "",
) -> Tuple[bool, str | None]:
    """
    Send first and last frame to OpenAI Vision and get scientific accuracy assessment.

    Args:
        first_frame_path: Path to the first frame image.
        last_frame_path: Path to the last frame image (candidate).
        shot_context: Short description of what this shot is about (e.g. detailed_description).
        topic_hint: Optional overall topic (e.g. "reflection of light") for context.

    Returns:
        (is_accurate, suggested_changes). If is_accurate is True, suggested_changes is None.
        If False, suggested_changes is a string to use when regenerating the last frame via I2I.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env (required for scientific revision)")

    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    if not last_frame_path.exists():
        raise FileNotFoundError(f"Last frame not found: {last_frame_path}")

    client = OpenAI(api_key=api_key)
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

    first_url = _image_to_base64_data_url(first_frame_path)
    last_url = _image_to_base64_data_url(last_frame_path)

    user_content = [
        {"type": "text", "text": (
            f"Shot context: {shot_context}\n"
            + (f"Overall topic: {topic_hint}\n" if topic_hint else "")
            + "First image = FIRST FRAME, second image = LAST FRAME. Assess if the last frame is scientifically accurate. Return only JSON."
        )},
        {"type": "image_url", "image_url": {"url": first_url}},
        {"type": "image_url", "image_url": {"url": last_url}},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=500,
    )
    raw = (response.choices[0].message.content or "").strip()
    # Strip markdown code block if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return True, None  # On parse error, assume accurate to avoid blocking

    accurate = data.get("scientifically_accurate", True)
    suggested = data.get("suggested_changes")
    if accurate:
        return True, None
    return False, (suggested if isinstance(suggested, str) and suggested.strip() else None)

"""
Reviser (lens): looks at the first frame image and the desired last-frame intent,
and outputs a detailed I2I prompt describing the changes to apply to the first frame
to produce the last frame. Used for continuity when generating the last frame via I2I
instead of pure T2I.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from PIL import Image

load_dotenv()

REVISER_SYSTEM = """You are a visual continuity expert. You are shown the FIRST FRAME image of a video shot and a text description of what the LAST FRAME of that shot should look like (the end state).

Your task: Write a single, detailed I2I (image-to-image) prompt that describes ONLY the changes that must be applied to the first frame to obtain the last frame. The prompt will be sent to an I2I model that takes this first frame as input and your text as the change description.

Include:
- What stays the same (briefly: "Keep the same environment, lighting, camera angle.")
- What changes: new or moved objects, changed positions, new elements (with position, size, color), any lighting or reflection changes.
- Spatial relationships in the final image (where each element is relative to others and to the camera).
- No narrative or meta-commentary. Only the concrete visual change description the I2I model should follow.
Output in one paragraph, suitable for direct use as the I2I prompt."""


def describe_changes_for_i2i(first_frame_path: Path, last_frame_intent: str) -> str:
    """
    Use a VLM (Gemini) to look at the first frame image and the desired last-frame
    intent, and return a detailed I2I prompt describing the changes to apply.

    Args:
        first_frame_path: Path to the first frame image.
        last_frame_intent: Text description of what the last frame should show
            (e.g. last_frame_t2i_prompt from the planner).

    Returns:
        A string suitable as the prompt for I2I(first_frame, prompt).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env (required for reviser)")

    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")

    client = genai.Client(api_key=api_key)
    model = (os.getenv("REVISER_MODEL") or os.getenv("PLANNER_MODEL") or "gemini-2.5-flash").strip()

    img = Image.open(first_frame_path)
    user_content = (
        f"The attached image is the FIRST FRAME of the shot.\n\n"
        f"Desired LAST FRAME (end state) description:\n\"\"\"{last_frame_intent}\"\"\"\n\n"
        f"Output only the I2I change prompt (one detailed paragraph), no preamble."
    )

    response = client.models.generate_content(
        model=model,
        contents=[img, user_content],
        config=GenerateContentConfig(system_instruction=REVISER_SYSTEM),
    )

    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None):
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)
    if not text:
        raise RuntimeError("Reviser returned empty response")
    return text.strip()

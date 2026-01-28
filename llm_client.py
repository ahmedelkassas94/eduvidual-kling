import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def generate_clip_plan_json(prompt: str, target_duration_s: int) -> str:
    """
    Image-based planner (trial contract):
    - Total duration fixed to 20 seconds
    - The 20s explainer is broken into multiple image-based frames
    - Each frame becomes:
        * one generated still image (text-to-image)
        * one short animated segment derived from that still (ffmpeg)
    - No narration/audio
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)
    model = (os.getenv("OPENAI_MODEL") or "gpt-5.2-chat-latest").strip()

    system_prompt = (
        "You are a professional storyboard artist and explainer-video designer. "
        "You design educational visuals for image-based animation pipelines. "
        "Return ONLY valid JSON. No markdown. No explanations. No extra text."
    )

    user_prompt = f"""
Design a 20-second EDUCATIONAL explainer plan that will be produced using
IMAGE-BASED ANIMATION (not direct text-to-video), for the following topic:

TOPIC:
\"\"\"{prompt}\"\"\"

PIPELINE (for your awareness):
- Step 1: Your plan becomes a 20-second master teaching script (no narration audio).
- Step 2: That script is split into a sequence of FRAMES.
- Step 3: Each frame is turned into a single still image via a text-to-image model.
- Step 4: Each still image is animated with simple camera moves (hold, pan, Ken Burns).
- Step 5: All animated segments are concatenated into one 20-second video.

HARD CONSTRAINTS (must follow exactly):
- Total video duration MUST be 20 seconds.
- The sum of all frame duration_s MUST be exactly 20.
- Use between 5 and 10 frames in total (inclusive).
- Output MUST be valid JSON only. No markdown. No comments. No extra keys beyond the schema.
- JSON must use double quotes for all keys/strings. No trailing commas.
- NO humans may appear on screen: no people, no faces, no hands, no silhouettes, no cockpit interior.
- NO logos/brands/watermarks.

EDUCATIONAL STRATEGY (must be used):
- Explain using clear visuals + on-screen text overlays + deliberate pacing.
- On-screen text overlays MUST be:
  - short (3–7 words)
  - large, high-contrast, readable
  - placed where they do NOT cover the main subject.

FRAME STRUCTURE (critical):
- Each frame will be generated as a still image, then animated using image-to-video AI to bring elements to life.
- Design images that contain multiple ELEMENTS/ITEMS that can be animated independently (e.g., arrows, particles, fluids, moving parts, changing values).
- For each frame, you MUST explicitly describe:
  - duration_s: how long this image should stay on screen (integer seconds, 1–5 typical).
  - image_prompt: the full scene as a set of concrete ITEMS/ELEMENTS and their SPATIAL RELATIONSHIPS. Describe what objects exist, their sizes, positions (left/right/center/foreground/background), orientations, colors, and any arrows, labels, or moving parts. IMPORTANT: Include elements that naturally move (arrows, particles, fluids, rotating parts, sliding elements, etc.).
  - animation_type: describes the overall camera/view movement, but individual elements within the image will also animate based on their nature.
  - on_screen_text_overlay: short text or "none".
  - continuity_note: how this frame connects to the next frame visually (which items stay fixed, which items appear/disappear, what stays in the same position, and how the camera framing changes).
  - environment_details: 1–4 concrete environmental details that help ground the scene (background, props, context).

MOTION AND LAYOUT RULES (very important):
- Every image_prompt must include ELEMENTS THAT CAN ANIMATE: arrows that slide, particles that move, fluids that flow, rotating gears, sliding panels, changing numbers/values, etc.
- Describe elements with their intended motion: "arrow pointing right and sliding left to right", "water droplets falling", "particles flowing upward", "rotating wheel", "number increasing from 0 to 100".
- The image-to-video AI will animate these elements naturally, so describe them in a way that suggests their motion.
- animation_type provides overall camera movement context, but the real animation comes from elements moving within the image.

ALLOWED animation_type VALUES:
- "hold"
- "ken_burns_zoom_in"
- "ken_burns_zoom_out"
- "pan_left"
- "pan_right"
- "pan_up"
- "pan_down"

OUTPUT JSON STRUCTURE (must match exactly):
{{
  "style_bible": {{
    "visual_style": ".",
    "camera_rules": ".",
    "lighting_rules": ".",
    "continuity_rules": "."
  }},
  "full_script_20s": ".",
  "frames": [
    {{
      "frame_id": 1,
      "duration_s": 3,
      "image_prompt": ".",
      "animation_type": "ken_burns_zoom_in",
      "on_screen_text_overlay": "none",
      "continuity_note": ".",
      "environment_details": ["...", "..."],
      "lighting": ".",
      "cinematic_look": "."
    }}
    /* Add more frames here so that:
       1) frame_id values are consecutive integers starting from 1; and
       2) the sum of all duration_s values is exactly 20. */
  ]
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # IMPORTANT: gpt-5.2-chat-latest does not support custom temperature
    )

    return response.choices[0].message.content


# Backward-compatible alias (optional)
def generate_scene_plan_json(prompt: str, target_duration_s: int) -> str:
    return generate_clip_plan_json(prompt, target_duration_s)

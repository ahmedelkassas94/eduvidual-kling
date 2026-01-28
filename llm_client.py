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
        "You are a professional storyboard artist and academic explainer-video designer. "
        "You design sophisticated educational visuals for university-level content, explaining advanced concepts "
        "with professional, cinematic quality suitable for higher education. "
        "Return ONLY valid JSON. No markdown. No explanations. No extra text."
    )

    user_prompt = f"""
Design a 20-second ACADEMIC EXPLAINER plan for university-level students that will be produced using
IMAGE-BASED ANIMATION (not direct text-to-video), for the following advanced topic:

TOPIC:
\"\"\"{prompt}\"\"\"

AUDIENCE: University students studying advanced concepts. The visual style must be:
- Professional, sophisticated, and academically rigorous
- Cinematic quality suitable for higher education
- Clean, modern, and visually engaging without being childish or cartoonish
- Appropriate for explaining complex, technical, or abstract concepts

PIPELINE (for your awareness):
- Step 1: Your plan becomes a 20-second master teaching script WITH narration text.
- Step 2: That script is split into a sequence of FRAMES (keyframes from ONE CONTINUOUS SHOT), each with its own narration segment.
- Step 3: Each frame is turned into a single still image via a text-to-image model.
- Step 4: Each still image is animated using image-to-video AI, with each segment CONTINUING from where the previous ended.
- Step 5: Narration audio is generated for each frame segment and synced with the video.
- Step 6: All animated segments with narration are concatenated into one 20-second video that appears as ONE CONTINUOUS TAKE.

CRITICAL: ONE CONTINUOUS TAKE REQUIREMENT (MOST IMPORTANT):
- ALL frames MUST be keyframes from ONE CONTINUOUS CAMERA MOVE (a "one-take" shot).
- The camera must follow ONE CONSISTENT MOVEMENT PATTERN throughout all 20 seconds (e.g., slow dolly left-to-right, slow push-in, slow orbit around subject, slow pan across a scene).
- DO NOT change camera angle, lens, or framing style between frames. Only advance the camera position smoothly along ONE path.
- Elements that appear in frame N must maintain their SPATIAL RELATIONSHIPS in frame N+1, only slightly advanced along the camera's path.
- Think of this as a Steadicam shot: the camera moves smoothly and continuously, never cutting or jumping.
- The style_bible.camera_rules MUST define ONE specific camera movement that applies to ALL frames (e.g., "slow dolly left-to-right at 35mm", "slow push-in at 50mm", "slow orbit clockwise at 24mm").

HARD CONSTRAINTS (must follow exactly):
- Total video duration MUST be 20 seconds.
- The sum of all frame duration_s MUST be exactly 20.
- Use between 5 and 10 frames in total (inclusive).
- Output MUST be valid JSON only. No markdown. No comments. No extra keys beyond the schema.
- JSON must use double quotes for all keys/strings. No trailing commas.
- NO humans may appear on screen: no people, no faces, no hands, no silhouettes, no cockpit interior.
- NO logos/brands/watermarks.

ACADEMIC PRESENTATION STRATEGY (must be used):
- Explain using sophisticated, professional visuals + on-screen text overlays + deliberate pacing.
- Visual style: Professional, clean, modern, cinematic. Think documentary/science channel quality, not cartoon or children's animation.
- On-screen text overlays MUST be:
  - concise and academically appropriate (3–7 words, technical terms allowed)
  - professional typography, high-contrast, highly readable
  - placed where they do NOT cover the main subject
  - styled appropriately for university-level content (e.g., sans-serif modern fonts, not playful fonts)

FRAME STRUCTURE (critical for one-take continuity):
- Each frame is a KEYFRAME from the same continuous camera move, not a separate shot.
- Design images that contain multiple ELEMENTS/ITEMS that can be animated independently (e.g., arrows, particles, fluids, moving parts, changing values).
- For each frame, you MUST explicitly describe:
  - duration_s: how long this keyframe segment lasts (integer seconds, 1–5 typical, can be less than 2 seconds or more).
  - image_prompt: the scene at THIS MOMENT in the continuous shot. Describe concrete ITEMS/ELEMENTS and their SPATIAL RELATIONSHIPS relative to the camera's current position. Include positions (left/right/center/foreground/background), sizes, orientations, colors. IMPORTANT: Include elements that naturally move (arrows, particles, fluids, rotating parts, sliding elements, etc.). The scene should be ADVANCED from the previous frame along the camera path, not a completely new composition.
  - animation_type: MUST be consistent across all frames (or smoothly transition). This describes the camera movement that continues from the previous frame. Use the SAME animation_type for all frames, or a smooth progression (e.g., all "pan_right", or "ken_burns_zoom_in" throughout).
  - on_screen_text_overlay: short text or "none".
  - narration_text: CRITICAL - The narration text to be spoken during this frame segment. Must be concise, clear, and match the duration_s. For a 2-second segment, use 5-8 words. For a 5-second segment, use 12-18 words. Speak at a natural pace (approximately 2.5-3 words per second). The narration should explain what's happening visually in this frame, using academic language appropriate for university students. Make it flow naturally from the previous frame's narration.
  - continuity_note: CRITICAL for one-take. Describe: (1) where the camera was in the previous frame, (2) where it is now (slightly advanced along the same path), (3) which elements maintain their positions relative to the camera, (4) which elements have moved/evolved naturally, (5) how the camera continues to the next frame (same movement, same speed, same lens).
  - environment_details: 1–4 concrete environmental details that help ground the scene (background, props, context). These should be CONSISTENT across frames (same environment, just viewed from different positions along the camera path).

MOTION AND LAYOUT RULES (very important):
- Every image_prompt must include ELEMENTS THAT CAN ANIMATE: arrows that slide, particles that move, fluids that flow, rotating gears, sliding panels, changing numbers/values, etc.
- Describe elements with their intended motion: "arrow pointing right and sliding left to right", "water droplets falling", "particles flowing upward", "rotating wheel", "number increasing from 0 to 100".
- The image-to-video AI will animate these elements naturally, so describe them in a way that suggests their motion.
- animation_type provides overall camera movement context, but the real animation comes from elements moving within the image.
- CRITICAL: All frames must use the SAME animation_type (or a smooth progression) to maintain the one-take effect.

ALLOWED animation_type VALUES (choose ONE and use consistently, or use a smooth progression):
- "hold" (static camera, subtle micro-movements)
- "ken_burns_zoom_in" (continuous slow zoom in)
- "ken_burns_zoom_out" (continuous slow zoom out)
- "pan_left" (continuous slow pan left)
- "pan_right" (continuous slow pan right)
- "pan_up" (continuous slow pan up)
- "pan_down" (continuous slow pan down)

OUTPUT JSON STRUCTURE (must match exactly):
{{
  "style_bible": {{
    "visual_style": "Professional, sophisticated, cinematic quality suitable for university-level academic content. Clean, modern, visually engaging. Think documentary/science channel aesthetic, not cartoon or children's animation. Suitable for explaining advanced, technical, or abstract concepts.",
    "camera_rules": "ONE CONTINUOUS CAMERA MOVE: [describe the single camera movement pattern that applies to ALL frames, e.g., 'slow dolly left-to-right at 35mm lens, maintaining consistent framing and speed throughout all 20 seconds']",
    "lighting_rules": "Professional, cinematic lighting appropriate for academic/educational content. Clean, clear, well-lit to show technical details clearly.",
    "continuity_rules": "All frames are keyframes from the same continuous shot. Camera never cuts or jumps. Elements maintain spatial relationships as camera moves smoothly along one path."
  }},
  "full_script_20s": ".",
  "frames": [
    {{
      "frame_id": 1,
      "duration_s": 3,
      "image_prompt": ".",
      "animation_type": "[USE THE SAME animation_type FOR ALL FRAMES, or a smooth progression]",
      "on_screen_text_overlay": "none",
      "narration_text": "Concise narration text for this segment, matching the duration_s (approximately 2.5-3 words per second).",
      "continuity_note": "This is the START of the continuous shot. Camera begins [describe starting position]. Next frame continues the same camera move, advancing to [describe next position along the path].",
      "environment_details": ["...", "..."],
      "lighting": ".",
      "cinematic_look": "."
    }}
    /* Add more frames here so that:
       1) frame_id values are consecutive integers starting from 1;
       2) the sum of all duration_s values is exactly 20;
       3) ALL frames use the SAME animation_type (or smooth progression);
       4) Each continuity_note describes how this frame continues from the previous frame's camera position;
       5) All frames are keyframes from ONE continuous camera move. */
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

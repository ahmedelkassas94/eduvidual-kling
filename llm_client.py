import json
import os
from pathlib import Path
from typing import Any, Dict

import env_loader  # noqa: F401 - load .env from project root first
from openai import OpenAI

from google import genai
from google.genai import errors as genai_errors
from google.genai.types import GenerateContentConfig

# Default Gemini 2.5 model for script/prompt writing (planner). Override with PLANNER_MODEL in .env.
DEFAULT_PLANNER_MODEL = "gemini-2.5-flash"

GEMINI_KEY_INVALID_HELP = (
    "Gemini rejected your API key (often due to key restrictions).\n"
    "Fix: Create a NEW key at https://aistudio.google.com/apikey\n"
    "  - Do NOT set 'Application restrictions' (no IP address or HTTP referrer limits).\n"
    "  - Put the new key in .env as GEMINI_API_KEY=...\n"
    "Then run the planner again from your terminal."
)


def _gemini_generate(system_prompt: str, user_prompt: str, model: str) -> str:
    """Call Gemini for text generation with system + user prompt. Returns response text."""
    api_key = env_loader.require_env("GEMINI_API_KEY", "Planner requires Gemini for script generation.")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=GenerateContentConfig(system_instruction=system_prompt),
        )
    except genai_errors.ClientError as e:
        err_str = str(e).lower()
        if "api key" in err_str and ("invalid" in err_str or "400" in err_str or "api_key_invalid" in err_str):
            raise RuntimeError(GEMINI_KEY_INVALID_HELP) from e
        raise
    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None):
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)
    if not text:
        raise RuntimeError("Gemini planner returned empty response")
    return text.strip()


# ---------------------------------------------------------
# Claude: generic text generation (script/plan) and I2V
# ---------------------------------------------------------
def _claude_generate(system_prompt: str, user_prompt: str, model: str = None, max_tokens: int = 8192) -> str:
    """Call Claude for text generation. Returns response text."""
    api_key = env_loader.require_env("ANTHROPIC_API_KEY", "Claude requires ANTHROPIC_API_KEY in .env")
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package required. pip install anthropic")
    model = (model or env_loader.get_env("CLAUDE_MODEL") or "claude-sonnet-4-6").strip()
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = ""
    if msg.content and isinstance(msg.content, list):
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "") or ""
                break
    if not text or not text.strip():
        raise RuntimeError("Claude returned empty response")
    return text.strip()


CLAUDE_I2V_SYSTEM = """You are an expert at writing image-to-video (I2V) prompts for generative video models (e.g. Google Veo 3.1).

Your ONLY task: Write a single movement_prompt that will be sent to the I2V model. The prompt must describe how the scene moves from the FIRST frame to the LAST frame of the shot.

Requirements:
- The movement_prompt MUST include (1) NARRATION for this shot — what is being explained (the voiceover content), so the video visuals support and sync with the narration; and (2) ASSISTING VISUAL AIDS for this shot — arrows, highlights, text boxes, text labels as specified for this shot (when present). Describe where they appear, what they point to or say, and how they support the explanation. If the shot has no assisting visual aids, do not invent any.
- The generated video MUST start exactly with the provided first frame (no variation). The generated video MUST end exactly with the provided last frame (no variation). State this explicitly in the prompt.
- Be extremely detailed: for every element give direction, speed, spatial relationships. If something is static, say "remains fixed at ...". Include camera motion.
- Indicate the expected duration only as an APPROXIMATE RANGE based on the provided shot duration (for example, "about 5–7 seconds" for a 6-second shot). This range is soft guidance, not a strict limit. Do not use exact timestamps or frame counts.
- End with: "The video must begin exactly with the provided first frame and end exactly with the provided last frame. Maintain structural integrity and consistency of the source image. No morphing. No hallucinating new objects."
- Output ONLY the movement_prompt text. No preamble, no "Here is the prompt", no markdown."""


def generate_i2v_prompt_claude(
    shot: Dict[str, Any],
    main_script_15s: str,
    first_frame_context: str,
) -> str:
    """
    Call Claude to write the I2V movement_prompt for one shot. Only I2V prompts are written by Claude.
    first_frame_context: for shot 1 use project first_frame_t2i_prompt; for shot N use previous shot's last_frame_t2i_prompt.
    """
    api_key = env_loader.get_env("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to use Claude for I2V prompts. "
            "Get a key at https://console.anthropic.com/"
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package required for Claude I2V prompts. pip install anthropic")

    model = (env_loader.get_env("CLAUDE_MODEL") or "claude-sonnet-4-6").strip()
    shot_id = shot.get("shot_id", 0)
    detailed = (shot.get("detailed_description") or "").strip()
    last_frame_t2i = (shot.get("last_frame_t2i_prompt") or "").strip()
    ingredient_names = shot.get("ingredient_names") or []
    narration_text = (shot.get("narration_text") or "").strip()
    assisting_visual_aids = (shot.get("assisting_visual_aids") or "").strip() or "none"
    on_screen_overlay = (shot.get("on_screen_text_overlay") or "").strip() or "none"
    duration_s = int(shot.get("duration_s", 0) or 0)
    if duration_s > 0:
        approx_min = max(1, duration_s - 1)
        approx_max = duration_s + 1
        duration_range_text = f"about {approx_min}–{approx_max} seconds (soft range, not a strict limit)"
    else:
        duration_range_text = "duration not specified; do not assume a strict time limit"

    user_content = f"""Write the I2V movement_prompt for SHOT {shot_id}.

MAIN SCRIPT (context; extract shot-relevant narration and visual aids from the shot-related part):
{main_script_15s[:2000]}

FIRST FRAME context (scene at start of this shot — T2I description or previous shot's end state):
{first_frame_context[:1500]}

THIS SHOT:
- detailed_description: {detailed}
- last_frame_t2i_prompt (end state of shot): {last_frame_t2i[:1500]}
- ingredient_names: {ingredient_names}

NARRATION for this shot (must be reflected in the I2V prompt so the video supports what is being said):
"{narration_text}"

ASSISTING VISUAL AIDS for this shot (include in the movement_prompt when not "none" — arrows, highlights, text boxes, text labels):
{assisting_visual_aids}

ON-SCREEN TEXT OVERLAY: {on_screen_overlay}

EXPECTED DURATION RANGE for this shot (to guide how you phrase the movement_prompt; this is soft guidance, not a strict limit):
{duration_range_text}

Your movement_prompt MUST include the narration content (what is being explained) and the assisting visual aids (arrows, highlights, labels, etc.) so the I2V model generates a video that syncs with the narration and shows those aids. You may mention duration only as an approximate range (for example, "{duration_range_text}") and must avoid strict timing or exact timestamps. Output only the movement_prompt text."""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=CLAUDE_I2V_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    text = ""
    if msg.content and isinstance(msg.content, list):
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "") or ""
                break
    if not text or not text.strip():
        raise RuntimeError("Claude returned empty I2V movement_prompt")
    return text.strip()


def revise_i2v_prompt_for_length(
    original_prompt: str,
    main_script_15s: str,
    shot: Dict[str, Any],
    max_chars: int,
) -> str:
    """
    Shorten / refine an existing I2V movement_prompt to fit within a soft character limit,
    while keeping the same scene, narration, assisting visual aids, and first/last-frame constraints.

    This is only used as a fallback when Veo returns:
      "Veo operation finished but no video in response. This can happen due to content policy,
       invalid first/last frame, or API limits."
    """
    api_key = env_loader.get_env("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to use Claude for I2V prompt revision. "
            "Get a key at https://console.anthropic.com/"
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package required for Claude I2V prompt revision. pip install anthropic")

    model = (env_loader.get_env("CLAUDE_MODEL") or "claude-sonnet-4-6").strip()
    shot_id = shot.get("shot_id", 0)
    duration_s = int(shot.get("duration_s", 0) or 0)
    narration_text = (shot.get("narration_text") or "").strip()
    assisting_visual_aids = (shot.get("assisting_visual_aids") or "").strip() or "none"
    on_screen_overlay = (shot.get("on_screen_text_overlay") or "").strip() or "none"

    if duration_s > 0:
        approx_min = max(1, duration_s - 1)
        approx_max = duration_s + 1
        duration_range_text = f"about {approx_min}–{approx_max} seconds (soft range, not a strict limit)"
    else:
        duration_range_text = "duration not specified; do not assume a strict time limit"

    user_content = f"""You will REVISE an existing image-to-video (I2V) movement_prompt for SHOT {shot_id}.

The previous prompt caused the Veo API to finish without returning a video (likely due to content policy or prompt limits).
Your job is to rewrite the prompt so that:
- It is SHORTER and more concise (aim to stay at or under {max_chars} characters as a soft cap).
- It keeps the SAME scene, narration, and assisting visual aids intention.
- It still clearly enforces: video MUST start with the provided first frame and end with the provided last frame.
- It still mentions the approximate duration range: {duration_range_text}.

MAIN SCRIPT (context; use only the parts relevant to this shot):
{main_script_15s[:2000]}

CURRENT I2V MOVEMENT PROMPT (to be shortened/refined):
\"\"\"\n{original_prompt.strip()}\n\"\"\"\n

SHOT METADATA (must still be respected):
- Narration: \"{narration_text}\"
- Assisting visual aids: {assisting_visual_aids}
- On-screen text overlay: {on_screen_overlay}
- Expected duration range: {duration_range_text}

Rewrite the movement_prompt so it is more compact and less verbose, but still:
- Describes the motion from FIRST frame to LAST frame.
- Includes the narration content and assisting visual aids where relevant.
- States that the video must begin exactly with the provided first frame and end exactly with the provided last frame.
- Mentions duration only as an approximate range (e.g., \"{duration_range_text}\") and does NOT use strict timing.

Output ONLY the revised movement_prompt text. No preamble, no extra commentary."""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=CLAUDE_I2V_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    text = ""
    if msg.content and isinstance(msg.content, list):
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "") or ""
                break
    if not text or not text.strip():
        raise RuntimeError("Claude returned empty I2V prompt revision")
    return text.strip()


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

    api_key = env_loader.require_env("OPENAI_API_KEY", "OpenAI is required for clip plan generation.")
    client = OpenAI(api_key=api_key)
    model = (env_loader.get_env("OPENAI_MODEL") or "gpt-5.2-chat-latest").strip()

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

CRITICAL: ENVIRONMENT REALISM (MOST IMPORTANT FOR VISUAL QUALITY):
- ANALYZE THE TOPIC to determine if it's a REAL-WORLD/PHYSICAL topic or ABSTRACT/THEORETICAL topic.
- For REAL-WORLD/PHYSICAL topics (e.g., airplanes, biology, chemistry, engineering, physics of everyday objects, natural phenomena):
  * Use PHOTOREALISTIC, REALISTIC ENVIRONMENTS - show real objects in their natural settings
  * Examples: Real commercial airplane flying in blue sky with clouds, real plants in natural environment, real chemical reactions in lab glassware, real mechanical systems in industrial settings
  * NO wind tunnels, NO digital/synthetic environments, NO abstract backgrounds unless the topic specifically requires it
  * The environment should be what you would see in real life: sky for airplanes, ocean for marine biology, lab for chemistry, etc.
- For ABSTRACT/THEORETICAL topics (e.g., quantum mechanics, pure mathematics, theoretical physics, abstract concepts):
  * Use APPROPRIATE ABSTRACT/DIGITAL ENVIRONMENTS - can be stylized, digital, or abstract visualizations
  * Examples: Abstract quantum field visualizations, mathematical graph spaces, theoretical particle interactions, digital/synthetic environments
  * These can include stylized backgrounds, abstract visualizations, or digital environments that help explain the concept
- The environment_details field MUST reflect the chosen environment type (realistic for real-world, abstract for theoretical)
- ALL image_prompt descriptions MUST describe the environment as realistic (for physical topics) or appropriately abstract (for theoretical topics)

PIPELINE (for your awareness):
- Step 1: Your plan becomes a 20-second MASTER SCRIPT that will be followed for the whole video. The script is split into a sequence of SEGMENTS (frames), each with its own narration and visual description.
- Step 2: ONLY THE FIRST FRAME is generated as a still image via text-to-image (Gemini). This image is the first frame of the first video segment. Items inside this image will be moved or animated.
- Step 3: The first image is animated using image-to-video (I2V): items inside move/animate → first video segment. The LAST FRAME of this first video becomes the input for the second video.
- Step 4: The second video is generated by I2V using the last frame of the first video as its first frame. The last frame of the second video becomes the input for the third. This chaining continues for all segments.
- Step 5: Narration audio is generated for each segment and synced with the video.
- Step 6: All segments are concatenated into one final video that follows LONG-TAKE CINEMATOGRAPHY: continuous movement, no visible cuts, long takes, real-time feeling, camera choreography.

CRITICAL: LONG-TAKE CINEMATOGRAPHY (MOST IMPORTANT):
- The final video must feel like ONE CONTINUOUS SHOT: long-take cinematography with continuous movement, NO visible cuts, long takes, real-time feeling, and deliberate camera choreography.
- Each segment CONTINUES directly from where the previous segment ended. The input image for segment N+1 is the last frame of segment N — so there are no jumps or cuts between segments.
- Use continuous camera movements: dolly, pan, orbit, crane, tracking shots, push-in, pull-out. The camera choreography should feel planned and smooth across all segments.
- Elements/objects must maintain spatial and visual consistency as the camera moves. What appears at the end of segment N is exactly what segment N+1 starts from.
- Think of this as a single Steadicam or gimbal shot: the camera moves smoothly through space, never cutting. Real-time feeling: the viewer experiences one unbroken take.
- The style_bible.camera_rules MUST define ONE continuous camera movement pattern that applies across all segments (e.g., "slow dolly forward while panning left, then orbit around subject, maintaining smooth choreography throughout").

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

FRAME STRUCTURE (script segments for long-take chaining):
- The script is split into SEGMENTS. Only the FIRST segment's image is generated by T2I; all later segments get their first frame from the last frame of the previous segment.
- Frame 1 image_prompt: This is the ONLY image generated by text-to-image. Describe the opening scene with DETAILED OBJECT/ITEM DESCRIPTIONS so that items can be animated (moved) within the image. Include elements that naturally move: arrows, particles, fluids, rotating parts, etc.
- Frames 2, 3, ... image_prompt: Describe what this segment should SHOW and how it continues from the previous segment. The I2V model will receive the previous segment's last frame as input; your description guides how the scene should animate and evolve (camera movement, object motion, continuity).
- For each frame, you MUST explicitly describe:
  - duration_s: how long this segment lasts (integer seconds, 1–5 typical, can be less than 2 seconds or more).
  - image_prompt: CRITICAL - For frame 1: full scene description with objects/items and spatial relationships (this is the only T2I image). For frames 2+: describe the scene and action for this segment so that I2V can continue from the previous last frame; include object positions, movements, and how the camera continues.
  - animation_type: How the camera moves in this segment (use same or smooth progression for long-take feel): "hold", "ken_burns_zoom_in", "ken_burns_zoom_out", "pan_left", "pan_right", "pan_up", "pan_down".
  - on_screen_text_overlay: short text or "none".
  - narration_text: Narration to be spoken during this segment. Explain the CONCEPT, not the visuals. Match duration (approx. 2.5-3 words per second). Academic language for university students.
  - continuity_note: CRITICAL for long-take - Describe how this segment continues from the previous: where the camera was, where it moves to, which elements stay consistent, which move or evolve. For frame 1, describe how the opening leads into the next segment.
  - transition_to_next_frame: How this segment leads into the next (camera movement, object movements, final state). For the last frame, describe a natural conclusion.
  - environment_details: 1–4 concrete environmental details. CONSISTENT across segments (same environment). REALISTIC for real-world topics, APPROPRIATELY ABSTRACT for theoretical topics.

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
    "visual_style": "Professional, sophisticated, cinematic quality suitable for university-level academic content. Clean, modern, visually engaging. Think documentary/science channel aesthetic, not cartoon or children's animation. For REAL-WORLD topics: photorealistic, realistic environments showing real objects in natural settings. For ABSTRACT topics: appropriately stylized or digital environments that aid conceptual understanding.",
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
      "transition_to_next_frame": "Describe how this frame transitions to frame 2. Camera movement: [how camera moves]. Object movements: [how objects move]. Final state: [should match frame 2's description].",
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


def generate_ingredients_plan_json(prompt: str, target_duration_s: int = 15) -> str:
    """
    New architecture: 15s shot-by-shot script + named ingredients.
    Script and all prompts are written by CLAUDE (planner). I2V prompts are also filled by Claude later.
    - main_script_15s: full shot-by-shot video description in VERY HIGH detail (every object, movement, intention).
    - ingredients: list of { name, description }.
    - shots: list of shots with detailed_description, movement_prompt, ingredient_names, new_ingredient_names.
    """
    system_prompt = (
        "You are a professional storyboard artist and academic explainer-video designer. "
        "You write the MAIN SCRIPT and ALL prompts that will be sent to image and video models. "
        "Your first-frame and last-frame prompts go to GEMINI (text-to-image). Movement prompts are filled by Claude separately for VEO (image-to-video). "
        "The MAIN SCRIPT must be in VERY HIGH level of detail: name every object and ingredient, describe exact movements and the INTENTION of each movement, and for each item state why it is in the video and what its intention is in that exact shot. "
        "Return ONLY valid JSON. No markdown. No explanations. No extra text."
    )

    user_prompt = f"""
Design a {target_duration_s}-second ACADEMIC EXPLAINER using the INGREDIENTS + SHOTS architecture.

ARCHITECTURE (first frame + last frame per shot, I2V between them):
- Each shot has a FIRST frame and a LAST frame. Both are generated via T2I (or last frame via I2I using the first frame as input for continuity).
- Shot 1: first frame = T2I(first_frame_t2i_prompt). Last frame = T2I(last_frame_t2i_prompt) or I2I(first_frame, reviser-described changes). I2V animates from first to last.
- Shot 2+: first frame = last frame of the previous shot (continuity). Last frame = T2I(last_frame_t2i_prompt) or I2I(first_frame, reviser-described changes). I2V animates from first to last.
- So you must output: (1) first_frame_t2i_prompt (shot 1 only, project-level), (2) for EVERY shot: last_frame_t2i_prompt (full T2I description of the end state). Do NOT write movement_prompt: set it to empty string "" for every shot. I2V (movement) prompts are written separately by Claude.

TARGET MODELS:
- first_frame_t2i_prompt and last_frame_t2i_prompt: GOOGLE GEMINI (text-to-image). Same detail level: environment, every object, spatial relationships. last_frame_t2i_prompt describes the END state of the shot (what the final still should look like).
- movement_prompt: Leave as "" for every shot. Claude will fill these later for VEO (image-to-video).

TOPIC:
\"\"\"{prompt}\"\"\"

ARCHITECTURE (follow exactly):

1. MAIN SCRIPT (main_script_15s) — VERY HIGH LEVEL OF DETAIL. This is the master reference for the whole video. You MUST:
   - List and name EVERY object and every ingredient that appears in the video. For each one, state: (a) what it is, (b) why it is here in the video, (c) what is its intention in the video overall, and (d) in each shot where it appears, what is its intention in that exact shot.
   - Describe the EXACT details of every movement: what moves, in which direction, at what pace, and the INTENTION of that movement (e.g. "the arrow moves left to right to show the flow of electrons" or "the camera slowly pushes in to focus the viewer on the chloroplast").
   - For each shot the main script MUST include: (1) NARRATION — the exact narration spoken during that shot, explaining the content; (2) ASSISTING VISUAL AIDS — when they serve the video, describe arrows, highlights, text boxes, text labels (what they point to, what they say, where they appear). Not every shot needs these; include them only when they help understanding (e.g. arrows showing flow, labels naming parts, highlights drawing attention). Use structure: "SHOT N — Title (Approx. t_start–t_end s)\n\nVisuals: [every object and ingredient; exact movement and intention]\n\nNarration: \"...\"\n\nAssisting visual aids: [arrows, highlights, text boxes, labels — or 'none' if not needed]". The script must be detailed enough that a reader knows exactly what appears, what is said, and what on-screen aids support the explanation.

2. SINGLE FIRST-FRAME T2I PROMPT (for Gemini): You MUST output one combined prompt "first_frame_t2i_prompt" that describes the ENTIRE first frame. This prompt is sent to GEMINI for text-to-image. It must be extremely detailed and include:
   - ENVIRONMENT: Full description of the background/setting (e.g., optics lab, classroom, sky) with lighting, colors, and style.
   - EVERY INGREDIENT in the first shot: For each visual element, give a very detailed description (scientific subject, state, materials, colors, proportions, lighting on that object) so Gemini 2.5 can render it accurately.
   - SPATIAL RELATIONSHIPS: Exact positions relative to each other and to the camera (e.g., "the mirror is placed center-right, vertically; the laser beam enters from the left at mid-height and meets the mirror at its center; camera is directly in front at eye level"). Include left/right/center, foreground/background, scale, and framing.
   Use the same 5-part style where helpful: [Scientific Subject], [Action/State], [Style/Medium], [Lighting & Texture], [Framing]. Output this as one continuous "first_frame_t2i_prompt" string.

3. INGREDIENTS: List only the NAMES and short "description" of each visual element (for I2V reference). No per-ingredient t2i_prompt. Give each a SHORT NAME and a one- or two-sentence "description" (used when the ingredient appears or is newly introduced in I2V prompts). Do NOT create ingredients for arrows/lines/labels—those are described only in movement_prompt.

4. SHOTS: Each shot has: time_range, duration_s, detailed_description, last_frame_t2i_prompt, movement_prompt, ingredient_names, new_ingredient_names, narration_text, on_screen_text_overlay, assisting_visual_aids.
   - narration_text: The exact narration spoken during this shot (explaining the content). Must match the Narration for this shot in main_script_15s.
   - assisting_visual_aids: Description of arrows, highlights, text boxes, text labels for this shot when they serve the video (e.g. "Arrows pointing from sun to leaf; label 'chloroplast' on green ovals; highlight around the reaction site"). Use "none" or empty when not needed.
   - last_frame_t2i_prompt: FULL T2I prompt for the END state of this shot (for Gemini 2.5). Same detail as first_frame_t2i_prompt: environment, every object in final positions, spatial relationships, lighting. This image becomes the first frame of the next shot.
   - movement_prompt: Set to "" (empty string) for every shot. Do not write I2V prompts; Claude will generate them separately.

RULES:
- Total duration MUST be exactly {target_duration_s} seconds. Sum of shot duration_s = {target_duration_s}.
- Each shot's duration_s MUST be between 5 and 8 seconds (inclusive). Use 2–4 shots so that every shot is 5–8s (e.g. for 15s: two shots of 7s and 8s, or three shots of 5s each). No shot shorter than 5s or longer than 8s.
- ingredient_names in each shot must be a subset of the names in ingredients (exact match).
- new_ingredient_names for shot 1: either list all ingredients that appear in shot 1, or leave empty and put them in ingredient_names. For shot 2+: only list ingredients that FIRST APPEAR in this shot.
- NO humans: no people, no faces, no hands. NO logos/watermarks.
- For REAL-WORLD topics use PHOTOREALISTIC descriptions; for ABSTRACT topics use appropriate stylized/digital descriptions.
STYLIST: The style_bible must include "style_suffix": a short phrase appended to every t2i_prompt for visual consistency (e.g., "muted professional teal and grey palette, 4k, cinematic"). The pipeline will append it automatically; you only need to define it once here.

OUTPUT JSON (must match exactly):
{{
  "main_script_15s": "VERY HIGH DETAIL: For each shot: every object and ingredient by name; exact movement and intention; then Visuals, then Narration (exact words spoken), then Assisting visual aids (arrows, highlights, text boxes, labels when they serve the video, or 'none'). Full shot-by-shot.",
  "first_frame_t2i_prompt": "One long paragraph for GEMINI 2.5: environment (full description), then each ingredient in the first shot with full visual detail, then exact spatial relationships between all of them and camera position. Professional scientific illustration, 8k, clear lighting.",
  "style_bible": {{
    "visual_style": "Professional, sophisticated, cinematic. Documentary/science channel.",
    "camera_rules": "One continuous camera move across all shots. No cuts.",
    "lighting_rules": "Professional, clear, well-lit.",
    "continuity_rules": "Same continuous shot. Elements consistent across shots.",
    "style_suffix": "Muted professional teal and grey palette, 4k, cinematic."
  }},
  "ingredients": [
    {{ "name": "mirror", "description": "Flat plane mirror, vertically oriented, sharp edges.", "t2i_prompt": null, "matplotlib_code": null }},
    {{ "name": "laser_beam", "description": "Narrow red laser beam, coherent, traveling toward mirror.", "t2i_prompt": null, "matplotlib_code": null }}
  ],
  "shots": [
    {{
      "shot_id": 1,
      "time_range": "0-6s",
      "duration_s": 6,
      "detailed_description": "What happens in this shot.",
      "last_frame_t2i_prompt": "Same environment as first frame. Mirror: center-right, vertical, static. Laser beam: now visible hitting mirror at center and reflected beam visible on the other side, same lighting and style. Camera position unchanged. Professional scientific illustration, 8k.",
      "movement_prompt": "",
      "ingredient_names": ["mirror", "laser_beam"],
      "new_ingredient_names": [],
      "narration_text": "Narration for this shot.",
      "on_screen_text_overlay": "none",
      "assisting_visual_aids": "Arrows showing incident and reflected beam; label 'angle of incidence' and 'angle of reflection' near the normal. Or none if not needed.",
      "i2i_spatial_prompt": ""
    }},
    {{
      "shot_id": 2,
      "time_range": "6-12s",
      "duration_s": 6,
      "detailed_description": "Normal line and angle arcs appear.",
      "last_frame_t2i_prompt": "Same scene. Mirror and laser unchanged. NEW: thin dashed normal line perpendicular to mirror at reflection point; two small angle arcs with arrowheads. Same style and lighting.",
      "movement_prompt": "",
      "ingredient_names": ["mirror", "laser_beam", "normal_line"],
      "new_ingredient_names": ["normal_line"],
      "narration_text": "Narration for this shot.",
      "on_screen_text_overlay": "none",
      "assisting_visual_aids": "none",
      "i2i_spatial_prompt": ""
    }}
  ]
}}
Use 2–4 shots. Every shot MUST have duration_s between 5 and 8 (inclusive). Sum of duration_s = {target_duration_s}. Every shot MUST have last_frame_t2i_prompt (full T2I description of the end state). Every shot MUST have movement_prompt set to "" (empty string); Claude will fill I2V prompts separately.
"""

    return _claude_generate(system_prompt, user_prompt, max_tokens=16384)


def generate_audio_revision_json(main_script_15s: str, shots: list) -> str:
    """
    Revise voiceover and plan sound design for the full video based on the main script.
    Returns JSON with:
      - revised_narration: [ { shot_id, revised_text }, ... ]
      - sound_design: [ { shot_id, start_s, end_s, description }, ... ]
    """
    api_key = env_loader.require_env("OPENAI_API_KEY", "OpenAI is required for audio revision.")
    client = OpenAI(api_key=api_key)
    model = (env_loader.get_env("OPENAI_MODEL") or "gpt-4o").strip()

    shots_summary = []
    t = 0
    for s in shots:
        shot_id = s.get("shot_id")
        dur = int(s.get("duration_s", 3))
        shots_summary.append({
            "shot_id": shot_id,
            "time_range": f"{t}-{t + dur}s",
            "duration_s": dur,
            "current_narration": (s.get("narration_text") or "").strip(),
            "visual_description": (s.get("detailed_description") or "").strip(),
        })
        t += dur

    system_prompt = (
        "You are a professional video editor and sound designer for academic explainer videos. "
        "You revise voiceover for clarity and coherence, and you plan ambient/sound-design that matches what is happening on screen. "
        "Return ONLY valid JSON. No markdown. No explanations."
    )
    user_prompt = f"""
Given the MAIN SCRIPT and the current shot-by-shot plan, output two things:

1. REVISED NARRATION: For each shot, provide improved voiceover text that explains the content clearly and matches the main script. Keep similar length (about 2.5-3 words per second). Make the full narration flow as one coherent explanation.

2. SOUND DESIGN: For each shot, describe the SOUNDS that should be heard while that part of the video plays (ambient, Foley, or abstract). Examples: "gentle wind, leaves rustling", "soft whoosh, subtle sci-fi hum", "quiet indoor ambience", "nature sounds, birds distant". One short phrase per shot.

MAIN SCRIPT (full video description):
\"\"\"{main_script_15s}\"\"\"

SHOTS (with current narration and visual description):
{json.dumps(shots_summary, indent=2)}

Output JSON exactly in this form (no extra keys):
{{
  "revised_narration": [
    {{ "shot_id": 1, "revised_text": "Revised voiceover for this shot." }},
    ...
  ],
  "sound_design": [
    {{ "shot_id": 1, "start_s": 0, "end_s": 3, "description": "gentle wind, leaves rustling" }},
    ...
  ]
}}
- revised_narration: one entry per shot, same shot_ids as input. revised_text = the new voiceover for that shot.
- sound_design: one entry per shot. start_s and end_s = start/end time in seconds (cumulative). description = short phrase for what sounds to play during that segment.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# Backward-compatible alias (optional)
def generate_scene_plan_json(prompt: str, target_duration_s: int) -> str:
    return generate_clip_plan_json(prompt, target_duration_s)

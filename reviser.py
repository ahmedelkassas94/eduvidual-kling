"""
Reviser (lens): looks at the first frame image and the desired last-frame intent,
and outputs a detailed I2I prompt describing the changes to apply to the first frame
to produce the last frame. Used for continuity when generating the last frame via I2I
instead of pure T2I.

Also contains the context-fit reviser: checks images against the full video script
and objectives to ensure they serve the narrative purpose of the 15s video.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from PIL import Image

load_dotenv()

REVISER_SYSTEM = """You are an I2I (image-to-image) prompt writer. I2I takes the FIRST FRAME image and your prompt to produce a modified image. Your prompt must describe CHANGES to that first frame, not a full description of another picture.

STEP 1 — ANALYZE THE FIRST FRAME:
Look at the attached image (the FIRST FRAME). Identify EVERY visible object, element, or region. Assign each a specific name or label (e.g. "the stroma", "the grana stacks", "the thylakoid discs", "the Photosystem II label"). Note their positions and the current location/setting (e.g. "inside a chloroplast", "inside a cell", "a circuit diagram").

STEP 2 — DECIDE: SAME LOCATION OR DIFFERENT PLACE?
- SAME LOCATION: The first frame and the desired last frame are in the SAME setting (e.g. both inside the chloroplast, both inside the same cell, same diagram, same room). The last frame may add new elements (e.g. Calvin cycle, new labels, new molecules) or change existing ones, but we have NOT zoomed or cut to a different place.
- DIFFERENT PLACE: The intent explicitly describes a transition to a different scale or location (e.g. "zoom from the cell into the interior of the chloroplast", "we are now inside the chloroplast", "cut to a different diagram"). The first frame shows place A and the last frame shows place B.

STEP 3 — WRITE THE I2I PROMPT:
- If SAME LOCATION (same setting as first frame): You MUST write IMPERATIVE CHANGES only. Use the exact object names from Step 1. Say what to Add, Move, Make, Remove, Intensify (e.g. "Add a large glowing teal circular arrow (Calvin cycle) in the left-center of the stroma. Add three CO2 icons entering the top node. Keep the grana stacks and stroma as they are; add a Glucose icon at the bottom node."). Do NOT output a full scene description. This preserves video continuity.
- If DIFFERENT PLACE (intent clearly says we have moved to a new location/scale): You may output a full standalone description of the target frame.

SAME-LOCATION RULE (critical for continuity): In multi-shot videos, the first frame of shot N is the last frame of shot N-1. So when the first frame already shows "inside chloroplast" and the desired last frame is also "inside chloroplast" with new elements (e.g. Calvin cycle, wider stroma view), you MUST use imperative form. Writing a full description from scratch would break continuity. When in doubt, prefer IMPERATIVE CHANGES.

FORBIDDEN: Do NOT output a full standalone description (e.g. "A highly detailed photorealistic illustration showing the interior of a chloroplast...") when the first frame is already in that same location. Always use imperative: Add X, Make Y, Keep Z.

OUTPUT FORMAT:
- Output ONLY the I2I prompt (imperative changes, or full description only when we have clearly moved to a different place).
- No preamble. If the intent includes "Required corrections", express them as imperative commands."""


def describe_changes_for_i2i(first_frame_path: Path, last_frame_intent: str) -> str:
    """
    Use a VLM (Gemini) to:
    1. Analyze the first frame and identify all objects with specific names/labels.
    2. Write an I2I prompt using those exact names in imperative form (or full description for major transitions).

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
        "The attached image is the FIRST FRAME (this shot's starting image). I2I will modify it using your prompt.\n\n"
        "STEP 1: Identify every object and the LOCATION/SETTING (e.g. 'inside a chloroplast', 'stroma', 'grana stacks', 'thylakoid discs').\n\n"
        "STEP 2: Read the desired last frame below. Is it in the SAME location as the first frame (e.g. still inside chloroplast, same cell, same diagram)? "
        "If YES → you MUST output IMPERATIVE CHANGES only: Add X, Make Y, Keep Z, using the object names from Step 1. Do NOT write a full scene description.\n"
        "If the intent clearly says we have MOVED to a different place (e.g. first frame was a cell, last frame is 'interior of chloroplast') → you may output a full target description.\n\n"
        "Desired LAST FRAME (end state):\n"
        f"\"\"\"{last_frame_intent}\"\"\"\n\n"
        "Output ONLY the I2I prompt. When same location: imperative changes only. No preamble."
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


# ---------------------------------------------------------
# I2V PROMPT REVISER (exact first/last frame)
# ---------------------------------------------------------
I2V_REVISER_SYSTEM = """You are an expert at writing image-to-video (I2V) prompts for generative video models (e.g. Veo, Wan).

Your task: Revise the given I2V movement prompt so that it EXPLICITLY and UNAMBIGUOUSLY requires:
1. The generated video MUST start with the exact first frame provided as input — the first frame of the output video must match the input image exactly, with no variation or re-interpretation.
2. The generated video MUST end with the exact last frame provided as the target — the final frame of the output video must match the target end-state image exactly.

Keep the rest of the prompt (motion, camera, objects, timing) unchanged. Add or strengthen sentences that state the exact start/end frame requirement. Output ONLY the revised prompt text, no preamble or explanation."""


def revise_i2v_prompt_for_exact_frames(movement_prompt: str) -> str:
    """
    Revise an I2V movement prompt so it explicitly requires the generated video
    to start with the exact first frame and end with the exact last frame.
    Uses Gemini to rewrite the prompt while preserving motion/camera detail.
    """
    if not (movement_prompt or "").strip():
        return movement_prompt

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return movement_prompt  # No API key: return as-is

    try:
        client = genai.Client(api_key=api_key)
        model = (os.getenv("REVISER_MODEL") or os.getenv("PLANNER_MODEL") or "gemini-2.5-flash").strip()
        user_content = (
            "Revise the following I2V (image-to-video) movement prompt so it explicitly requires "
            "that the generated video MUST start exactly with the provided first frame and MUST end "
            "exactly with the provided last frame. Keep all motion, camera, and object details. "
            "Output only the revised prompt.\n\n"
            f"Current prompt:\n\"\"\"{movement_prompt.strip()}\"\"\""
        )
        response = client.models.generate_content(
            model=model,
            contents=[user_content],
            config=GenerateContentConfig(system_instruction=I2V_REVISER_SYSTEM),
        )
        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None) and response.candidates:
            c = response.candidates[0]
            parts = getattr(getattr(c, "content", None), "parts", None) or []
            if parts:
                text = getattr(parts[0], "text", None)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    return movement_prompt


# ---------------------------------------------------------
# CONTEXT-FIT REVISER: ensure images serve the video narrative
# ---------------------------------------------------------
CONTEXT_FIRST_FRAME_SYSTEM = """You are a video frame reviewer for educational explainer videos.

You are shown:
1. The FIRST FRAME image of the entire video (shot 1, first frame — the opening image).
2. The FULL SCRIPT of the video (all shots, 15 seconds total).
3. The main OBJECTIVE of the video.

Your task: Determine if this first frame image fits the context and narrative purpose of the video.

Check:
- Does the image correctly introduce the subject of the video?
- Does it set up the visual elements that will be used/animated in subsequent shots?
- Does it match the style, setting, and tone described in the script?
- Is there anything missing, wrong, or out of place that would confuse viewers or break continuity?

Return ONLY a JSON object with these keys (no markdown, no code fence):
"fits_context" (boolean): true if the image fits well, false if it needs changes.
"changes_needed" (string or null): if fits_context is false, describe the specific changes needed to make this image fit the video context. Be concrete and actionable (e.g. "Add X element", "Change Y to Z", "Remove W"). If fits_context is true, set to null."""

CONTEXT_SHOT_FRAMES_SYSTEM = """You are a video frame reviewer for educational explainer videos.

You are shown:
1. The FIRST FRAME image of a shot (this is the starting point for the shot).
2. The LAST FRAME image of the same shot (this is where the shot ends; this image becomes the first frame of the next shot).
3. The shot number and total shots.
4. The FULL SCRIPT of the video (all shots, 15 seconds total).
5. The main OBJECTIVE of the video.

Your task: Determine if these two images (first and last frame of this shot) fit the context and narrative purpose of the video.

Check:
- Do the images correctly represent what this shot should show according to the script?
- Is the transition from first to last frame logical and does it serve the educational purpose?
- Do the images match the style, setting, and elements described in the script?
- Does the last frame set up the next shot properly (continuity)?
- Is there anything missing, wrong, or out of place that would confuse viewers?

IMPORTANT: You can only suggest changes to the LAST FRAME. The first frame is fixed (it's either the opening T2I image or the previous shot's last frame).

Return ONLY a JSON object with these keys (no markdown, no code fence):
"fits_context" (boolean): true if both images fit well, false if the last frame needs changes.
"changes_needed" (string or null): if fits_context is false, describe the specific changes needed for the LAST FRAME to make it fit the video context. Be concrete and actionable. If fits_context is true, set to null."""


def _extract_json_obj(text: str) -> dict:
    """Extract JSON object from model output, stripping markdown if present."""
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1)
    return json.loads(text)


def revise_first_frame_for_context(
    first_frame_path: Path,
    full_script: str,
    video_objective: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check if the first frame of the video (shot 1, first frame) fits the video context.
    
    Args:
        first_frame_path: Path to the first frame image.
        full_script: The full 15s script of all shots.
        video_objective: The main objective/topic of the video.
    
    Returns:
        (fits_context, changes_needed). If fits_context is True, changes_needed is None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env (required for context reviser)")

    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")

    client = genai.Client(api_key=api_key)
    model = (os.getenv("REVISER_MODEL") or os.getenv("PLANNER_MODEL") or "gemini-2.5-flash").strip()

    img = Image.open(first_frame_path)
    user_content = (
        "This is the FIRST FRAME of the entire video (shot 1, opening image).\n\n"
        f"VIDEO OBJECTIVE:\n{video_objective[:1000]}\n\n"
        f"FULL SCRIPT (all shots):\n{full_script[:3000]}\n\n"
        "Does this image fit the video context? Return JSON only."
    )

    response = client.models.generate_content(
        model=model,
        contents=[img, user_content],
        config=GenerateContentConfig(system_instruction=CONTEXT_FIRST_FRAME_SYSTEM),
    )

    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None) and response.candidates:
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)
    if not text or not text.strip():
        raise RuntimeError("Context reviser returned empty response")

    out = _extract_json_obj(text)
    fits = bool(out.get("fits_context", True))
    changes = out.get("changes_needed") or None
    return fits, changes


# ---------------------------------------------------------
# I2V PROMPT-TO-FRAMES VERIFICATION
# ---------------------------------------------------------
I2V_PROMPT_VERIFY_SYSTEM = """You are a video continuity reviewer. You are shown:
1. The FIRST FRAME image of a video shot (starting point).
2. The LAST FRAME image of the same shot (ending point).
3. The I2V (image-to-video) movement prompt that will be used to generate the video transitioning from first to last frame.

Your task: Determine if the I2V movement prompt correctly and accurately describes the visual transition between the first frame and the last frame.

Check:
- Does the prompt describe the actual changes visible between the two images?
- Does the prompt reference the correct objects (matching what's actually in the images)?
- Does the prompt describe the motion/transition in a way that would produce a video going from first to last frame?
- Are there any hallucinations in the prompt (describing things not in the images)?
- Are there any missing descriptions (changes visible in images but not mentioned in prompt)?

Return ONLY a JSON object with these keys (no markdown, no code fence):
"prompt_matches_frames" (boolean): true if the I2V prompt correctly describes the transition, false if there are issues.
"issues_found" (string or null): if prompt_matches_frames is false, describe what's wrong (e.g. "prompt mentions X but image shows Y", "transition from A to B is not described").
"""

I2V_PROMPT_FIX_SYSTEM = """You are a video continuity fixer. You have determined that an I2V movement prompt does NOT correctly describe the transition between first and last frame images.

You are now given the FULL SCRIPT of the video and asked to diagnose and fix the issue.

Analyze:
1. Is the issue in the I2V PROMPT? (prompt doesn't match what should happen according to script and images)
2. Is the issue in the LAST FRAME IMAGE? (image doesn't show what it should according to script)
3. Or BOTH?

Return ONLY a JSON object with these keys (no markdown, no code fence):
"fix_prompt" (boolean): true if the I2V prompt needs to be revised.
"fix_last_frame" (boolean): true if the last frame image needs to be regenerated.
"prompt_revision" (string or null): if fix_prompt is true, provide the COMPLETE revised I2V movement prompt (ready to use, imperative form, referencing objects by name).
"last_frame_changes" (string or null): if fix_last_frame is true, describe the specific changes needed for the last frame image (to be used as I2I correction instructions).
"""


def verify_i2v_prompt_matches_frames(
    first_frame_path: Path,
    last_frame_path: Path,
    i2v_prompt: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check if the I2V movement prompt correctly describes the transition between first and last frame.
    
    Args:
        first_frame_path: Path to the first frame image.
        last_frame_path: Path to the last frame image.
        i2v_prompt: The I2V movement prompt to verify.
    
    Returns:
        (prompt_matches, issues_found). If prompt_matches is True, issues_found is None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env (required for I2V verification)")

    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    if not last_frame_path.exists():
        raise FileNotFoundError(f"Last frame not found: {last_frame_path}")

    client = genai.Client(api_key=api_key)
    model = (os.getenv("REVISER_MODEL") or os.getenv("PLANNER_MODEL") or "gemini-2.5-flash").strip()

    img_first = Image.open(first_frame_path)
    img_last = Image.open(last_frame_path)
    user_content = (
        "Image 1: FIRST FRAME (starting point).\n"
        "Image 2: LAST FRAME (ending point).\n\n"
        f"I2V MOVEMENT PROMPT:\n\"\"\"\n{i2v_prompt.strip()}\n\"\"\"\n\n"
        "Does this I2V prompt correctly describe the visual transition from first to last frame? Return JSON only."
    )

    response = client.models.generate_content(
        model=model,
        contents=[img_first, img_last, user_content],
        config=GenerateContentConfig(system_instruction=I2V_PROMPT_VERIFY_SYSTEM),
    )

    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None) and response.candidates:
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)
    if not text or not text.strip():
        raise RuntimeError("I2V verification returned empty response")

    out = _extract_json_obj(text)
    matches = bool(out.get("prompt_matches_frames", True))
    issues = out.get("issues_found") or None
    return matches, issues


def fix_i2v_prompt_and_last_frame(
    first_frame_path: Path,
    last_frame_path: Path,
    i2v_prompt: str,
    issues_found: str,
    full_script: str,
    video_objective: str,
) -> dict:
    """
    Given issues with the I2V prompt not matching the frames, diagnose and provide fixes.
    
    Args:
        first_frame_path: Path to the first frame image.
        last_frame_path: Path to the last frame image.
        i2v_prompt: The current I2V movement prompt.
        issues_found: Description of the issues found.
        full_script: The full 15s script of all shots.
        video_objective: The main objective/topic of the video.
    
    Returns:
        dict with keys:
        - fix_prompt (bool): whether the I2V prompt needs revision
        - fix_last_frame (bool): whether the last frame needs regeneration
        - prompt_revision (str or None): the revised I2V prompt if fix_prompt is True
        - last_frame_changes (str or None): changes for last frame if fix_last_frame is True
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env (required for I2V fix)")

    client = genai.Client(api_key=api_key)
    model = (os.getenv("REVISER_MODEL") or os.getenv("PLANNER_MODEL") or "gemini-2.5-flash").strip()

    img_first = Image.open(first_frame_path)
    img_last = Image.open(last_frame_path)
    user_content = (
        "Image 1: FIRST FRAME (starting point).\n"
        "Image 2: LAST FRAME (ending point).\n\n"
        f"CURRENT I2V MOVEMENT PROMPT:\n\"\"\"\n{i2v_prompt.strip()}\n\"\"\"\n\n"
        f"ISSUES FOUND:\n{issues_found}\n\n"
        f"VIDEO OBJECTIVE:\n{video_objective[:1000]}\n\n"
        f"FULL SCRIPT (all shots):\n{full_script[:3000]}\n\n"
        "In the context of this video, identify whether the issue is in the I2V prompt, the last frame image, or both. "
        "Provide the fixes. Return JSON only."
    )

    response = client.models.generate_content(
        model=model,
        contents=[img_first, img_last, user_content],
        config=GenerateContentConfig(system_instruction=I2V_PROMPT_FIX_SYSTEM),
    )

    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None) and response.candidates:
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)
    if not text or not text.strip():
        raise RuntimeError("I2V fix returned empty response")

    out = _extract_json_obj(text)
    return {
        "fix_prompt": bool(out.get("fix_prompt", False)),
        "fix_last_frame": bool(out.get("fix_last_frame", False)),
        "prompt_revision": out.get("prompt_revision") or None,
        "last_frame_changes": out.get("last_frame_changes") or None,
    }


def revise_shot_frames_for_context(
    first_frame_path: Path,
    last_frame_path: Path,
    shot_id: int,
    total_shots: int,
    full_script: str,
    video_objective: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check if the first and last frame of a shot fit the video context.
    Only the last frame can be changed (first frame is fixed).
    
    Args:
        first_frame_path: Path to the shot's first frame image.
        last_frame_path: Path to the shot's last frame image.
        shot_id: The shot number (1-indexed).
        total_shots: Total number of shots in the video.
        full_script: The full 15s script of all shots.
        video_objective: The main objective/topic of the video.
    
    Returns:
        (fits_context, changes_needed). If fits_context is True, changes_needed is None.
        changes_needed applies to the LAST FRAME only.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env (required for context reviser)")

    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    if not last_frame_path.exists():
        raise FileNotFoundError(f"Last frame not found: {last_frame_path}")

    client = genai.Client(api_key=api_key)
    model = (os.getenv("REVISER_MODEL") or os.getenv("PLANNER_MODEL") or "gemini-2.5-flash").strip()

    img_first = Image.open(first_frame_path)
    img_last = Image.open(last_frame_path)
    user_content = (
        f"SHOT {shot_id} of {total_shots}.\n"
        "Image 1: FIRST FRAME of this shot (starting point).\n"
        "Image 2: LAST FRAME of this shot (end point; becomes first frame of next shot).\n\n"
        f"VIDEO OBJECTIVE:\n{video_objective[:1000]}\n\n"
        f"FULL SCRIPT (all shots):\n{full_script[:3000]}\n\n"
        "Do these images fit the video context? If changes are needed, describe changes for the LAST FRAME only. Return JSON only."
    )

    response = client.models.generate_content(
        model=model,
        contents=[img_first, img_last, user_content],
        config=GenerateContentConfig(system_instruction=CONTEXT_SHOT_FRAMES_SYSTEM),
    )

    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None) and response.candidates:
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)
    if not text or not text.strip():
        raise RuntimeError("Context reviser returned empty response")

    out = _extract_json_obj(text)
    fits = bool(out.get("fits_context", True))
    changes = out.get("changes_needed") or None
    return fits, changes

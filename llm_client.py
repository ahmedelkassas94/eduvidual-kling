import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------
# Narration-first pipeline: 24s audio script → TTS → transcribe → main script from timeline
# ---------------------------------------------------------
MAX_NARRATION_DURATION_S = 24  # Max narration (and video) length in seconds
NARRATION_WORDS_MAX = (MAX_NARRATION_DURATION_S * 3)  # ~3 words/sec → 72 words for 24s
NARRATION_WORDS_MIN = (MAX_NARRATION_DURATION_S * 2)  # ~2 words/sec → 48 words

NARRATION_24S_SYSTEM = f"""You are an expert writer of educational narration scripts for short explainer videos.
Your ONLY task is to write a single, coherent NARRATION SCRIPT that will be spoken aloud (voiceover).
CRITICAL CONSTRAINT: The narration MUST be {MAX_NARRATION_DURATION_S} SECONDS OR LESS when read aloud at a natural pace (approximately 2.5 to 3 words per second). So use at most {NARRATION_WORDS_MIN} to {NARRATION_WORDS_MAX} words. Do NOT exceed this. There will be no trimming later—the generated audio must be at most {MAX_NARRATION_DURATION_S} seconds from the start. Write concisely.
Output ONLY the narration text. No preamble, no "Here is the script", no timestamps, no stage directions."""


def generate_narration_script_24s(topic: str) -> str:
    """
    Generate a narration script that explains the topic. When spoken at natural pace,
    the script MUST be 24 seconds or less (approx 48–72 words). No trimming; TTS will
    generate audio that must not exceed 24s.
    """
    user = f"""Write a clear, educational narration script for a short explainer video on this topic:

\"\"\"{topic}\"\"\"

The script will be converted to speech. It MUST be {MAX_NARRATION_DURATION_S} seconds or less when read aloud (about 2.5–3 words per second → max {NARRATION_WORDS_MIN}–{NARRATION_WORDS_MAX} words). Do not exceed {MAX_NARRATION_DURATION_S} seconds of spoken content. Output only the narration text."""
    return _claude_generate(NARRATION_24S_SYSTEM, user, max_tokens=1024)


def transcribe_audio_with_timestamps(audio_path: Path) -> List[Dict[str, Any]]:
    """
    Transcribe audio file and return segments with exact start_s and end_s for each sentence/segment.
    Uses OpenAI Whisper. Returns list of { "start_s": float, "end_s": float, "text": str }.
    """
    api_key = env_loader.require_env("OPENAI_API_KEY", "Transcription requires OPENAI_API_KEY (Whisper).")
    client = OpenAI(api_key=api_key)
    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    with open(path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    segments = getattr(transcription, "segments", None) or []
    out = []
    for s in segments:
        start_s = float(s.get("start", 0) if isinstance(s, dict) else getattr(s, "start", 0))
        end_s = float(s.get("end", 0) if isinstance(s, dict) else getattr(s, "end", 0))
        text = (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "") or "").strip()
        if text:
            out.append({"start_s": start_s, "end_s": end_s, "text": text})
    return out


MAIN_SCRIPT_FROM_TRANSCRIPTION_SYSTEM = """You are a professional storyboard artist for academic explainer videos.
You are given: (1) the topic, and (2) a transcription of the narration with exact start and end times in seconds for each sentence.
Your task is to write the MAIN SCRIPT for the video: a detailed description of what the VIDEO shows, aligned to the narration timeline.
For each time range (from the transcription), describe in detail: what appears on screen, what moves, camera, visuals, and how they support the narration in that moment. Include assisting visual aids (arrows, labels, highlights) when they help. Do not repeat the narration verbatim; describe the visuals. The main script is the master reference for the whole video. Output only the main script text, no JSON."""


def generate_main_script_from_transcription(topic: str, segments: List[Dict[str, Any]]) -> str:
    """
    Generate the main script (video description) from the topic and timed transcription.
    The main script describes the video in detail, aligned to the timeline of each sentence.
    """
    seg_lines = "\n".join(
        f"[{s['start_s']:.1f}s - {s['end_s']:.1f}s] {s['text']}" for s in segments
    )
    user = f"""Topic: {topic}

Transcription (exact start and end time for each sentence):
{seg_lines}

Write the MAIN SCRIPT for the video: for each time range above, describe in detail what the video should show (visuals, movement, camera, elements). Align the video description to this timeline. Include narration and assisting visual aids where relevant. Output only the main script."""
    return _claude_generate(MAIN_SCRIPT_FROM_TRANSCRIPTION_SYSTEM, user, max_tokens=8192)


# Legacy: older pipelines used fixed 8-second segments.
# New Kling 3.0 pipeline uses variable shot durations (3–15s) and first-frame-only chaining.
CLIP_DURATION_S = 8


def generate_shots_and_ingredients_from_main_script(
    main_script_15s: str, segments: List[Dict[str, Any]], topic: str
) -> Dict[str, Any]:
    """
    From the main script and timed transcription, produce the full plan JSON:
    style_bible, ingredients, shots with variable duration (3–15s), first_frame_t2i_prompt.

    Shot durations must be chosen based on the narration timeline; narration_text per shot
    is from transcription segments that fall in that shot's time_range.
    """
    import math
    seg_lines = "\n".join(
        f"[{s['start_s']:.1f}s - {s['end_s']:.1f}s] {s['text']}" for s in segments
    )
    total_duration = max(s["end_s"] for s in segments) if segments else MAX_NARRATION_DURATION_S
    # We require integer shot durations that sum exactly to this rounded total.
    total_duration_int = int(math.ceil(total_duration))

    system = (
        "You are a professional storyboard artist. You output ONLY valid JSON. No markdown. "
        "Given the main script and the timed transcription, produce: style_bible, ingredients, shots, first_frame_t2i_prompt. "
        "This video is divided into a SHOT LIST. "
        f"The sum of ALL shot duration_s MUST be EXACTLY {total_duration_int} seconds. "
        "Each shot duration_s MUST be an integer between 3 and 15 seconds (inclusive). "
        "You MUST set time_range cumulatively starting at 0s, e.g. Shot 1 = '0-7s', Shot 2 = '7-14s', etc. "
        "Set time_range and duration_s so the sum matches EXACTLY the total above. "
        "For each shot, set narration_text to the concatenated text from the transcription segments that fall in that shot's time range. "
        "Set movement_prompt to empty string for every shot (movement_prompt is written later for the video model). "
        "Because the new Kling 3.0 pipeline is FIRST-FRAME-ONLY (we generate only the start frame, then animate with movement), "
        "last_frame_t2i_prompt must be the empty string \"\" for every shot. "
        "detailed_description must describe what happens in that shot window of the main script. "
        "In style_bible.camera_rules, specify that the camera follows the action (dynamic camera, not static). "
        "DYNAMIC CAMERA INTENT: For EVERY shot you MUST include a camera_path object. The camera should 'hunt' for the action described in the narration. "
        "camera_path structure: {\"type\": \"...\", \"start_focus\": \"...\", \"end_focus\": \"...\", \"zoom_level\": \"wide\"|\"medium\"|\"close\", \"target\": \"...\" (optional)}. "
        "type values: tilt_up, tilt_down, pan_left, pan_right, zoom_in, zoom_out, orbit (slight arc), static. "
        "Use tilt_up when narration describes upward motion (e.g. magma rising); tilt_down for downward (e.g. lava flowing). "
        "Use zoom_in when a specific ingredient/label is introduced (set target to ingredient name, e.g. \"SURROUNDING ROCK\"); zoom_out to reveal wider context. "
        "Use pan_left/pan_right when the focus shifts horizontally. start_focus and end_focus describe the spatial element the camera moves from/to (e.g. outer_core, crust, magma_chamber). "
        "zoom_level: wide (establishing), medium (default), close (detail). Never leave camera_path null; use {\"type\": \"static\", \"zoom_level\": \"medium\"} when no movement is needed."
    )
    user = f"""Topic: {topic}

Main script (video description aligned to timeline):
{main_script_15s[:12000]}

Timed transcription:
{seg_lines}

Divide the script into a sequence of shots (each 3–15 seconds) so that the sum of all duration_s equals EXACTLY {total_duration_int} seconds.

Produce JSON with: style_bible (visual_style, camera_rules, lighting_rules, continuity_rules, style_suffix), ingredients (list of {{ name, description, t2i_prompt: null, matplotlib_code: null }}), shots (list with shot_id 1..N, for each shot: time_range \"start-end s\" and duration_s integer 3..15, detailed_description, last_frame_t2i_prompt (must be \"\"), movement_prompt: \"\", camera_path (OBJECT with type, start_focus, end_focus, zoom_level, optional target—REQUIRED for every shot), ingredient_names, new_ingredient_names, reference_element_names (list of EXACTLY two ingredient names you will use as Kling elements reference images for this shot), narration_text from segments in that range, on_screen_text_overlay, assisting_visual_aids, i2i_spatial_prompt: \"\"), first_frame_t2i_prompt (one long T2I prompt for shot 1 first frame). Output ONLY valid JSON."""
    raw = _claude_generate(system, user, max_tokens=16384)
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    return json.loads(raw)


def _extract_labels_from_shots(shots: List[Dict[str, Any]], up_to_shot_id: Optional[int] = None) -> List[str]:
    """
    Extract text labels from project shots for the INVARIANTS section.
    Collects on_screen_text_overlay and label-like terms from assisting_visual_aids.
    If up_to_shot_id is set, only include shots with shot_id <= up_to_shot_id (labels visible so far).
    """
    labels: List[str] = []
    seen: set[str] = set()
    for s in (shots or []):
        sid = int(s.get("shot_id", 0) or 0)
        if up_to_shot_id is not None and sid > up_to_shot_id:
            continue
        overlay = (s.get("on_screen_text_overlay") or "").strip()
        if overlay and overlay.lower() != "none":
            # Strip surrounding quotes if present
            clean = overlay.strip('"\'')
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                labels.append(clean)
        aids = (s.get("assisting_visual_aids") or "").strip()
        if aids and aids.lower() not in ("none", ""):
            for m in re.finditer(r'"([^"]+)"', aids):
                lab = m.group(1).strip()
                if lab and lab.lower() not in seen:
                    seen.add(lab.lower())
                    labels.append(lab)
    return labels


def _camera_path_to_action_guidance(camera_path: Optional[Dict[str, Any]], shot: Dict[str, Any]) -> str:
    """
    Translate camera_path into explicit directional guidance for the I2V ACTION section.
    Template: '[ACTION] the camera [DIRECTION] while the [SUBJECT] [ANIMATION].'
    """
    if not camera_path or not isinstance(camera_path, dict):
        return ""
    ctype = (camera_path.get("type") or "").strip().lower()
    if not ctype or ctype == "static":
        return "ANIMATION: Keep camera stable. Subtle parallax: foreground elements move slightly faster than background when the scene has depth."
    start_f = (camera_path.get("start_focus") or "").strip()
    end_f = (camera_path.get("end_focus") or "").strip()
    target = (camera_path.get("target") or "").strip()  # e.g. ingredient name
    zoom = (camera_path.get("zoom_level") or "medium").strip().lower()
    subject = (shot.get("detailed_description") or "")[:200] or "the main subject"
    lines: List[str] = []
    if ctype == "tilt_up":
        lines.append(f"Slowly tilt the camera upward from {start_f or 'below'} toward {end_f or 'above'}, following the action described in the narration.")
    elif ctype == "tilt_down":
        lines.append(f"Slowly tilt the camera downward from {start_f or 'above'} toward {end_f or 'below'}, following the action described in the narration.")
    elif ctype == "pan_left":
        lines.append(f"Slowly pan the camera left, revealing more on the right. Focus moves from {start_f or 'right'} to {end_f or 'left'}.")
    elif ctype == "pan_right":
        lines.append(f"Slowly pan the camera right, revealing more on the left. Focus moves from {start_f or 'left'} to {end_f or 'right'}.")
    elif ctype == "zoom_in":
        if target:
            lines.append(f"Slowly zoom in toward the {target} while [SUBJECT] animates. The camera hunts for this element as it is introduced.")
        else:
            lines.append(f"Slowly zoom in toward {end_f or 'the focal point'}, following the narration. Foreground moves faster than background (parallax).")
    elif ctype == "zoom_out":
        lines.append(f"Slowly zoom out to reveal wider context. Foreground recedes faster than background (parallax). End focus: {end_f or 'wider scene'}.")
    elif ctype == "orbit":
        lines.append(f"Slowly orbit the camera in a slight arc around the subject, moving from {start_f or 'initial angle'} to {end_f or 'new angle'}.")
    else:
        lines.append(f"Move the camera ({ctype}) from {start_f or 'start'} to {end_f or 'end'}, following the narration.")
    lines.append("PHYSICS & PARALLAX: The camera has depth. When zooming or tilting, foreground elements move faster than background elements. Create 3D parallax to reinforce scale and spatial relationships.")
    lines.append("LABELS: All on-screen text and labels must move perfectly with the camera perspective—they are part of the scene, not floating.")
    return " ".join(lines)


CLAUDE_I2V_SYSTEM = """You are an expert at writing image-to-video (I2V) prompts for generative video models (e.g. Kling 3.0).

Your ONLY task: Write a single movement_prompt that will be sent to the I2V model. The prompt MUST follow this exact structure for one-shot accuracy:

---
ACTION: [Use the template: "[DIRECTION] the camera [MOVEMENT] while the [SUBJECT] [ANIMATION]." E.g. "Slowly tilt the camera upward and zoom in toward the upper crust, following the flow of the rising orange magma. The labels must move perfectly with the camera perspective." You will be given CAMERA_PATH_GUIDANCE—use it to build explicit directional vectors. NEVER use generic "move forward" prompts. Include camera motion (tilt, pan, zoom) synced with the narration and ingredients. The action must fit exactly within N seconds.]

INVARIANTS: [Explicitly list elements that MUST NOT CHANGE. You will be given LABELS_TO_PROTECT—quote each one in quotes and state they must remain perfectly legible, static, and unchanged in spelling. Also include: @Element1 and @Element2 (if provided) must remain consistent in identity, style, and position. Any diagrams, schematics, or key visual elements that must not morph or warp.]

STYLE CONSISTENCY: [Maintain the 2D clean vector illustration style. No new objects or labels should appear. Keep the same lighting, color palette, and framing unless the ACTION explicitly calls for a change.]

NEGATIVE MOTION: [Avoid text distortion, gibberish, morphing, blurring, extra labels, spelling changes. No hallucinating new objects. Structural integrity must be preserved.]
---

Requirements:
- The movement_prompt MUST include (1) NARRATION for this shot — what is being explained (the voiceover content); and (2) ASSISTING VISUAL AIDS for this shot (arrows, highlights, text boxes, text labels) when specified. If the shot has no assisting visual aids, do not invent any.
- The generated video MUST start exactly with the provided first frame (no variation).
- IMPORTANT: We do NOT provide/lock an explicit last-frame image. Ensure the motion naturally settles into a coherent end state.
- The movement_prompt SHOULD refer to @Element1 and @Element2 (if provided) in the INVARIANTS section.
- HARD LENGTH CAP: The final output MUST be <= 2400 characters. This is required for the Kling API.
- STRICT DURATION: Start with "A high-quality N-second clip. Exactly N seconds duration." Never use "about".
- COMPLETE ACTION: The action must begin and end within exactly N seconds. Avoid open-ended motions.
- If LABELS_TO_PROTECT is provided, you MUST list every label in the INVARIANTS section with: "The text labels [quoted list] must remain perfectly legible, static, and unchanged in spelling."
- DYNAMIC CAMERA: When CAMERA_PATH_GUIDANCE is provided, the ACTION MUST be driven by it. The camera must "hunt" for the action—synchronize with narration and ingredient introductions.
- PHYSICS: Include parallax when zooming/tilting—foreground moves faster than background to create depth.
- Output ONLY the movement_prompt text. No preamble, no "Here is the prompt", no markdown."""


def generate_i2v_prompt_claude(
    shot: Dict[str, Any],
    main_script_15s: str,
    first_frame_context: str,
    all_shots: Optional[List[Dict[str, Any]]] = None,
    style_bible: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call Claude to write the I2V movement_prompt for one shot. Only I2V prompts are written by Claude.
    first_frame_context: for shot 1 use project first_frame_t2i_prompt;
      for shot N>1 use a short textual context describing the scene at the start of this shot.
      (The actual first frame image is provided to the video model at runtime.)
    all_shots: When provided, labels from on_screen_text_overlay and assisting_visual_aids are
      extracted and injected into the INVARIANTS section of the movement prompt.
    style_bible: Optional visual style config for STYLE CONSISTENCY guidance.
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
    ingredient_names = shot.get("ingredient_names") or []

    # Extract labels from project_state for INVARIANTS injection (one-shot accuracy)
    labels_to_protect: List[str] = []
    if all_shots:
        labels_to_protect = _extract_labels_from_shots(all_shots, up_to_shot_id=shot_id)
    labels_block = ""
    if labels_to_protect:
        labels_block = (
            f"\n\nLABELS_TO_PROTECT (MUST list every one in INVARIANTS—they must remain legible, static, unchanged in spelling):\n"
            + ", ".join(f'"{l}"' for l in labels_to_protect)
        )
    style_block = ""
    if style_bible:
        vs = (style_bible.get("visual_style") or "").strip()
        if vs:
            style_block = f"\n\nSTYLE BIBLE (use for STYLE CONSISTENCY section): {vs[:500]}"
    camera_path = shot.get("camera_path")
    camera_path_block = _camera_path_to_action_guidance(camera_path, shot)
    if camera_path_block:
        camera_path_block = f"\n\nCAMERA_PATH_GUIDANCE (USE THIS to build the ACTION section—explicit direction, no generic moves):\n{camera_path_block}"
    narration_text = (shot.get("narration_text") or "").strip()
    assisting_visual_aids = (shot.get("assisting_visual_aids") or "").strip() or "none"
    on_screen_overlay = (shot.get("on_screen_text_overlay") or "").strip() or "none"
    duration_s = int(shot.get("duration_s", 0) or 0)
    reference_element_names = (shot.get("reference_element_names") or []) or []
    ref1 = reference_element_names[0] if len(reference_element_names) > 0 else "none"
    ref2 = reference_element_names[1] if len(reference_element_names) > 1 else "none"
    if duration_s > 0:
        exact_duration_text = (
            f"exactly {duration_s} seconds. Use technical phrasing: e.g. 'A high-quality {duration_s}-second clip', "
            f"'Short-form, exactly {duration_s} seconds duration.' Never use 'about'. "
            f"The action must be a complete, self-contained movement that fits in {duration_s} seconds (starts and ends within the window)."
        )
    else:
        exact_duration_text = "duration not specified in main script; state a reasonable exact duration in seconds in the prompt"

    user_content = f"""Write the I2V movement_prompt for SHOT {shot_id}.

MAIN SCRIPT (context; extract shot-relevant narration and visual aids from the shot-related part):
{main_script_15s[:2000]}

FIRST FRAME context (scene at start of this shot — T2I description or previous shot's end state):
{first_frame_context[:1500]}

THIS SHOT:
- detailed_description: {detailed}
- ingredient_names: {ingredient_names}

NARRATION for this shot (must be reflected in the I2V prompt so the video supports what is being said):
"{narration_text}"

ASSISTING VISUAL AIDS for this shot (include in the movement_prompt when not "none" — arrows, highlights, text boxes, text labels):
{assisting_visual_aids}

ON-SCREEN TEXT OVERLAY: {on_screen_overlay}

REFERENCE ELEMENTS (Kling 'elements' in the request; refer to them as @Element1 and @Element2):
- @Element1: {ref1}
- @Element2: {ref2}
{labels_block}
{style_block}
{camera_path_block}

REQUIRED DURATION (from main script): {exact_duration_text}.

Your movement_prompt MUST: (1) Use the exact structure: ACTION, INVARIANTS, STYLE CONSISTENCY, NEGATIVE MOTION. (2) Start with a clear technical duration line, e.g. "A high-quality {duration_s}-second clip. Exactly {duration_s} seconds duration." (3) In INVARIANTS, include LABELS_TO_PROTECT (if any) and @Element1/@Element2. (4) In STYLE CONSISTENCY, maintain 2D clean vector style; no new objects or labels. (5) Include the narration and assisting visual aids. Output only the movement_prompt text."""

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
    while keeping the same scene, narration, assisting visual aids, and the FIRST-FRAME-only constraints.

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
        exact_duration_text = (
            f"exactly {duration_s} seconds. Use 'A high-quality {duration_s}-second clip', 'Exactly {duration_s} seconds duration.' "
            f"Never use 'about'. Action must be complete within {duration_s} seconds."
        )
    else:
        exact_duration_text = "duration not specified; state a reasonable exact duration in seconds"

    user_content = f"""You will REVISE an existing image-to-video (I2V) movement_prompt for SHOT {shot_id}.

    The previous prompt caused the video API to finish without returning a video (likely due to content policy or prompt limits).
Your job is to rewrite the prompt so that:
- It is SHORTER and more concise (aim to stay at or under {max_chars} characters as a soft cap).
- It keeps the SAME scene, narration, and assisting visual aids intention.
- It still clearly enforces: video MUST start with the provided first frame.
  IMPORTANT: do NOT require an explicit locked match to a provided last frame (we extract the last frame after generation).
- It MUST state the exact required duration in seconds when given (from main script; no trimming).

MAIN SCRIPT (context; use only the parts relevant to this shot):
{main_script_15s[:2000]}

CURRENT I2V MOVEMENT PROMPT (to be shortened/refined):
\"\"\"\n{original_prompt.strip()}\n\"\"\"\n

SHOT METADATA (must still be respected):
- Narration: \"{narration_text}\"
- Assisting visual aids: {assisting_visual_aids}
- On-screen text overlay: {on_screen_overlay}
- Required duration: {exact_duration_text}

Rewrite the movement_prompt so it is more compact and less verbose, but still:
- Describes the motion starting from the FIRST frame (no explicit locked last frame requirement).
- Includes the narration content and assisting visual aids where relevant.
- States that the video must begin exactly with the provided first frame.
  Ensures the motion settles coherently within the exact duration.
{f'- States that the video must be exactly {duration_s} seconds long.' if duration_s > 0 else ''}

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

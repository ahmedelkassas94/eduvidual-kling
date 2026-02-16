import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

# Load .env for THIS process (critical for DRY_RUN, etc.)
load_dotenv()

from video_actions import (  # noqa: E402
    is_dry_run,
    generate_placeholder_video,
    is_video_valid,
    extract_last_frame,
    trim_video_to_duration,
)
from image_client import generate_image, generate_image_from_images  # noqa: E402
from frame_uploader import frame_to_public_url  # noqa: E402
from wan_client import submit_wan_i2v_job, wait_for_wan_result, download_file  # noqa: E402
from veo_client import (  # noqa: E402
    submit_veo_i2v_job,
    wait_for_veo_result,
    save_veo_video,
)
from tts_client import (  # noqa: E402
    generate_speech,
    adjust_audio_duration,
    process_audio_for_consistency,
    DEFAULT_VOICE,
    DEFAULT_MODEL,
    DEFAULT_SPEED,
)
from compositor import composite_images  # noqa: E402


# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _is_interactive() -> bool:
    """True if stdin is a TTY (user can press Enter / type). When False, skip interactive prompts."""
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _run_matplotlib_ingredient(code: str, output_path: Path) -> None:
    """Run planner-generated Matplotlib code in a subprocess and save figure to output_path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path.resolve())
    runner = f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
output_path = {repr(out_str)}
try:
    exec({repr(code)})
except Exception as e:
    raise RuntimeError(f"Matplotlib code failed: {{e}}")
if plt.get_fignums():
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
else:
    raise RuntimeError("No figure was created by the code.")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(runner)
        script = f.name
    try:
        r = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=output_path.parent,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout or f"Exit code {r.returncode}")
    finally:
        try:
            os.unlink(script)
        except Exception:
            pass
    if not output_path.exists():
        raise RuntimeError("Matplotlib did not produce an output file.")


def _require_approval(
    prompt_text: str,
    skip_message: str = "⏭️ Not approved. Exiting.",
    allow_auto: bool = True,
) -> None:
    """
    Ask user to type APPROVE; otherwise exit. Saves money by not generating without approval.
    When non-interactive: if allow_auto and APPROVE_ALL_INGREDIENTS=1, continue; else exit.
    """
    try:
        if _is_interactive():
            ans = input(prompt_text).strip().upper()
            if ans != "APPROVE":
                print(skip_message)
                sys.exit(0)
        else:
            if allow_auto and _env_bool("APPROVE_ALL_INGREDIENTS", False):
                print("(Non-interactive: APPROVE_ALL_INGREDIENTS=1, proceeding.)")
            else:
                print("(Non-interactive: no approval possible. Set APPROVE_ALL_INGREDIENTS=1 to auto-approve, or run interactively.)")
                print(skip_message)
                sys.exit(0)
    except EOFError:
        if allow_auto and _env_bool("APPROVE_ALL_INGREDIENTS", False):
            print("(No stdin: APPROVE_ALL_INGREDIENTS=1, proceeding.)")
        else:
            print(skip_message)
            sys.exit(0)


def _approve_shot(
    shot_id: int,
    first_frame_path: Path,
    reference_paths: Optional[List[Path]],
    movement_prompt: str,
) -> bool:
    """
    Show input images + prompt for one shot; return True if user approves, False if skip/exit.
    Saves money: no video generated until APPROVE.
    """
    ref_list = reference_paths or []
    try:
        if _is_interactive():
            print("\n" + "=" * 78)
            print(f"APPROVAL 2 & 3: SHOT {shot_id} — Input images and prompt (no video yet)")
            print("=" * 78)
            print("INPUT IMAGES FOR THIS VIDEO:")
            print(f"  • First frame (or last frame of previous): {first_frame_path.resolve()}")
            if ref_list:
                print("  • Reference images (elements):")
                for p in ref_list:
                    print(f"    - {p.resolve()}")
            else:
                print("  • Reference images: (none)")
            print("-" * 78)
            print("PROMPT FOR THIS VIDEO (I2V movement_prompt):")
            print("-" * 78)
            print(movement_prompt)
            print("=" * 78)
            ans = input(
                "Type APPROVE to generate this video segment, or SKIP to skip this segment and exit (no more videos): "
            ).strip().upper()
            if ans == "APPROVE":
                return True
            print("⏭️ Shot not approved. Exiting (no video generated for this segment).")
            return False
        else:
            if _env_bool("APPROVE_ALL_INGREDIENTS", False):
                return True
            print("(Non-interactive: set APPROVE_ALL_INGREDIENTS=1 to auto-approve each shot.)")
            return False
    except EOFError:
        if _env_bool("APPROVE_ALL_INGREDIENTS", False):
            return True
        return False


def _pretty_json(x: Any) -> str:
    return json.dumps(x, indent=2, ensure_ascii=False)


def build_image_prompt_from_frame(frame: dict, style_bible: Optional[dict] = None) -> str:
    """
    Convert a structured ImageFrame dict + style_bible into a single
    text-to-image prompt string.
    """
    frame_id = int(frame.get("frame_id"))
    image_prompt = (frame.get("image_prompt") or "").strip()
    env_details: List[str] = frame.get("environment_details") or []

    lines: List[str] = []

    # Professional academic style header
    lines.append("STYLE: Professional, sophisticated academic explainer video for university-level students. Cinematic quality, clean and modern visual design. Documentary/science channel aesthetic. No people, no faces, no hands.")
    lines.append("")
    lines.append("CRITICAL: ENVIRONMENT REALISM:")
    lines.append("- If this is a REAL-WORLD/PHYSICAL topic (airplanes, biology, chemistry, engineering, natural phenomena):")
    lines.append("  * Use PHOTOREALISTIC, REALISTIC environments - real objects in natural settings")
    lines.append("  * NO wind tunnels, NO digital/synthetic environments, NO abstract backgrounds")
    lines.append("  * Show real objects as they appear in real life (e.g., real commercial airplane flying in blue sky)")
    lines.append("- If this is an ABSTRACT/THEORETICAL topic (quantum mechanics, pure math, theoretical physics):")
    lines.append("  * Use appropriately stylized or digital environments that aid conceptual understanding")
    lines.append("")
    lines.append(f"FRAME INTENT: Academic explainer still #{frame_id} for university-level content (no narration, no UI chrome).")

    if style_bible:
        lines.append("")
        lines.append("STYLE BIBLE (global rules):")
        lines.append(f"- Visual style: {style_bible.get('visual_style', '').strip()}")
        lines.append(f"- Camera rules: {style_bible.get('camera_rules', '').strip()}")
        lines.append(f"- Lighting rules: {style_bible.get('lighting_rules', '').strip()}")
        lines.append(f"- Continuity rules: {style_bible.get('continuity_rules', '').strip()}")

    lines.append("")
    lines.append("IMAGE DESCRIPTION:")
    lines.append(image_prompt)

    if env_details:
        lines.append("")
        lines.append("ENVIRONMENT DETAILS:")
        for e in env_details:
            lines.append(f"- {e}")

    overlay = (frame.get("on_screen_text_overlay") or "").strip()
    if overlay and overlay.lower() != "none":
        lines.append("")
        lines.append(f"ON-SCREEN TEXT (do not cover main subject): {overlay}")

    return "\n".join(lines).strip() + "\n"


def build_animation_prompt_to_next_frame(
    current_frame: dict,
    next_frame: Optional[dict],
    style_bible: Optional[dict] = None,
) -> str:
    """
    Build a prompt for image-to-video that animates the current frame TO transition
    to the next frame. The animation should end looking like the next keyframe.
    
    Args:
        current_frame: Current frame dict (the frame to animate)
        next_frame: Next frame dict (the target state to transition to)
        style_bible: Visual style guidelines
    """
    current_image_prompt = (current_frame.get("image_prompt") or "").strip()
    transition_note = (current_frame.get("transition_to_next_frame") or "").strip()
    animation_type = (current_frame.get("animation_type") or "hold").strip().lower()
    
    lines: List[str] = []
    
    # Academic style header
    lines.append("STYLE: Professional, sophisticated academic explainer video for university-level students. Cinematic quality, clean and modern visual design.")
    lines.append("")
    
    lines.append("CURRENT FRAME (STARTING STATE):")
    lines.append(current_image_prompt)
    lines.append("")
    
    # Transition target: next frame description
    if next_frame:
        next_image_prompt = (next_frame.get("image_prompt") or "").strip()
        lines.append("TRANSITION TARGET (FINAL STATE - where this animation should end):")
        lines.append(next_image_prompt)
        lines.append("")
        lines.append("CRITICAL: The animation should TRANSITION from the current frame TO match the next frame's description.")
        lines.append("By the end of this segment, the scene should look like the transition target described above.")
        lines.append("")
    else:
        lines.append("NOTE: This is the final frame. Animate naturally to conclude the video.")
        lines.append("")
    
    # Transition instructions
    if transition_note:
        lines.append("TRANSITION INSTRUCTIONS:")
        lines.append(transition_note)
        lines.append("")
    
    # Camera movement from style_bible
    camera_rules = ""
    if style_bible:
        camera_rules = (style_bible.get("camera_rules") or "").strip()
        if camera_rules:
            lines.append("CAMERA MOVEMENT (ONE CONTINUOUS CINEMATIC SHOT):")
            lines.append(camera_rules)
            lines.append("")
    
    # Animation type guidance
    if animation_type == "hold":
        lines.append("ANIMATION: Keep camera position stable while objects/elements move naturally.")
    elif animation_type == "ken_burns_zoom_in":
        lines.append("ANIMATION: Slowly zoom in (push forward) while transitioning to the next frame state.")
    elif animation_type == "ken_burns_zoom_out":
        lines.append("ANIMATION: Slowly zoom out (pull back) while transitioning to the next frame state.")
    elif animation_type == "pan_left":
        lines.append("ANIMATION: Pan camera left while transitioning to the next frame state.")
    elif animation_type == "pan_right":
        lines.append("ANIMATION: Pan camera right while transitioning to the next frame state.")
    elif animation_type == "pan_up":
        lines.append("ANIMATION: Pan camera up while transitioning to the next frame state.")
    elif animation_type == "pan_down":
        lines.append("ANIMATION: Pan camera down while transitioning to the next frame state.")
    else:
        lines.append("ANIMATION: Move camera and animate elements to transition smoothly to the next frame state.")
    
    lines.append("")
    lines.append("OBJECT ANIMATION:")
    lines.append("- Animate individual objects/items within the image naturally")
    lines.append("- Objects should move/change to match the transition target state")
    lines.append("- Maintain object consistency - objects should remain correctly shaped and identifiable")
    lines.append("- Spatial relationships should evolve smoothly toward the next frame's layout")
    
    if next_frame:
        lines.append("")
        lines.append("FINAL STATE REQUIREMENT:")
        lines.append("By the end of this animation segment, the scene must match the transition target description above.")
        lines.append("The camera position, object positions, and scene state should align with the next keyframe.")
    
    lines.append("")
    lines.append("IMPORTANT: This is part of a continuous one-take cinematic shot. The transition should be smooth and seamless.")
    
    return "\n".join(lines).strip() + "\n"


def build_animation_prompt_from_frame(frame: dict, prev_frame: Optional[dict] = None, style_bible: Optional[dict] = None) -> str:
    """
    Build a prompt for image-to-video that describes what elements should animate
    and how they should move within the image.
    
    If prev_frame is provided, this segment continues from the previous segment,
    maintaining the same camera move for a "one-take" effect.
    """
    image_prompt = (frame.get("image_prompt") or "").strip()
    animation_type = (frame.get("animation_type") or "hold").strip().lower()
    continuity_note = (frame.get("continuity_note") or "").strip()
    
    lines: List[str] = []
    
    # Academic style header
    lines.append("STYLE: Professional, sophisticated academic explainer video for university-level students. Cinematic quality, clean and modern visual design.")
    lines.append("")
    
    # ONE-TAKE CONTINUITY HEADER (if this is not the first frame)
    if prev_frame:
        prev_animation_type = (prev_frame.get("animation_type") or "hold").strip().lower()
        prev_continuity = (prev_frame.get("continuity_note") or "").strip()
        
        lines.append("🎬 ONE-TAKE CONTINUITY: This video segment CONTINUES directly from the previous segment.")
        lines.append("The previous segment ended with the camera in a specific position and framing.")
        lines.append("")
        lines.append("CONTINUITY REQUIREMENTS:")
        lines.append("- Start this segment EXACTLY where the previous segment ended (same camera angle, same lens, same framing).")
        lines.append("- Continue the SAME camera movement pattern from the previous segment.")
        lines.append("- Maintain the same spatial relationships between elements as they were at the end of the previous segment.")
        lines.append("- Elements that were moving in the previous segment should continue their motion naturally.")
        lines.append("- NO cuts, NO jumps, NO camera angle changes. This is a continuous shot.")
        lines.append("")
        if prev_continuity:
            lines.append(f"Previous segment context: {prev_continuity}")
            lines.append("")
        lines.append("---")
        lines.append("")
    
    lines.append("Animate the elements in this image with smooth, natural motion.")
    lines.append("")
    lines.append("IMAGE CONTENT:")
    lines.append(image_prompt)
    lines.append("")
    
    # Get camera rules from style_bible if available
    camera_rules = ""
    if style_bible:
        camera_rules = (style_bible.get("camera_rules") or "").strip()
    
    # Describe animation based on animation_type
    if animation_type == "hold":
        if prev_frame:
            lines.append("ANIMATION: Continue the same camera position and framing from the previous segment. Keep subtle, natural micro-movements.")
        else:
            lines.append("ANIMATION: Keep the scene mostly static with subtle, natural micro-movements.")
    elif animation_type == "ken_burns_zoom_in":
        if prev_frame:
            lines.append("ANIMATION: Continue the slow zoom-in from the previous segment. The camera should be closer than where the previous segment ended, continuing the same smooth zoom-in motion.")
        else:
            lines.append("ANIMATION: Slowly zoom in on the main subject while keeping all elements in their relative positions.")
    elif animation_type == "ken_burns_zoom_out":
        if prev_frame:
            lines.append("ANIMATION: Continue the slow zoom-out from the previous segment. The camera should be wider than where the previous segment ended, continuing the same smooth zoom-out motion.")
        else:
            lines.append("ANIMATION: Slowly zoom out to reveal more of the scene while keeping all elements in their relative positions.")
    elif animation_type == "pan_left":
        if prev_frame:
            lines.append("ANIMATION: Continue the slow pan-left from the previous segment. The camera should have moved further left, revealing more of the scene on the right side, continuing the same smooth panning motion.")
        else:
            lines.append("ANIMATION: Pan the camera slowly to the left, revealing more of the scene on the right side.")
    elif animation_type == "pan_right":
        if prev_frame:
            lines.append("ANIMATION: Continue the slow pan-right from the previous segment. The camera should have moved further right, revealing more of the scene on the left side, continuing the same smooth panning motion.")
        else:
            lines.append("ANIMATION: Pan the camera slowly to the right, revealing more of the scene on the left side.")
    elif animation_type == "pan_up":
        if prev_frame:
            lines.append("ANIMATION: Continue the slow pan-up from the previous segment. The camera should have moved further up, revealing more of the scene below, continuing the same smooth panning motion.")
        else:
            lines.append("ANIMATION: Pan the camera slowly upward, revealing more of the scene below.")
    elif animation_type == "pan_down":
        if prev_frame:
            lines.append("ANIMATION: Continue the slow pan-down from the previous segment. The camera should have moved further down, revealing more of the scene above, continuing the same smooth panning motion.")
        else:
            lines.append("ANIMATION: Pan the camera slowly downward, revealing more of the scene above.")
    else:
        if prev_frame:
            lines.append("ANIMATION: Continue the same camera movement from the previous segment. Animate elements naturally based on their type (e.g., fluids flow, arrows move, values change).")
        else:
            lines.append("ANIMATION: Animate elements naturally based on their type (e.g., fluids flow, arrows move, values change).")
    
    if camera_rules:
        lines.append("")
        lines.append(f"GLOBAL CAMERA RULES (applies to all segments): {camera_rules}")
    
    if continuity_note:
        lines.append("")
        lines.append(f"CONTINUITY NOTE: {continuity_note}")
    
    lines.append("")
    lines.append("IMPORTANT: Animate the individual elements/items within the image (e.g., arrows sliding, fluids flowing, particles moving), not just camera movement. Each element should move according to its nature and purpose in the explanation.")
    
    if prev_frame:
        lines.append("")
        lines.append("CRITICAL: This segment must seamlessly connect to the previous segment. The transition should be invisible - as if this is all one continuous shot.")
    
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------
# LOAD / SAVE PROJECT STATE
# ---------------------------------------------------------
def load_state(project_dir: Path) -> dict:
    state_path = project_dir / "project_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"project_state.json not found in {project_dir}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(project_dir: Path, state: dict) -> None:
    state_path = project_dir / "project_state.json"
    state_path.write_text(_pretty_json(state), encoding="utf-8")


# ---------------------------------------------------------
# COST GATES
# ---------------------------------------------------------
def require_cost_acknowledgement_if_needed() -> None:
    if is_dry_run():
        print("🛑 DRY_RUN=true — no paid API calls will be made.")
        return

    if not _env_bool("REQUIRE_PROMPT_APPROVAL", True):
        print("[!] REQUIRE_PROMPT_APPROVAL=false -- proceeding without confirmation.")
        return

    try:
        ans = input(
            "[!] COST ALERT: This will call external AI APIs (text-to-image).\n"
            "Type YES to continue: "
        ).strip()
    except EOFError:
        print("No input (non-interactive). Run in a terminal to type YES and approve.")
        sys.exit(0)
    if ans != "YES":
        print("Cancelled (no API calls made).")
        sys.exit(0)


def prompt_approval(clip_id: int, prompt_preview: str) -> bool:
    require_approval = _env_bool("REQUIRE_PROMPT_APPROVAL", True)

    if not require_approval:
        return True

    print("\n" + "=" * 78)
    print(f"CLIP {clip_id} PROMPT PREVIEW (SHOT-BY-SHOT; THIS IS WHAT WILL BE SENT)")
    print("-" * 78)
    print(prompt_preview.strip())
    print("=" * 78)

    ans = input("Type APPROVE to generate this clip, or SKIP to skip: ").strip().upper()
    return ans == "APPROVE"


# ---------------------------------------------------------
# REPORTING
# ---------------------------------------------------------
def print_continuity_report(
    clip_id: int,
    meta: Dict[str, Any],
    first_frame_url: Optional[str],
) -> None:
    print("\n🧩 CONTINUITY REPORT")
    print(f"Clip: {clip_id}")
    print(f"Mode used: {meta.get('mode')}")
    print(f"Model used: {meta.get('model_used')}")
    print(f"Used first_frame_url: {meta.get('used_first_frame_url')}")
    print(f"first_frame_url (input): {first_frame_url}")
    print(f"Fallback used: {meta.get('fallback_used')}")


# ---------------------------------------------------------
# INGREDIENTS PIPELINE (15s shot-by-shot + named ingredients)
# ---------------------------------------------------------
def _sanitize_ingredient_filename(name: str) -> str:
    """Turn ingredient name into a safe filename (e.g. 'blue_arrows' -> 'blue_arrows')."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name.strip()).strip("_") or "ingredient"


def run_project_ingredients(project_dir: Path, state: dict) -> None:
    """
    15s shot-by-shot pipeline:
    1. Generate one T2I image per ingredient (lengthy prompt each).
    2. For each shot: first frame = composite of ingredients for shot 1, or last frame of prev (+ new ingredients if any).
    3. I2V with movement_prompt; extract last frame for next shot.
    """
    ingredients = state.get("ingredients", [])
    shots = state.get("shots", [])
    style_bible = state.get("style_bible") or {}

    clips_dir = project_dir / "clips"
    frames_dir = project_dir / "frames"
    ingredients_dir = project_dir / "ingredients"
    audio_dir = project_dir / "audio"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    ingredients_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    video_backend = (os.getenv("VIDEO_BACKEND") or "veo").strip().lower()
    print("\n" + "-" * 78)
    print("RUN CONFIG (INGREDIENTS PIPELINE — 15s SHOT-BY-SHOT)")
    print("-" * 78)
    print(f"Project: {project_dir}")
    print(f"VIDEO_BACKEND: {video_backend}")
    print(f"Shots: {len(shots)} | Ingredients: {len(ingredients)}")
    print("Approval: interactive (main script, T2I, and I2V prompts require APPROVE)")
    print("-" * 78)

    # Main script: show once and require explicit approval before any generation
    main_script = (state.get("main_script_15s") or "").strip()
    if main_script:
        print("\n" + "-" * 78)
        print("MAIN SCRIPT (15s reference) — REVIEW & APPROVE")
        print("-" * 78)
        print(main_script)
        _require_approval(
            prompt_text="Type APPROVE to confirm the main 15s script.",
            skip_message="Main script not approved. Exiting before generation.",
            allow_auto=False,
        )

    # require_cost_acknowledgement_if_needed()  # optional cost guard; kept disabled here

    # Step 1: Generate one T2I image per ingredient (lengthy prompt each, with approval)
    print("\n" + "=" * 78)
    print("STEP 1: GENERATE INGREDIENT IMAGES (T2I — ONE PER INGREDIENT)")
    print("=" * 78)

    ingredient_paths: Dict[str, Path] = {}
    for ing in ingredients:
        name = ing.get("name", "").strip()
        if not name:
            continue
        safe_name = _sanitize_ingredient_filename(name)
        image_path = ingredients_dir / f"{safe_name}.png"
        ingredient_paths[name] = image_path

        if not image_path.exists():
            matplotlib_code = (ing.get("matplotlib_code") or "").strip()
            prompt = (ing.get("t2i_prompt") or "").strip()

            if matplotlib_code:
                # Chart/plot ingredient: render via Matplotlib instead of T2I
                print(f"\n[MATPLOTLIB] Ingredient '{name}' has matplotlib_code — rendering plot...")
                try:
                    _run_matplotlib_ingredient(matplotlib_code, image_path)
                    print(f"[OK] {name} (Matplotlib)")
                except Exception as e:
                    print(f"[WARN] Matplotlib render failed for '{name}': {e}. Skipping.")
                continue

            if not prompt:
                continue
            # Prepend style context and append style_suffix (Stylist injection for consistency)
            style_prefix = ""
            if style_bible:
                vs = (style_bible.get("visual_style") or "").strip()
                if vs:
                    style_prefix = f"STYLE: {vs}\n\n"
            style_suffix = (style_bible.get("style_suffix") or "").strip() if style_bible else ""
            full_prompt = style_prefix + prompt + (" " + style_suffix if style_suffix else "")

            # Show full T2I prompt for this ingredient and require approval before calling Nano Banana
            print(f"\n" + "-" * 78)
            print(f"[IMG] INGREDIENT '{name}' T2I PROMPT — REVIEW & APPROVE")
            print("-" * 78)
            print(full_prompt)
            _require_approval(
                prompt_text=f"Type APPROVE to generate ingredient '{name}' ({image_path.name}).",
                skip_message="Ingredient not approved. Exiting before image generation.",
                allow_auto=False,
            )

            print(f"[IMG] Generating ingredient: {name} -> {image_path.name}")
            generate_image(full_prompt, image_path)
            print(f"[OK] {name}")
        else:
            print(f"[REUSE] Reusing: {name} -> {image_path.name}")

    print(f"\n[OK] All {len(ingredient_paths)} ingredient images ready.")

    # Step 2: For each shot — I2V (per-shot approval for inputs + prompt)
    print("\n" + "=" * 78)
    print("STEP 2: I2V SEGMENTS (last-frame chaining + reference images)")
    print("=" * 78)

    chain_frame_path = frames_dir / "chain_frame.png"
    current_first_frame_path: Optional[Path] = None
    current_first_frame_url: str = ""
    use_veo = video_backend == "veo"

    for idx, shot in enumerate(shots):
        shot_id = int(shot.get("shot_id", idx + 1))
        duration_s = int(shot.get("duration_s", 3))
        movement_prompt = (shot.get("movement_prompt") or "").strip()
        ingredient_names = shot.get("ingredient_names") or []
        new_ingredient_names = shot.get("new_ingredient_names") or []

        out_mp4 = clips_dir / f"clip_{shot_id:03d}.mp4"

        if out_mp4.exists() and is_video_valid(out_mp4):
            print(f"[REUSE] Reusing segment: {out_mp4.name}")
            extract_last_frame(out_mp4, chain_frame_path)
            current_first_frame_path = chain_frame_path
            current_first_frame_url = frame_to_public_url(current_first_frame_path) if not use_veo else ""
            # Per-shot narration skipped when POST_STITCH_AUDIO_ONLY=1 (audio from GPT+ElevenLabs after stitch)
            if not _env_bool("POST_STITCH_AUDIO_ONLY", False):
                narration_text = (shot.get("narration_text") or "").strip()
                if narration_text:
                    audio_path = audio_dir / f"narration_{shot_id:03d}.mp3"
                    try:
                        generate_speech(text=narration_text, output_path=audio_path, voice=DEFAULT_VOICE, model=DEFAULT_MODEL, speed=DEFAULT_SPEED)
                        process_audio_for_consistency(audio_path, duration_s)
                    except Exception as e:
                        print(f"[WARN] Narration failed: {e}")
            continue

        # Build first frame for this shot
        veo_reference_paths: Optional[List[Path]] = None  # Unused for now: Veo API does not support image + reference_images together
        if idx == 0:
            # Shot 1: Use I2I (Image-to-Image) with Nano Banana Pro to generate first frame from ingredient images
            ingredient_image_paths = [ingredient_paths[n] for n in ingredient_names if n in ingredient_paths and ingredient_paths[n].exists()]
            if not ingredient_image_paths:
                ingredient_image_paths = [ingredient_paths[n] for n in ingredient_names if n in ingredient_paths]
            if not ingredient_image_paths:
                raise RuntimeError(f"Shot 1: no ingredient images found for {ingredient_names}")
            
            # Get I2I spatial prompt from shot data
            i2i_spatial_prompt = (shot.get("i2i_spatial_prompt") or "").strip()
            if not i2i_spatial_prompt:
                raise RuntimeError(f"Shot 1: i2i_spatial_prompt is required but missing. The LLM must generate a detailed spatial relationship prompt for I2I generation.")
            
            first_frame_path = frames_dir / f"shot_{shot_id:03d}_first.png"
            
            # Show I2I prompt for approval before generating
            print("\n" + "-" * 78)
            print(f"SHOT 1 — I2I SPATIAL PROMPT (REVIEW & APPROVE)")
            print("-" * 78)
            print(f"Ingredient images to compose ({len(ingredient_image_paths)}):")
            for img_path in ingredient_image_paths:
                print(f"  - {img_path.name}")
            print(f"\nI2I Spatial Relationship Prompt:\n")
            print(i2i_spatial_prompt)
            _require_approval(
                prompt_text=f"Type APPROVE to generate shot 1's first frame via I2I ({len(ingredient_image_paths)} ingredient images).",
                skip_message="Shot 1 I2I not approved. Exiting before I2I call.",
                allow_auto=False,
            )
            
            # Generate first frame using I2I
            print(f"\n[I2I] Generating first frame from {len(ingredient_image_paths)} ingredient images...")
            generate_image_from_images(
                prompt=i2i_spatial_prompt,
                image_paths=ingredient_image_paths,
                out_path=first_frame_path,
            )
            print(f"[I2I] First frame generated: {first_frame_path.name}")
            current_first_frame_path = first_frame_path
        else:
            if new_ingredient_names:
                new_paths = [ingredient_paths[n] for n in new_ingredient_names if n in ingredient_paths and ingredient_paths[n].exists()]
                paths = [current_first_frame_path] if (current_first_frame_path and current_first_frame_path.exists()) else []
                paths.extend(new_paths)
                if paths:
                    first_frame_path = frames_dir / f"shot_{shot_id:03d}_first.png"
                    composite_images(paths, first_frame_path)
                    current_first_frame_path = first_frame_path
            else:
                current_first_frame_path = chain_frame_path
        if not current_first_frame_path or not current_first_frame_path.exists():
            raise RuntimeError(f"Shot {shot_id}: no first frame available")

        # Only upload for Wan (Veo uses local paths)
        if use_veo:
            current_first_frame_url = ""
        else:
            current_first_frame_url = frame_to_public_url(current_first_frame_path)

        # Show shot inputs (first frame + ingredients) and I2V movement prompt for approval
        print("\n" + "-" * 78)
        print(f"SHOT {shot_id} — INPUT & I2V PROMPT (REVIEW & APPROVE)")
        print("-" * 78)
        print(f"Duration: {duration_s}s")
        print(f"First frame image: {current_first_frame_path}")
        print(f"Ingredients in shot: {ingredient_names}")
        if new_ingredient_names:
            print(f"New ingredients introduced here: {new_ingredient_names}")
        print("\nI2V movement prompt:\n")
        print(movement_prompt or "[EMPTY]")
        _require_approval(
            prompt_text=f"Type APPROVE to generate shot {shot_id} ({duration_s}s).",
            skip_message="Shot not approved. Exiting before I2V call.",
            allow_auto=False,
        )

        # DRY RUN: placeholder clip
        if is_dry_run():
            from video_actions import animate_still_to_video
            print(f"[I2V] (DRY_RUN) Shot {shot_id} -> {out_mp4.name}")
            animate_still_to_video(current_first_frame_path, out_mp4, duration_s, "hold")
            extract_last_frame(out_mp4, chain_frame_path)
            current_first_frame_path = chain_frame_path
            if not _env_bool("POST_STITCH_AUDIO_ONLY", False):
                narration_text = (shot.get("narration_text") or "").strip()
                if narration_text:
                    audio_path = audio_dir / f"narration_{shot_id:03d}.mp3"
                    try:
                        generate_speech(text=narration_text, output_path=audio_path, voice=DEFAULT_VOICE, model=DEFAULT_MODEL, speed=DEFAULT_SPEED)
                        process_audio_for_consistency(audio_path, duration_s)
                    except Exception as e:
                        print(f"[WARN] (DRY_RUN) Narration failed: {e}")
            continue

        # Real I2V
        print(f"\n[I2V] Shot {shot_id} -> {out_mp4.name} (duration={duration_s}s)")
        try:
            if use_veo:
                client, operation = submit_veo_i2v_job(
                    prompt=movement_prompt,
                    image_path=current_first_frame_path,
                    duration_s=duration_s,
                    reference_image_paths=veo_reference_paths if veo_reference_paths else None,
                    aspect_ratio="16:9",
                    resolution=(os.getenv("VEO_RESOLUTION") or "720p").strip(),
                    negative_prompt="logos, watermarks, people, human, face, hands",
                )
                client, generated_video = wait_for_veo_result(client, operation)
                save_veo_video(client, generated_video, out_mp4)
            else:
                task_id = submit_wan_i2v_job(
                    prompt=movement_prompt,
                    first_frame_url=current_first_frame_url,
                    duration_s=duration_s,
                    resolution=os.getenv("WAN_I2V_RESOLUTION") or "720P",
                    aspect_ratio="16:9",
                    model=os.getenv("WAN_I2V_MODEL") or "wan2.6-i2v",
                    negative_prompt="logos, watermarks, people, human, face, hands",
                    shot_type=os.getenv("WAN_I2V_SHOT_TYPE") or "single",
                )
                video_url = wait_for_wan_result(task_id)
                download_file(video_url, out_mp4)

            need_trim = (use_veo and duration_s not in (4, 6, 8)) or (not use_veo and duration_s not in (5, 10))
            if need_trim:
                trim_video_to_duration(out_mp4, duration_s)

            extract_last_frame(out_mp4, chain_frame_path)
            current_first_frame_path = chain_frame_path
            current_first_frame_url = frame_to_public_url(current_first_frame_path) if not use_veo else ""
            print(f"[OK] Shot {shot_id} OK: {out_mp4.name}")

            if not _env_bool("POST_STITCH_AUDIO_ONLY", False):
                narration_text = (shot.get("narration_text") or "").strip()
                if narration_text:
                    audio_path = audio_dir / f"narration_{shot_id:03d}.mp3"
                    try:
                        generate_speech(text=narration_text, output_path=audio_path, voice=DEFAULT_VOICE, model=DEFAULT_MODEL, speed=DEFAULT_SPEED)
                        process_audio_for_consistency(audio_path, duration_s)
                    except Exception as e:
                        print(f"[WARN] Narration failed: {e}")
        except Exception as e:
            print(f"[ERR] Error generating shot {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            if idx == 0:
                raise RuntimeError(f"First shot failed: {e}") from e

    print("\n[OK] Ingredients pipeline complete.")
    print(f"Clips: {clips_dir}")
    print("Next: run stitch_video.py to assemble final_video.mp4")


# ---------------------------------------------------------
# MAIN ORCHESTRATION (IMAGE-BASED — LEGACY 20s FRAMES)
# ---------------------------------------------------------
def run_project(project_dir: Path) -> None:
    state = load_state(project_dir)

    # New pipeline: 15s shot-by-shot + named ingredients
    ingredients = state.get("ingredients", [])
    shots = state.get("shots", [])
    if ingredients and shots:
        run_project_ingredients(project_dir, state)
        return

    # Legacy pipeline: frames-based (20s)
    frames = state.get("frames", [])
    if not frames:
        raise RuntimeError("No frames and no shots/ingredients found in project_state.json")

    style_bible = state.get("style_bible") or {}

    clips_dir = project_dir / "clips"
    frames_dir = project_dir / "frames"
    audio_dir = project_dir / "audio"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    video_backend = (os.getenv("VIDEO_BACKEND") or "veo").strip().lower()
    print("\n" + "-" * 78)
    print("RUN CONFIG")
    print("-" * 78)
    print(f"Project: {project_dir}")
    print(f"VIDEO_BACKEND: {video_backend} (veo = Veo 3 Fast with audio, wan = DashScope Wan I2V)")
    print(f"DRY_RUN: {is_dry_run()}")
    print(f"REQUIRE_PROMPT_APPROVAL: {_env_bool('REQUIRE_PROMPT_APPROVAL', True)}")
    print("-" * 78)

    require_cost_acknowledgement_if_needed()

    # NEW ARCHITECTURE: Script-first, long-take cinematography with last-frame chaining
    # Step 1: Generate ONLY the first frame image (T2I) — first frame of first video
    # Step 2: For each segment: input = T2I (seg 1) or last frame of prev (seg 2+); animate; extract last frame
    # Step 3: Stitch for one continuous long-take (no visible cuts)
    
    print("\n" + "=" * 78)
    print("STEP 1: GENERATE FIRST FRAME IMAGE (T2I) — ONLY FRAME 1")
    print("=" * 78)
    
    first_frame = frames[0]
    first_frame_id = int(first_frame.get("frame_id"))
    first_frame_image_path = frames_dir / f"frame_{first_frame_id:03d}.png"
    
    if not first_frame_image_path.exists():
        print(f"\n🖼️ Generating first frame image (T2I): {first_frame_image_path.name}")
        prompt_t2i = build_image_prompt_from_frame(first_frame, style_bible=style_bible)
        
        if _env_bool("REQUIRE_PROMPT_APPROVAL", True):
            print("\n" + "=" * 78)
            print("FIRST FRAME (T2I) IMAGE PROMPT")
            print("=" * 78)
            print(prompt_t2i.strip())
            print("=" * 78)
            ans = input("Type APPROVE to generate this image, or SKIP to cancel: ").strip().upper()
            if ans != "APPROVE":
                print("⏭️ Cancelled.")
                return
        
        generate_image(prompt_t2i, first_frame_image_path)
        print(f"✅ First frame image generated")
    else:
        print(f"♻️ Reusing existing first frame image: {first_frame_image_path.name}")
        # Show T2I prompt used for this image (for review)
        prompt_t2i = build_image_prompt_from_frame(first_frame, style_bible=style_bible)
        print("\n" + "-" * 78)
        print("FIRST FRAME (T2I) PROMPT (used for existing image)")
        print("-" * 78)
        print(prompt_t2i.strip())
        print("-" * 78)
    
    current_first_frame_path: Path = first_frame_image_path
    current_first_frame_url: str = frame_to_public_url(current_first_frame_path)
    print(f"\n✅ First frame ready. Segment 2+ will use last frame of previous segment.")
    
    # Pause so user can review the first image before videos
    print("\n" + "-" * 78)
    print("REVIEW FIRST IMAGE")
    print("-" * 78)
    print(f"First image path: {current_first_frame_path.resolve()}")
    print("Please open and review the image above, then continue.")
    try:
        if _is_interactive():
            input("Press Enter to continue to prompt preview (or Ctrl+C to exit)...")
        else:
            print("(Non-interactive: auto-continuing to prompt preview.)")
    except EOFError:
        print("(No stdin: auto-continuing to prompt preview.)")
    
    # Preview all segment animation prompts before generating any videos
    print("\n" + "=" * 78)
    print("PREVIEW: ALL SEGMENT ANIMATION PROMPTS (I2V)")
    print("=" * 78)
    prev_preview: Optional[dict] = None
    for idx, f in enumerate(frames):
        frame_id = int(f.get("frame_id"))
        prompt_preview = build_animation_prompt_from_frame(
            f, prev_frame=prev_preview, style_bible=style_bible
        )
        print("\n" + "---" * 26)
        print(f"SEGMENT {frame_id} (frame {idx + 1}/{len(frames)}) → clip_{frame_id:03d}.mp4")
        print("---" * 26)
        print(prompt_preview.strip())
        prev_preview = f
    print("\n" + "=" * 78)
    try:
        if _is_interactive():
            ans = input("Type APPROVE to generate all video segments, or SKIP to exit: ").strip().upper()
            if ans != "APPROVE":
                print("⏭️ Video generation skipped. You can run again later to generate videos.")
                return
        else:
            print("(Non-interactive: auto-approving video generation.)")
    except EOFError:
        print("(No stdin: auto-approving video generation.)")
    
    print("\n" + "=" * 78)
    print("STEP 2: I2V SEGMENTS (LAST-FRAME CHAINING — LONG-TAKE CINEMATOGRAPHY)")
    print("=" * 78)
    
    chain_frame_path = frames_dir / "chain_frame.png"
    prev_frame: Optional[dict] = None

    for idx, f in enumerate(frames):
        frame_id = int(f.get("frame_id"))
        duration_s = int(f.get("duration_s", 3))
        animation_type = f.get("animation_type", "hold")

        out_mp4 = clips_dir / f"clip_{frame_id:03d}.mp4"

        # Reuse existing clip if already valid
        if out_mp4.exists() and is_video_valid(out_mp4):
            print(f"♻️ Reusing existing segment: {out_mp4.name}")
            extract_last_frame(out_mp4, chain_frame_path)
            current_first_frame_path = chain_frame_path
            current_first_frame_url = frame_to_public_url(current_first_frame_path)
            prev_frame = f
            continue

        animation_prompt = build_animation_prompt_from_frame(
            f, prev_frame=prev_frame, style_bible=style_bible
        )

        # DRY RUN: chain last frame for next segment
        if is_dry_run():
            image_path = current_first_frame_path
            print(
                f"🎬 (DRY_RUN) Segment {frame_id} -> {out_mp4.name} "
                f"(input={'first frame (T2I)' if idx == 0 else 'last frame of previous'}, duration={duration_s}s, type={animation_type})"
            )
            from video_actions import animate_still_to_video  # local import to avoid circular hint

            animate_still_to_video(image_path, out_mp4, duration_s, animation_type)
            extract_last_frame(out_mp4, chain_frame_path)
            current_first_frame_path = chain_frame_path
            current_first_frame_url = frame_to_public_url(current_first_frame_path)
            
            narration_text = (f.get("narration_text") or "").strip()
            if narration_text:
                audio_path = audio_dir / f"narration_{frame_id:03d}.mp3"
                print(f"🔊 (DRY_RUN) Generating narration audio for frame {frame_id} (consistent voice/speed)...")
                try:
                    generate_speech(
                        text=narration_text,
                        output_path=audio_path,
                        voice=DEFAULT_VOICE,
                        model=DEFAULT_MODEL,
                        speed=DEFAULT_SPEED,
                    )
                    process_audio_for_consistency(audio_path, duration_s)
                    print(f"✅ (DRY_RUN) Narration audio generated and normalized: {audio_path.name}")
                except Exception as e:
                    print(f"⚠️ (DRY_RUN) Failed to generate narration audio: {e}")
            
            prev_frame = f
            continue

        print(
            f"\n🎬 Segment {frame_id} -> {out_mp4.name} "
            f"(input={'first frame (T2I)' if idx == 0 else 'last frame of previous segment'}, duration={duration_s}s)"
        )
        use_veo = (os.getenv("VIDEO_BACKEND") or "veo").strip().lower() == "veo"

        try:
            if use_veo:
                veo_resolution = (os.getenv("VEO_RESOLUTION") or "720p").strip()
                client, operation = submit_veo_i2v_job(
                    prompt=animation_prompt,
                    image_path=current_first_frame_path,
                    duration_s=duration_s,
                    aspect_ratio="16:9",
                    resolution=veo_resolution,
                    negative_prompt=(
                        "logos, watermarks, brand marks, extra limbs, distorted faces, "
                        "flicker, glitch, people, human, face, hands"
                    ),
                )
                print(f"⏳ Waiting for Veo I2V generation...")
                client, generated_video = wait_for_veo_result(client, operation)
                save_veo_video(client, generated_video, out_mp4)
            else:
                model = os.getenv("WAN_I2V_MODEL") or "wan2.6-i2v"
                i2v_resolution = os.getenv("WAN_I2V_RESOLUTION") or "720P"
                i2v_shot_type = os.getenv("WAN_I2V_SHOT_TYPE") or "single"
                task_id = submit_wan_i2v_job(
                    prompt=animation_prompt,
                    first_frame_url=current_first_frame_url,
                    duration_s=duration_s,
                    resolution=i2v_resolution,
                    aspect_ratio="16:9",
                    model=model,
                    negative_prompt=(
                        "logos, watermarks, brand marks, extra limbs, distorted faces, "
                        "flicker, glitch, people, human, face, hands"
                    ),
                    shot_type=i2v_shot_type,
                )
                print(f"⏳ Waiting for I2V generation (task_id: {task_id})...")
                video_url = wait_for_wan_result(task_id)
                download_file(video_url, out_mp4)

            need_trim = (use_veo and duration_s not in (4, 6, 8)) or (not use_veo and duration_s not in (5, 10))
            if need_trim:
                print(f"✂️ Trimming video from API duration to exact {duration_s}s...")
                trim_video_to_duration(out_mp4, duration_s)
            
            # Chain: extract last frame for next segment
            extract_last_frame(out_mp4, chain_frame_path)
            current_first_frame_path = chain_frame_path
            current_first_frame_url = frame_to_public_url(current_first_frame_path)
            
            print(f"✅ Segment {frame_id} OK: {out_mp4.name}")
            
            narration_text = (f.get("narration_text") or "").strip()
            if narration_text:
                audio_path = audio_dir / f"narration_{frame_id:03d}.mp3"
                print(f"🔊 Generating narration audio for frame {frame_id} (consistent voice/speed)...")
                try:
                    generate_speech(
                        text=narration_text,
                        output_path=audio_path,
                        voice=DEFAULT_VOICE,
                        model=DEFAULT_MODEL,
                        speed=DEFAULT_SPEED,
                    )
                    process_audio_for_consistency(audio_path, duration_s)
                    print(f"✅ Narration audio generated and normalized: {audio_path.name}")
                except Exception as e:
                    print(f"⚠️ Failed to generate narration audio: {e}")
                    print(f"   Continuing without audio for this segment...")
            else:
                print(f"⚠️ No narration text provided for frame {frame_id}, skipping audio generation.")
        except Exception as e:
            print(f"❌ Error generating segment {frame_id}: {e}")
            print(f"⚠️ Error details: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"⚠️ Traceback:")
            traceback.print_exc()
            print(f"⚠️ Continuing with next segment...")
            if frame_id == first_frame_id:
                print(f"❌ CRITICAL: First segment failed. Cannot continue without initial video.")
                raise RuntimeError(f"First segment generation failed: {e}") from e
            print(f"⚠️ Next segment will use last frame of last successful clip.")
            prev_frame = f
            continue

        prev_frame = f

    print("\n✅ Orchestration complete.")
    print(f"Clips folder: {clips_dir}")
    print(f"Frames folder: {frames_dir}")
    print("Next: run stitch_video.py to assemble final_video.mp4")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py projects/<PROJECT_ID>")
        sys.exit(1)

    project_dir = Path(sys.argv[1]).resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project dir not found: {project_dir}")

    run_project(project_dir)


if __name__ == "__main__":
    main()

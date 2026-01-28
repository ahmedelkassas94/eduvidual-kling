import json
import os
import sys
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
from image_client import generate_image  # noqa: E402
from frame_uploader import frame_to_public_url  # noqa: E402
from wan_client import submit_wan_i2v_job, wait_for_wan_result, download_file  # noqa: E402
from tts_client import generate_speech, adjust_audio_duration  # noqa: E402


# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


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
        print("⚠️ REQUIRE_PROMPT_APPROVAL=false — proceeding without confirmation.")
        return

    ans = input(
        "⚠️ COST ALERT: This will call external AI APIs (text-to-image).\n"
        "Type YES to continue: "
    ).strip()
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
# MAIN ORCHESTRATION (IMAGE-BASED)
# ---------------------------------------------------------
def run_project(project_dir: Path) -> None:
    state = load_state(project_dir)

    frames = state.get("frames", [])
    if not frames:
        raise RuntimeError("No frames found in project_state.json")

    style_bible = state.get("style_bible") or {}

    clips_dir = project_dir / "clips"
    frames_dir = project_dir / "frames"
    audio_dir = project_dir / "audio"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 78)
    print("RUN CONFIG")
    print("-" * 78)
    print(f"Project: {project_dir}")
    print(f"DRY_RUN: {is_dry_run()}")
    print(f"REQUIRE_PROMPT_APPROVAL: {_env_bool('REQUIRE_PROMPT_APPROVAL', True)}")
    print("-" * 78)

    require_cost_acknowledgement_if_needed()

    # Main loop: one animated segment per frame (one-take continuity).
    # New architecture:
    # - ONE master still is created via Nano Banana (frame 1).
    # - All WAN I2V segments are chained by feeding the LAST FRAME of each
    #   segment as the first frame for the next segment.
    prev_frame: Optional[dict] = None
    current_first_frame_url: Optional[str] = None

    for idx, f in enumerate(frames):
        frame_id = int(f.get("frame_id"))
        duration_s = int(f.get("duration_s", 3))
        animation_type = f.get("animation_type", "hold")

        out_mp4 = clips_dir / f"clip_{frame_id:03d}.mp4"

        # Reuse existing clip if already valid
        if out_mp4.exists() and is_video_valid(out_mp4):
            print(f"♻️ Reusing existing segment: {out_mp4.name}")
            # Still need to extract last frame for chaining if this isn't the last frame
            if idx < len(frames) - 1:
                final_frame_path = frames_dir / f"frame_{frame_id:03d}_last.png"
                if not final_frame_path.exists():
                    print(f"📸 Extracting last frame for chaining: {final_frame_path.name}")
                    extract_last_frame(out_mp4, final_frame_path)
                    current_first_frame_url = frame_to_public_url(final_frame_path)
                else:
                    current_first_frame_url = frame_to_public_url(final_frame_path)
            prev_frame = f
            continue

        # DRY RUN: use ffmpeg-based still animation (no WAN cost)
        if is_dry_run():
            image_path = frames_dir / f"frame_{frame_id:03d}.png"
            print(f"🖼️ (DRY_RUN) Generating still image for frame {frame_id}: {image_path.name}")
            prompt = build_image_prompt_from_frame(f, style_bible=style_bible)
            generate_image(prompt, image_path)

            print(
                f"🎬 (DRY_RUN) Animating frame {frame_id} -> {out_mp4.name} "
                f"(duration={duration_s}s, type={animation_type})"
            )
            from video_actions import animate_still_to_video  # local import to avoid circular hint

            animate_still_to_video(image_path, out_mp4, duration_s, animation_type)
            
            # Generate narration audio for DRY_RUN too (with consistent settings)
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

        # For the first real segment, create the master still via Nano Banana
        if current_first_frame_url is None:
            image_path = frames_dir / "frame_001.png"
            print(f"🖼️ Generating initial still image (master scene) for frame 1: {image_path.name}")
            prompt_t2i = build_image_prompt_from_frame(frames[0], style_bible=style_bible)

            if _env_bool("REQUIRE_PROMPT_APPROVAL", True):
                print("\n" + "=" * 78)
                print("INITIAL FRAME 1 T2I PROMPT PREVIEW (MASTER SCENE)")
                print("-" * 78)
                print(prompt_t2i.strip())
                print("=" * 78)
                ans = input("Type APPROVE to generate the initial image, or SKIP to cancel: ").strip().upper()
                if ans != "APPROVE":
                    print("⏭️ Skipping project run (initial image not approved).")
                    return

            generate_image(prompt_t2i, image_path)
            print("☁️ Uploading initial still to get public URL for I2V chaining...")
            current_first_frame_url = frame_to_public_url(image_path)

        # Build animation prompt with continuity from previous frame (one-take)
        if prev_frame:
            print(
                f"🎬 Building animation prompt with continuity from previous frame "
                f"{int(prev_frame.get('frame_id'))}..."
            )
        animation_prompt = build_animation_prompt_from_frame(
            f, prev_frame=prev_frame, style_bible=style_bible
        )

        # Use WAN I2V with chaining: each segment starts from current_first_frame_url
        print(
            f"🎬 Animating elements in frame {frame_id} -> {out_mp4.name} "
            f"(duration={duration_s}s, using I2V, chained from previous segment)"
        )

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
        try:
            video_url = wait_for_wan_result(task_id)
            download_file(video_url, out_mp4)
            
            # Trim video to exact duration if needed (API only supports 5s or 10s)
            # Check actual duration and trim if necessary
            if duration_s not in [5, 10]:
                print(f"✂️ Trimming video from API duration to exact {duration_s}s...")
                trim_video_to_duration(out_mp4, duration_s)
            
            print(f"✅ Segment {frame_id} OK: {out_mp4.name}")
            
            # Generate narration audio for this segment with consistent settings
            narration_text = (f.get("narration_text") or "").strip()
            if narration_text:
                audio_path = audio_dir / f"narration_{frame_id:03d}.mp3"
                print(f"🔊 Generating narration audio for frame {frame_id} (consistent voice/speed)...")
                try:
                    generate_speech(
                        text=narration_text,
                        output_path=audio_path,
                        voice=DEFAULT_VOICE,  # Consistent voice across all segments
                        model=DEFAULT_MODEL,  # Consistent model
                        speed=DEFAULT_SPEED,  # Consistent speed
                    )
                    # Process audio for consistency: normalize levels and adjust duration
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
            # CRITICAL: If we don't have a valid video, we can't extract a last frame
            # So we need to use the previous frame's last frame URL for chaining
            # But if this is frame 1, we're stuck - need to abort or retry
            if frame_id == 1:
                print(f"❌ CRITICAL: Frame 1 failed. Cannot continue without initial frame.")
                print(f"   Please check the error above and retry.")
                raise RuntimeError(f"Frame 1 generation failed: {e}") from e
            
            # For subsequent frames, try to continue using the previous frame's last frame
            # This maintains some continuity even if one segment fails
            if current_first_frame_url:
                print(f"⚠️ Will use previous frame's last frame ({current_first_frame_url[:50]}...) for next segment")
            else:
                print(f"❌ WARNING: No previous frame URL available. Next segment may fail.")
            
            # Still update prev_frame for continuity in prompts (even though video failed)
            prev_frame = f
            continue

        # Extract last frame and upload as next first_frame_url
        final_frame_path = frames_dir / f"frame_{frame_id:03d}_last.png"
        print(f"📸 Extracting last frame for chaining: {final_frame_path.name}")
        extract_last_frame(out_mp4, final_frame_path)
        current_first_frame_url = frame_to_public_url(final_frame_path)

        # Update prev_frame for continuity in prompts
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

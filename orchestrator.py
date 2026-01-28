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
)
from image_client import generate_image  # noqa: E402
from frame_uploader import frame_to_public_url  # noqa: E402
from wan_client import submit_wan_i2v_job, wait_for_wan_result, download_file  # noqa: E402


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

    # Safety header: keep it short and non-triggering
    lines.append("SAFETY: family-friendly educational illustration; no people; no violence; no nudity; no weapons; no blood.")
    lines.append("")
    lines.append(f"FRAME INTENT: Educational still #{frame_id} (no narration, no UI chrome).")

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


def build_animation_prompt_from_frame(frame: dict) -> str:
    """
    Build a prompt for image-to-video that describes what elements should animate
    and how they should move within the image.
    """
    image_prompt = (frame.get("image_prompt") or "").strip()
    animation_type = (frame.get("animation_type") or "hold").strip().lower()
    
    lines: List[str] = []
    lines.append("Animate the elements in this image with smooth, natural motion.")
    lines.append("")
    lines.append("IMAGE CONTENT:")
    lines.append(image_prompt)
    lines.append("")
    
    # Describe animation based on animation_type
    if animation_type == "hold":
        lines.append("ANIMATION: Keep the scene mostly static with subtle, natural micro-movements.")
    elif animation_type == "ken_burns_zoom_in":
        lines.append("ANIMATION: Slowly zoom in on the main subject while keeping all elements in their relative positions.")
    elif animation_type == "ken_burns_zoom_out":
        lines.append("ANIMATION: Slowly zoom out to reveal more of the scene while keeping all elements in their relative positions.")
    elif animation_type == "pan_left":
        lines.append("ANIMATION: Pan the camera slowly to the left, revealing more of the scene on the right side.")
    elif animation_type == "pan_right":
        lines.append("ANIMATION: Pan the camera slowly to the right, revealing more of the scene on the left side.")
    elif animation_type == "pan_up":
        lines.append("ANIMATION: Pan the camera slowly upward, revealing more of the scene below.")
    elif animation_type == "pan_down":
        lines.append("ANIMATION: Pan the camera slowly downward, revealing more of the scene above.")
    else:
        lines.append("ANIMATION: Animate elements naturally based on their type (e.g., fluids flow, arrows move, values change).")
    
    lines.append("")
    lines.append("IMPORTANT: Animate the individual elements/items within the image (e.g., arrows sliding, fluids flowing, particles moving), not just camera movement. Each element should move according to its nature and purpose in the explanation.")
    
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
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 78)
    print("RUN CONFIG")
    print("-" * 78)
    print(f"Project: {project_dir}")
    print(f"DRY_RUN: {is_dry_run()}")
    print(f"REQUIRE_PROMPT_APPROVAL: {_env_bool('REQUIRE_PROMPT_APPROVAL', True)}")
    print("-" * 78)

    require_cost_acknowledgement_if_needed()

    # Main loop: one animated segment per frame
    for f in frames:
        frame_id = int(f.get("frame_id"))
        duration_s = int(f.get("duration_s", 3))
        animation_type = f.get("animation_type", "hold")

        out_mp4 = clips_dir / f"clip_{frame_id:03d}.mp4"

        # Reuse existing clip if already valid
        if out_mp4.exists() and is_video_valid(out_mp4):
            print(f"♻️ Reusing existing segment: {out_mp4.name}")
            continue

        if is_dry_run():
            # No external API calls: just generate a black placeholder segment.
            print(
                f"🛑 DRY RUN ENABLED — generating placeholder MP4 for frame {frame_id} "
                "(no paid API call)"
            )
            generate_placeholder_video(
                clip_id=frame_id,
                duration_s=duration_s,
                prompt="",
                negative_prompt="",
                output_dir=clips_dir,
            )
            continue

        # Build text-to-image prompt and generate the still
        prompt = build_image_prompt_from_frame(f, style_bible=style_bible)

        if _env_bool("REQUIRE_PROMPT_APPROVAL", True):
            print("\n" + "=" * 78)
            print(f"FRAME {frame_id} T2I PROMPT PREVIEW")
            print("-" * 78)
            print(prompt.strip())
            print("=" * 78)
            ans = input("Type APPROVE to generate this frame, or SKIP to skip: ").strip().upper()
            if ans != "APPROVE":
                print(f"⏭️ Skipping frame {frame_id} (no generation).")
                continue

        image_path = frames_dir / f"frame_{frame_id:03d}.png"
        print(f"🖼️ Generating still image for frame {frame_id}: {image_path.name}")
        generate_image(prompt, image_path)

        # Upload image to get a public URL for I2V
        print(f"☁️ Uploading image to get public URL for I2V...")
        image_url = frame_to_public_url(image_path)

        # Build animation prompt describing what should move
        animation_prompt = build_animation_prompt_from_frame(f)

        # Use image-to-video to animate elements within the image
        print(
            f"🎬 Animating elements in frame {frame_id} -> {out_mp4.name} "
            f"(duration={duration_s}s, using I2V)"
        )
        
        model = os.getenv("WAN_I2V_MODEL") or "wan2.6-i2v"
        i2v_resolution = os.getenv("WAN_I2V_RESOLUTION") or "720P"
        i2v_shot_type = os.getenv("WAN_I2V_SHOT_TYPE") or "single"
        
        task_id = submit_wan_i2v_job(
            prompt=animation_prompt,
            first_frame_url=image_url,
            duration_s=duration_s,
            resolution=i2v_resolution,
            aspect_ratio="16:9",
            model=model,
            negative_prompt="logos, watermarks, brand marks, extra limbs, distorted faces, flicker, glitch, people, human, face, hands",
            shot_type=i2v_shot_type,
        )
        
        print(f"⏳ Waiting for I2V generation (task_id: {task_id})...")
        video_url = wait_for_wan_result(task_id)
        download_file(video_url, out_mp4)
        
        print(f"✅ Segment {frame_id} OK: {out_mp4.name}")

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

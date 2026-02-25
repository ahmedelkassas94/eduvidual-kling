from __future__ import annotations

from pathlib import Path
import subprocess
import os
import shutil
from typing import Optional, Tuple, Dict, Any

from wan_client import submit_wan_job, wait_for_wan_result, download_file


# ---------------------------------------------------------
# ENV / SAFETY
# ---------------------------------------------------------
def is_dry_run() -> bool:
    return os.getenv("DRY_RUN", "true").strip().lower() == "true"


def _ffmpeg_exe() -> str:
    """Return path to ffmpeg: project bundle, imageio-ffmpeg, or system ffmpeg."""
    import platform
    base = Path(__file__).resolve().parent / "ffmpeg" / "bin"
    on_windows = platform.system() == "Windows"
    # Windows: use .exe from bundle if present
    if on_windows:
        exe_win = base / "ffmpeg.exe"
        if exe_win.exists():
            return str(exe_win)
    # macOS/Linux: use binary without .exe from bundle
    exe_unix = base / "ffmpeg"
    if exe_unix.exists() and os.access(exe_unix, os.X_OK):
        return str(exe_unix)
    # Bundled ffmpeg from imageio-ffmpeg (pip install imageio-ffmpeg)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    # System ffmpeg from PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    raise FileNotFoundError(
        "ffmpeg not found. Either:\n"
        "  - pip install imageio-ffmpeg (provides a bundled ffmpeg), or\n"
        "  - Install ffmpeg (e.g. brew install ffmpeg) and ensure it is on PATH, or\n"
        "  - Place ffmpeg in: project/ffmpeg/bin/ffmpeg (macOS/Linux) or ffmpeg.exe (Windows)"
    )


# ---------------------------------------------------------
# VIDEO GENERATION (BACKWARD COMPAT)
# ---------------------------------------------------------
def generate_placeholder_video(
    clip_id: int,
    duration_s: int,
    prompt: str,
    negative_prompt: str,
    output_dir: Path,
) -> Path:
    """
    Backward-compatible / safe generator:
    - DRY_RUN=true  -> generates a black placeholder MP4 via ffmpeg
    - DRY_RUN=false -> calls WAN (paid)

    Uses clip naming (clip_###.mp4).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"clip_{clip_id:03d}.mp4"

    if is_dry_run():
        print("[DRY RUN] DRY RUN ENABLED - generating placeholder MP4 (no paid API call)")
        _make_black_clip(out_file, duration_s)
        return out_file

    # REAL COST HAPPENS HERE
    task_id = submit_wan_job(
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration_s=duration_s,
        resolution="720p",
        aspect_ratio="16:9",
        model=os.getenv("WAN_T2V_MODEL") or "wan2.5-t2v-preview",
    )
    video_url = wait_for_wan_result(task_id)
    return download_file(video_url, out_file)


def is_video_valid(video_path: Path) -> bool:
    if not video_path.exists():
        return False
    # Placeholder black clips can be small; keep threshold low but non-trivial
    return video_path.stat().st_size >= 10_000


# ---------------------------------------------------------
# NEW API (FOR CHAINING + CONTINUITY REPORTING)
# ---------------------------------------------------------
def generate_clip_video(
    clip_id: int,
    duration_s: int,
    prompt: str,
    negative_prompt: str,
    output_dir: Path,
    *,
    first_frame_url: Optional[str] = None,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Unified generator used by the orchestrator.

    Returns:
      (video_path, meta)

    meta keys:
      - mode: "T2V" | "I2V" | "T2V_DRY_RUN" | "I2V_DRY_RUN"
      - model_used: str | None
      - used_first_frame_url: bool
      - fallback_used: bool
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"clip_{clip_id:03d}.mp4"

    # ---------------------------
    # DRY RUN (NO COST) — but still report intended chaining
    # ---------------------------
    if is_dry_run():
        print("[DRY RUN] DRY RUN ENABLED - generating placeholder MP4 (no paid API call)")
        _make_black_clip(out_file, duration_s)

        intended_mode = "I2V_DRY_RUN" if first_frame_url else "T2V_DRY_RUN"
        intended_model = (
            (os.getenv("WAN_I2V_MODEL") or "wan2.6-i2v") if first_frame_url
            else (os.getenv("WAN_T2V_MODEL") or "wan2.5-t2v-preview")
        )

        meta = {
            "mode": intended_mode,
            "model_used": intended_model,
            "used_first_frame_url": bool(first_frame_url),
            "fallback_used": False,
        }
        return out_file, meta

    # ---------------------------
    # REAL RUN (PAID): Prefer I2V when first_frame_url is available
    # ---------------------------
    if first_frame_url:
        try:
            from wan_client import submit_wan_i2v_job  # type: ignore

            model = os.getenv("WAN_I2V_MODEL") or "wan2.6-i2v"

            # Fix #3: use I2V-specific env resolution + shot type
            i2v_resolution = os.getenv("WAN_I2V_RESOLUTION") or "720P"
            i2v_shot_type = os.getenv("WAN_I2V_SHOT_TYPE") or "single"

            task_id = submit_wan_i2v_job(
                prompt=prompt,
                first_frame_url=first_frame_url,
                duration_s=duration_s,
                resolution=i2v_resolution,
                aspect_ratio=aspect_ratio,  # now accepted (ignored in payload)
                model=model,
                negative_prompt=negative_prompt,
                shot_type=i2v_shot_type,
            )
            video_url = wait_for_wan_result(task_id)
            video_path = download_file(video_url, out_file)

            meta = {
                "mode": "I2V",
                "model_used": model,
                "used_first_frame_url": True,
                "fallback_used": False,
            }
            return video_path, meta

        except Exception as e:
            print("[WARN] I2V path failed; falling back to T2V.")
            print(f"Reason: {e}")

            # Fall back to T2V (paid)
            model = os.getenv("WAN_T2V_MODEL") or "wan2.5-t2v-preview"
            task_id = submit_wan_job(
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration_s=duration_s,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                model=model,
            )
            video_url = wait_for_wan_result(task_id)
            video_path = download_file(video_url, out_file)

            meta = {
                "mode": "T2V",
                "model_used": model,
                "used_first_frame_url": True,   # input existed, but not used successfully
                "fallback_used": True,
            }
            return video_path, meta

    # ---------------------------
    # T2V path (paid)
    # ---------------------------
    model = os.getenv("WAN_T2V_MODEL") or "wan2.5-t2v-preview"
    task_id = submit_wan_job(
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration_s=duration_s,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        model=model,
    )
    video_url = wait_for_wan_result(task_id)
    video_path = download_file(video_url, out_file)

    meta = {
        "mode": "T2V",
        "model_used": model,
        "used_first_frame_url": False,
        "fallback_used": False,
    }
    return video_path, meta


# Backward-compatible alias for older imports
def generate_scene_video(*args, **kwargs):
    return generate_clip_video(*args, **kwargs)


# ---------------------------------------------------------
# FRAME EXTRACTION (ROBUST)
# ---------------------------------------------------------
def extract_last_frame(video_path: Path, out_png: Path) -> Path:
    """
    Extract a near-last frame as PNG (robust across mp4/keyframes).
    Retries with progressively wider windows.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()

    offsets = ["-0.1", "-0.25", "-0.5", "-1.0"]

    for off in offsets:
        cmd = [
            ffmpeg,
            "-y",
            "-sseof", off,
            "-i", video_path.as_posix(),
            "-frames:v", "1",
            out_png.as_posix(),
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if out_png.exists() and out_png.stat().st_size > 0:
            return out_png

    raise RuntimeError(
        f"Failed to extract last frame from {video_path}. Tried offsets: {offsets}"
    )


# ---------------------------------------------------------
# INTERNAL
# ---------------------------------------------------------
def _make_black_clip(out_file: Path, duration_s: int) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:r=30",
        "-t", str(duration_s),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        out_file.as_posix(),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------
# IMAGE-BASED ANIMATION HELPERS
# ---------------------------------------------------------
def trim_video_to_duration(
    video_path: Path,
    target_duration_s: float,
    output_path: Optional[Path] = None,
    *,
    keep_end: bool = False,
) -> Path:
    """
    Trim a video to an exact duration using ffmpeg.
    If output_path is None, uses a temporary file then replaces the original.
    When keep_end is True, keeps the *last* target_duration_s seconds (trims from start).
    Use keep_end=True for first+last frame Veo clips so the video ends on the provided last frame.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg = _ffmpeg_exe()

    # Probe duration (ffmpeg may exit non-zero e.g. 8 on some builds but still print duration to stdout)
    probe_cmd = [
        ffmpeg,
        "-i", str(path),
        "-show_entries", "format=duration",
        "-v", "quiet",
        "-of", "csv=p=0",
    ]
    actual_duration = None
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10, cwd=path.parent)
        if result.stdout and result.stdout.strip():
            try:
                actual_duration = float(result.stdout.strip())
            except ValueError:
                pass
        if actual_duration is not None and actual_duration > 0:
            if abs(actual_duration - target_duration_s) < 0.1:
                print(f"[INFO] Video duration ({actual_duration:.2f}s) already matches target ({target_duration_s}s), skipping trim")
                return path
    except subprocess.TimeoutExpired:
        print(f"[WARN] Video duration probe timed out. Proceeding with trim anyway...")
    except Exception as e:
        print(f"[WARN] Could not probe video duration: {e}. Proceeding with trim anyway...")

    # Use temporary file to avoid overwriting input while reading
    if output_path is None:
        temp_path = path.parent / f"{path.stem}_temp{path.suffix}"
        out_path = temp_path
    else:
        out_path = Path(output_path).resolve()

    if keep_end:
        # Keep last N seconds: start reading target_duration_s before end (-sseof -N), output N seconds
        cmd = [
            ffmpeg,
            "-y",
            "-sseof", f"-{target_duration_s}",
            "-i", str(path),
            "-t", str(target_duration_s),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
        cmd_reencode = [
            ffmpeg,
            "-y",
            "-sseof", f"-{target_duration_s}",
            "-i", str(path),
            "-t", str(target_duration_s),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
    else:
        # Keep first N seconds (default)
        cmd = [
            ffmpeg,
            "-y",
            "-i", str(path),
            "-t", str(target_duration_s),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
        cmd_reencode = [
            ffmpeg,
            "-y",
            "-i", str(path),
            "-t", str(target_duration_s),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except subprocess.CalledProcessError:
        print(f"[WARN] Stream copy trim failed, re-encoding...")
        subprocess.run(cmd_reencode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg trim operation timed out for {path}")

    # If we used a temp file, replace the original
    if output_path is None:
        import shutil
        path.unlink(missing_ok=True)
        shutil.move(str(out_path), str(path))
        return path

    return out_path


def strip_audio_from_video(video_path: Path) -> Path:
    """
    Remove audio track from a video file in-place.
    Clips are kept video-only; final audio is added from the main narration script at stitch time.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    ffmpeg = _ffmpeg_exe()
    temp_path = path.parent / f"{path.stem}_no_audio{path.suffix}"
    cmd = [ffmpeg, "-y", "-i", str(path), "-an", "-c:v", "copy", str(temp_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except subprocess.CalledProcessError as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to strip audio from {path}: {e}") from e
    path.unlink(missing_ok=True)
    shutil.move(str(temp_path), str(path))
    return path


def animate_still_to_video(
    image_path: Path,
    out_file: Path,
    duration_s: int,
    animation_type: str = "hold",
) -> None:
    """
    Turn a single still image into a short animated segment using ffmpeg.

    animation_type controls a simple camera move over the still:
      - "hold": no motion, just a static hold
      - "ken_burns_zoom_in": slow zoom-in
      - "ken_burns_zoom_out": slow zoom-out
      - "pan_left": slow horizontal pan left
      - "pan_right": slow horizontal pan right
      - "pan_up": slow vertical pan upward
      - "pan_down": slow vertical pan downward
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()

    atype = (animation_type or "hold").strip().lower()

    if atype == "hold":
        vf = "scale=1280:720"
    elif atype == "ken_burns_zoom_in":
        vf = (
            "scale=1280:720,"
            "zoompan=z='min(zoom+0.0015,1.5)':"
            "d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        )
    elif atype == "ken_burns_zoom_out":
        vf = (
            "scale=1280:720,"
            "zoompan=z='max(zoom-0.0015,1.0)':"
            "d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        )
    elif atype == "pan_left":
        vf = (
            "scale=1920:720,"
            "crop=1280:720:"
            "x='(in_w-out_w)-((in_w-out_w)/("
            + str(max(duration_s, 1))
            + "*25))*n':y=0"
        )
    elif atype == "pan_right":
        vf = (
            "scale=1920:720,"
            "crop=1280:720:"
            "x='((in_w-out_w)/("
            + str(max(duration_s, 1))
            + "*25))*n':y=0"
        )
    elif atype == "pan_up":
        vf = (
            "scale=1280:1080,"
            "crop=1280:720:"
            "x=0:y='(in_h-out_h)-((in_h-out_h)/("
            + str(max(duration_s, 1))
            + "*25))*n'"
        )
    elif atype == "pan_down":
        vf = (
            "scale=1280:1080,"
            "crop=1280:720:"
            "x=0:y='((in_h-out_h)/("
            + str(max(duration_s, 1))
            + "*25))*n'"
        )
    else:
        # Fallback to a simple static hold
        vf = "scale=1280:720"

    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-i",
        image_path.as_posix(),
        "-t",
        str(duration_s),
        "-r",
        "25",
        "-vf",
        vf,
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        out_file.as_posix(),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

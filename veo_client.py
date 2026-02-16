"""
Veo 3 Fast video generation via Gemini API.
Supports image-to-video (I2V) with native audio.
Veo 3.1 accepts multiple images: one as first frame (image) + up to 3 reference_images.
Uses GEMINI_API_KEY (same as image_client).
"""
import os
import time
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Veo 3.1 standard (quality over speed); override with VEO_MODEL in .env
DEFAULT_VEO_MODEL = "veo-3.1-generate-preview"
# Fast variant: veo-3.1-fast-generate-preview

# Supported durations (seconds) for Veo 3.1; 8s required when using reference_images
VEO_DURATIONS = (4, 6, 8)
# Max reference images (Veo 3.1 supports up to 3)
VEO_MAX_REFERENCE_IMAGES = 3


def _duration_seconds(duration_s: int) -> str:
    """Map requested duration to Veo-supported value (4, 6, or 8)."""
    if duration_s <= 4:
        return "4"
    if duration_s <= 6:
        return "6"
    return "8"


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env (required for Veo)")
    return genai.Client(api_key=api_key)


def _load_image_for_veo(image_path: Path):
    """Load image from path for Veo API (types.Image.from_file)."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return types.Image.from_file(location=str(image_path))


def submit_veo_i2v_job(
    prompt: str,
    image_path: Path,
    duration_s: int,
    *,
    last_frame_path: Optional[Path] = None,
    reference_image_paths: Optional[List[Path]] = None,
    model: Optional[str] = None,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    negative_prompt: Optional[str] = None,
):
    """
    Submit a Veo image-to-video job.
    - image_path: first frame (required).
    - last_frame_path: optional; when set, Veo generates video that starts with the first frame
      and ends with the last frame (frame-specific generation).
    - reference_image_paths: optional list of up to 3 additional images (Veo 3.1 "reference images")
      to guide content; no compositing needed.
    Returns an operation object; poll with wait_for_veo_result().
    Video will include native audio (dialogue/SFX/ambience from prompt).
    """
    client = _client()
    model_name = (model or os.getenv("VEO_MODEL") or DEFAULT_VEO_MODEL).strip()
    image_part = _load_image_for_veo(image_path)

    # Veo 3.1: up to 3 reference images (no compositing); last_frame for start/end
    ref_paths = (reference_image_paths or [])[: VEO_MAX_REFERENCE_IMAGES]
    use_last_frame = last_frame_path and Path(last_frame_path).exists()
    # API requires duration_seconds="8" when using reference_images
    duration_param = "8" if ref_paths else _duration_seconds(duration_s)

    config_kw: dict = {
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "duration_seconds": duration_param,
        "negative_prompt": negative_prompt or None,
    }

    if use_last_frame:
        config_kw["last_frame"] = _load_image_for_veo(Path(last_frame_path))

    if ref_paths:
        ref_images = []
        for p in ref_paths:
            if not Path(p).exists():
                continue
            img = _load_image_for_veo(Path(p))
            ref_images.append(types.VideoGenerationReferenceImage(image=img, reference_type="asset"))
        if ref_images:
            config_kw["reference_images"] = ref_images

    config = types.GenerateVideosConfig(**config_kw)

    print("[Veo] I2V payload preview (Gemini API):")
    print(f"   Model: {model_name}")
    print(f"   Image (first frame): {image_path.name}")
    if use_last_frame:
        print(f"   Last frame (end): {Path(last_frame_path).name}")
    if ref_paths:
        print(f"   Reference images: {[Path(p).name for p in ref_paths]}")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Duration: {duration_param}s (Veo), requested {duration_s}s")
    print(f"   Resolution: {resolution}, Aspect: {aspect_ratio}")

    operation = client.models.generate_videos(
        model=model_name,
        prompt=prompt,
        image=image_part,
        config=config,
    )
    return client, operation


def wait_for_veo_result(
    client: genai.Client,
    operation,
    poll_s: int = 15,
    timeout_s: int = 600,
):
    """
    Poll until the Veo operation is done.
    Returns (client, generated_video) so caller can save to path.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if operation.done:
            result = getattr(operation, "result", None) or getattr(operation, "response", None)
            if not result or not getattr(result, "generated_videos", None):
                raise RuntimeError("Veo operation finished but no video in response")
            return client, result.generated_videos[0]
        time.sleep(poll_s)
        operation = client.operations.get(operation=operation)
    raise TimeoutError(f"Veo operation timed out after {timeout_s}s")


def save_veo_video(client: genai.Client, generated_video, out_path: Path) -> Path:
    """Download the generated video and save to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_file = generated_video.video
    client.files.download(file=video_file)
    if hasattr(video_file, "save"):
        video_file.save(str(out_path))
    elif hasattr(video_file, "video_bytes") and video_file.video_bytes:
        out_path.write_bytes(video_file.video_bytes)
    else:
        raise RuntimeError("Veo video has no save() or video_bytes")
    return out_path

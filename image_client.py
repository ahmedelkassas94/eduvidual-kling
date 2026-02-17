import os
from pathlib import Path
from typing import Optional, List

import env_loader  # noqa: F401 - load .env from project root first
from google import genai
from PIL import Image


def _response_parts_or_raise(response, context: str = "Gemini"):
    """Return response.parts, or raise a clear error if None/empty (e.g. blocked by safety)."""
    if response is None:
        raise RuntimeError(f"{context}: API returned no response.")
    parts = getattr(response, "parts", None)
    if parts is None or (hasattr(parts, "__len__") and len(parts) == 0):
        reason = "no content returned"
        if getattr(response, "candidates", None) and len(response.candidates) > 0:
            c = response.candidates[0]
            reason = getattr(c, "finish_reason", None) or getattr(c, "finishReason", None) or reason
            if getattr(c, "safety_ratings", None):
                reason = f"{reason} (safety_ratings may have blocked)"
        raise RuntimeError(
            f"{context}: {reason}. Try a different prompt or image; "
            "if blocked by safety filters, simplify the scene."
        )
    return parts


def generate_image(
    prompt: str,
    out_path: Path,
    *,
    size: str = "1280x720",
    model: Optional[str] = None,
) -> Path:
    """
    Generate a single image from text using Google's Gemini image model
    via the Google GenAI SDK.

    Configuration:
      - GEMINI_API_KEY: your Gemini API key from Google AI Studio
      - IMAGE_MODEL (optional): override model name; defaults to "gemini-3-pro-image-preview"
    """
    api_key = env_loader.require_env("GEMINI_API_KEY", "Image generation (T2I) requires Gemini.")

    client = genai.Client(api_key=api_key)
    # Default: Gemini 3 Pro Image (override with IMAGE_MODEL in .env)
    model_name = (model or env_loader.get_env("IMAGE_MODEL") or "gemini-3-pro-image-preview").strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Gemini image models use content generation with inline image data.
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
    )

    parts = _response_parts_or_raise(response, "Gemini T2I")
    image_saved = False
    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            img = part.as_image()
            img.save(out_path)
            image_saved = True
            break

    if not image_saved:
        raise RuntimeError("Gemini image generation did not return any image data.")

    return out_path


def generate_image_from_images(
    prompt: str,
    image_paths: List[Path],
    out_path: Path,
    *,
    model: Optional[str] = None,
) -> Path:
    """
    Generate an image from multiple input images (I2I) using Gemini image model
    (gemini-3-pro-image-preview), which supports up to 14 input images.
    
    The prompt should describe in EXTREME DETAIL:
    - Spatial relationships between all objects (positions relative to each other and camera)
    - Camera position and angle (e.g., "camera is facing directly at the scene", "mirror is placed above the table")
    - Exact placement of each ingredient in the final composition
    - Lighting, shadows, and how objects interact spatially
    
    Args:
        prompt: Detailed description of spatial relationships, camera position, and composition
        image_paths: List of ingredient images (isolated objects) to compose (up to 14)
        out_path: Output path for the generated composite image
        model: Optional model override (defaults to gemini-3-pro-image-preview)
    
    Returns:
        Path to the generated image
    """
    api_key = env_loader.require_env("GEMINI_API_KEY", "Image generation (T2I) requires Gemini.")

    client = genai.Client(api_key=api_key)
    model_name = (model or env_loader.get_env("I2I_MODEL") or "gemini-3-pro-image-preview").strip()

    if len(image_paths) > 14:
        raise ValueError(f"Model supports max 14 images, got {len(image_paths)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build contents: list of PIL Image objects + text prompt
    # The GenAI SDK accepts PIL Image objects in the contents list
    contents = []
    for img_path in image_paths:
        if not img_path.exists():
            raise FileNotFoundError(f"Ingredient image not found: {img_path}")
        # Load image as PIL Image and add to contents
        img = Image.open(img_path)
        contents.append(img)
    
    # Add text prompt describing spatial relationships
    contents.append(prompt)

    # Call I2I API
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
    )

    parts = _response_parts_or_raise(response, "Gemini I2I")
    image_saved = False
    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            img = part.as_image()
            img.save(out_path)
            image_saved = True
            break

    if not image_saved:
        raise RuntimeError("Gemini I2I generation did not return any image data.")

    return out_path


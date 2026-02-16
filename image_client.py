import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from google import genai
from PIL import Image


load_dotenv()


def generate_image(
    prompt: str,
    out_path: Path,
    *,
    size: str = "1280x720",
    model: Optional[str] = None,
) -> Path:
    """
    Generate a single image from text using Google's Gemini Pro image model
    (Gemini 3 Pro Image / Nano Banana Pro) via the Google GenAI SDK.

    Configuration:
      - GEMINI_API_KEY: your Gemini API key from Google AI Studio
      - IMAGE_MODEL (optional): override model name; defaults to "gemini-3-pro-image-preview" (Gemini Pro Vision / image generation)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    # Default: Gemini 3 Pro Image (Nano Banana Pro) - Pro-tier image generation
    model_name = (model or os.getenv("IMAGE_MODEL") or "gemini-3-pro-image-preview").strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Gemini image models use content generation with inline image data.
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
    )

    # Save the first returned image part to disk.
    image_saved = False
    for part in response.parts:
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
    Generate an image from multiple input images (I2I) using Nano Banana Pro
    (gemini-3-pro-image-preview) which supports up to 14 input images.
    
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    model_name = (model or os.getenv("I2I_MODEL") or "gemini-3-pro-image-preview").strip()

    if len(image_paths) > 14:
        raise ValueError(f"Nano Banana Pro supports max 14 images, got {len(image_paths)}")

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

    # Save the first returned image part to disk
    image_saved = False
    for part in response.parts:
        if getattr(part, "inline_data", None) is not None:
            img = part.as_image()
            img.save(out_path)
            image_saved = True
            break

    if not image_saved:
        raise RuntimeError("Gemini I2I generation did not return any image data.")

    return out_path


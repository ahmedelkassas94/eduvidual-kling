import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai


load_dotenv()


def generate_image(
    prompt: str,
    out_path: Path,
    *,
    size: str = "1280x720",
    model: Optional[str] = None,
) -> Path:
    """
    Generate a single image from text using Google's Gemini "Nano Banana"
    image model (gemini-2.5-flash-image) via the Google GenAI SDK.

    Configuration:
      - GEMINI_API_KEY: your Gemini API key from Google AI Studio
      - IMAGE_MODEL (optional): override model name; defaults to "gemini-2.5-flash-image"
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    model_name = (model or os.getenv("IMAGE_MODEL") or "gemini-2.5-flash-image").strip()

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


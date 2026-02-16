"""
Optional sound effects generation via ElevenLabs Text-to-Sound Effects API.
If ELEVENLABS_API_KEY is not set, no SFX are generated (caller uses silence or skips).
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def generate_sound_effect(
    text: str,
    duration_seconds: float,
    output_path: Path,
) -> Optional[Path]:
    """
    Generate a sound effect from a text description using ElevenLabs API.
    duration_seconds: 0.5 to 30. Clamped if out of range.
    Returns output_path if successful, None if API key missing or request fails.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None

    duration_seconds = max(0.5, min(30.0, float(duration_seconds)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
    except ImportError:
        return None

    url = "https://api.elevenlabs.io/v1/sound-generation"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "duration_seconds": duration_seconds,
        "prompt_influence": 0.4,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return output_path
    except Exception as e:
        print(f"[WARN] Sound effect generation failed: {e}")
        return None

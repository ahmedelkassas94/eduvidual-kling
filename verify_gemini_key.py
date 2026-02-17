#!/usr/bin/env python3
"""
Verify that GEMINI_API_KEY from .env is accepted by the Gemini API.
Run from project root: python verify_gemini_key.py
If this passes locally but the planner fails in Cursor, your key likely has
IP/referrer restrictions — create a new key with no application restrictions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env_loader
from google import genai
from google.genai import errors as genai_errors

def main():
    key = env_loader.require_env("GEMINI_API_KEY", "GEMINI_API_KEY not set.")
    client = genai.Client(api_key=key)
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents="Say OK")
        text = getattr(r, "text", None) or ""
        if "ok" in text.lower() or len(text) > 0:
            print("OK — Gemini API key is valid.")
            return 0
    except genai_errors.ClientError as e:
        err = str(e).lower()
        if "api key" in err and ("invalid" in err or "400" in err):
            print("Gemini rejected your API key.", file=sys.stderr)
            print(
                "Fix: Create a NEW key at https://aistudio.google.com/apikey with NO application restrictions.",
                file=sys.stderr,
            )
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

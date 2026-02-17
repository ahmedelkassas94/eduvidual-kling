"""
Single source of truth for loading .env from the project root.
Import this first in entry points (planner, orchestrator, etc.) so all code
relies on the same .env file regardless of current working directory.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = directory containing this file (Eduvidual repo root)
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load once; override=True so .env wins over existing env vars
load_dotenv(ENV_PATH, override=True)


def get_env(key: str, default: str = "") -> str:
    """Get env var (after .env load). Use for optional vars."""
    return (os.getenv(key) or default).strip()


def require_env(key: str, purpose: str = "") -> str:
    """
    Get required env var; raise with a clear message if missing.
    Use so errors point to .env in project root.
    """
    value = (os.getenv(key) or "").strip()
    if not value:
        msg = (
            f"{key} is not set. Add it to your .env file in the project root.\n"
            f"  File: {ENV_PATH}\n"
            f"  Example: {key}=your_key_here"
        )
        if purpose:
            msg = f"{purpose}\n{msg}"
        raise RuntimeError(msg)
    return value


def check_planner_env() -> None:
    """Verify env needed for the planner (Gemini). Call at planner startup."""
    require_env("GEMINI_API_KEY", "Planner requires Gemini for script generation.")

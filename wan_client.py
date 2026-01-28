import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------
# ENV SETUP
# ---------------------------------------------------------
load_dotenv()

# Alibaba Cloud Model Studio (DashScope) API key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("WAN_API_KEY")

# Region selection:
# - Singapore (intl): dashscope-intl.aliyuncs.com
# - Beijing: dashscope.aliyuncs.com
DASHSCOPE_REGION = (os.getenv("DASHSCOPE_REGION") or "sg").strip().lower()

BASE_URL = (
    "https://dashscope-intl.aliyuncs.com"
    if DASHSCOPE_REGION in {"sg", "singapore", "intl", "international"}
    else "https://dashscope.aliyuncs.com"
)

# Optional overrides (advanced; usually leave unset)
WAN_CREATE_URL = os.getenv("WAN_CREATE_URL", "").strip()
WAN_STATUS_URL = os.getenv("WAN_STATUS_URL", "").strip()

# Default endpoints
DEFAULT_T2V_CREATE_URL = f"{BASE_URL}/api/v1/services/aigc/video-generation/video-synthesis"
# ✅ Fixed: I2V uses the same endpoint as T2V, not /image2video/ (that's for EMO model only)
DEFAULT_I2V_CREATE_URL = f"{BASE_URL}/api/v1/services/aigc/video-generation/video-synthesis"
DEFAULT_TASK_URL_BASE = f"{BASE_URL}/api/v1/tasks"

CREATE_T2V_URL = WAN_CREATE_URL or DEFAULT_T2V_CREATE_URL
CREATE_I2V_URL = DEFAULT_I2V_CREATE_URL
TASK_URL_BASE = WAN_STATUS_URL or DEFAULT_TASK_URL_BASE


# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------
def _headers(async_enable: bool = True) -> dict:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "Missing API key. Set DASHSCOPE_API_KEY in .env "
            "(recommended), or WAN_API_KEY as a fallback."
        )

    h = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    if async_enable:
        h["X-DashScope-Async"] = "enable"
    return h


def _size_from_resolution(resolution: str) -> str:
    """
    T2V uses `size` as WIDTH*HEIGHT (e.g., 1280*720), not "720p".
    """
    r = (resolution or "").strip().lower()
    if r in {"480p", "480"}:
        return "854*480"
    if r in {"720p", "720"}:
        return "1280*720"
    if r in {"1080p", "1080"}:
        return "1920*1080"
    return "1280*720"


# ---------------------------------------------------------
# WAN (DashScope) API CLIENT
# ---------------------------------------------------------
def submit_wan_job(
    prompt: str,
    duration_s: int,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",  # kept for signature compatibility
    model: str = "wan2.5-t2v-preview",
    negative_prompt: Optional[str] = None,
    prompt_extend: bool = False,
    watermark: bool = False,
) -> str:
    """
    Submit a WAN text-to-video job.
    RETURNS: task_id
    COST: YES (guarded by DRY_RUN upstream)
    """

    # Preview tiers commonly support 5s or 10s for T2V
    duration = 5 if duration_s <= 5 else 10

    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "size": _size_from_resolution(resolution),
            "duration": duration,
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
        },
    }

    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt

    print("🧾 DashScope WAN T2V payload preview (no cost yet):")
    print(payload)
    print(f"🧾 DashScope endpoint: {CREATE_T2V_URL}")

    r = requests.post(
        CREATE_T2V_URL,
        headers=_headers(async_enable=True),
        json=payload,
        timeout=60,
    )

    # ✅ Fix #2: make 400 errors readable
    if not r.ok:
        raise RuntimeError(f"T2V HTTP {r.status_code}: {r.text}")

    data = r.json()
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"Could not find task_id in response: {data}")

    return task_id


def submit_wan_i2v_job(
    prompt: str,
    first_frame_url: str,
    duration_s: int,
    resolution: str = "720P",
    aspect_ratio: str = "16:9",  # ✅ Fix #1: accept for compatibility (ignored in payload)
    model: str = "wan2.6-i2v",
    negative_prompt: Optional[str] = None,
    prompt_extend: bool = False,
    watermark: bool = False,
    shot_type: Optional[str] = "single",
) -> str:
    """
    Submit a WAN first-frame image-to-video job.
    RETURNS: task_id
    COST: YES (guarded by DRY_RUN upstream)

    Notes:
    - aspect_ratio is accepted for signature compatibility but not used by DashScope I2V.
    - shot_type is optional; use "single" for smooth continuity when supported.
    """

    duration = 5 if duration_s <= 5 else 10

    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "img_url": first_frame_url,  # ✅ Fixed: API uses "img_url" not "first_frame_url"
        },
        "parameters": {
            "resolution": resolution,  # typically 720P/1080P for I2V
            "duration": duration,
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
        },
    }

    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt

    if shot_type:
        payload["parameters"]["shot_type"] = shot_type

    print("🧾 DashScope WAN I2V payload preview (no cost yet):")
    print(f"   Model: {model}")
    print(f"   First Frame URL: {first_frame_url}")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Duration: {duration}s")
    print(f"   Resolution: {resolution}")
    print(f"🧾 DashScope endpoint: {CREATE_I2V_URL}")

    r = requests.post(
        CREATE_I2V_URL,
        headers=_headers(async_enable=True),
        json=payload,
        timeout=60,
    )

    # ✅ Fix #2: make 400 errors readable
    if not r.ok:
        raise RuntimeError(f"I2V HTTP {r.status_code}: {r.text}")

    data = r.json()
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"Could not find task_id in response: {data}")

    return task_id


def wait_for_wan_result(
    task_id: str,
    poll_s: int = 15,
    timeout_s: int = 1800,
) -> str:
    """
    Poll task until completion.
    RETURNS: video_url
    COST: NO (already paid at submission)
    """
    deadline = time.time() + timeout_s
    url = f"{TASK_URL_BASE}/{task_id}"

    while time.time() < deadline:
        r = requests.get(url, headers=_headers(async_enable=False), timeout=60)

        if not r.ok:
            raise RuntimeError(f"Task status HTTP {r.status_code}: {r.text}")

        data = r.json()
        output = data.get("output") or {}
        status = (output.get("task_status") or "").upper()

        if status == "SUCCEEDED":
            video_url = output.get("video_url")
            if not video_url:
                raise RuntimeError(f"Task succeeded but no video_url found: {data}")
            return video_url

        if status in {"FAILED", "CANCELED"}:
            code = output.get("code")
            msg = output.get("message")
            raise RuntimeError(f"DashScope task {status}: {code} {msg} | full={data}")

        time.sleep(poll_s)

    raise TimeoutError(f"DashScope WAN task timed out after {timeout_s}s (task_id={task_id})")


def download_file(url: str, out_path: Path) -> Path:
    """
    Download resulting MP4.
    COST: NO
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300) as r:
        if not r.ok:
            raise RuntimeError(f"Download HTTP {r.status_code}: {r.text}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return out_path

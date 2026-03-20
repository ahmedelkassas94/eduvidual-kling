"""
Kling 3.0 image-to-video (I2V) via direct HTTP calls to Kling (klingai.com).

This repo previously used `fal-client`. For "direct klingai.com" usage, we call
Kling's image-to-video endpoint with:
  - start_image_url (required): first frame
  - duration_s: 3..15 seconds
  - prompt: movement description
  - elements: optional list of up to 2 objects/characters to keep consistent.

Auth + endpoint details are configured via environment variables because different
Kling deployments/documentation use slightly different base URLs/paths:
  - KLINGAI_API_BASE_URL (default: https://api.klingai.com)
  - KLINGAI_IMAGE2VIDEO_ENDPOINT (default: /klingai/v1/videos/image2video)
  - KLINGAI_API_KEY (Bearer token)

If your Kling docs specify a different endpoint/payload/auth header names, update
the env vars and/or this module accordingly.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import base64
import hashlib
import hmac
import json
from typing import Any, Iterable, Optional, List, Dict, Tuple

import requests

import env_loader  # noqa: F401 - load .env from project root first


def _require_klingai_api_key() -> str:
    """
    Kling direct endpoints typically use bearer auth:
      Authorization: Bearer <token>

    For the official `klingai.com` API, this repo supports the common AK/SK flow by
    generating an HS256 JWT from:
      - KLINGAI_ACCESS_KEY
      - KLINGAI_SECRET_KEY
    """
    bearer_token = (os.getenv("KLINGAI_API_KEY") or "").strip()
    if bearer_token:
        return bearer_token

    # Some Kling deployments also accept a single bearer token, but if you only
    # have Access Key + Secret Key, the official API usually expects a JWT.
    access_key = (os.getenv("KLINGAI_ACCESS_KEY") or "").strip()
    secret_key = (os.getenv("KLINGAI_SECRET_KEY") or "").strip()

    if access_key and not secret_key:
        # If only access key is provided, fall back to bearer access key mode.
        return access_key

    if access_key and secret_key:
        combine = (
            os.getenv("KLINGAI_AUTH_COMBINE_ACCESS_SECRET", "").strip().lower()
            in {"1", "true", "yes", "y", "on"}
        )
        if combine:
            # Non-JWT fallback mode (some third-party gateways)
            return f"{access_key}:{secret_key}"

        # Default: JWT auth.
        auth_mode = (os.getenv("KLINGAI_AUTH_MODE") or "").strip().lower() or "jwt"
        if auth_mode in {"jwt", "hs256"}:
            return _generate_kling_jwt(access_key=access_key, secret_key=secret_key)
        if auth_mode in {"bearer_access", "access"}:
            return access_key

        raise RuntimeError(
            "Invalid KLINGAI_AUTH_MODE. Use 'jwt' (default) or 'bearer_access'."
        )

    # Backwards-compat: if user reuses the old variable, still work.
    legacy_fal_key = (os.getenv("FAL_KEY") or "").strip()
    if legacy_fal_key:
        return legacy_fal_key

    raise RuntimeError(
        "Missing Kling auth. Set KLINGAI_API_KEY in .env (or reuse FAL_KEY as a fallback)."
    )


def _b64url_encode(raw: bytes) -> str:
    """RFC7515 base64url without padding."""
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _generate_kling_jwt(*, access_key: str, secret_key: str) -> str:
    """
    Generate HS256 JWT for klingai.com-style AK/SK authentication.
    """
    now = int(time.time())
    exp_seconds = int(os.getenv("KLINGAI_JWT_EXP_SECONDS", "1800").strip() or "1800")

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": access_key,
        "exp": now + exp_seconds,
        "nbf": now - 5,
    }

    header_json = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    header_b64 = _b64url_encode(header_json)
    payload_b64 = _b64url_encode(payload_json)
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    sig = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _extract_video_url_from_json(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract a playable video URL from a Kling response.

    Kling implementations vary (sync vs async, and response nesting), so we try
    several common shapes.
    """
    def _deep_find(obj: Any) -> Optional[str]:
        # Recursively find likely URL strings.
        if isinstance(obj, str):
            s = obj.strip()
            if not s:
                return None
            if s.startswith("http") and ("mp4" in s.lower() or "video" in s.lower()):
                return s
            return None
        if isinstance(obj, dict):
            # Prefer specific keys when present.
            for k in ("video_url", "download_url", "file_url", "url"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for v in obj.values():
                found = _deep_find(v)
                if found:
                    return found
        if isinstance(obj, list):
            for it in obj:
                found = _deep_find(it)
                if found:
                    return found
        return None

    # Try deep extraction first (works across nesting and gateways).
    found = _deep_find(payload)
    if found:
        return found

    # Fallback: keep previous shallow behavior for compatibility.
    for key in ("video_url", "url", "file_url", "download_url"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("video_url", "url"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _require_env_nonempty(name: str, default: str = "") -> str:
    value = (os.getenv(name) or default).strip()
    if not value:
        raise RuntimeError(f"{name} is empty; set it in .env.")
    return value


def generate_kling_i2v_video(
    *,
    prompt: str,
    start_image_url: str,
    duration_s: int,
    output_path: Path,
    element_frontal_image_urls: Optional[Iterable[str]] = None,
    negative_prompt: Optional[str] = None,
    generate_audio: bool = False,
    shot_type: str = "customize",
) -> Path:
    """Generate a Kling I2V clip (direct API) and save it to output_path."""
    api_key = _require_klingai_api_key()

    duration_s_int = int(duration_s)
    if duration_s_int < 3 or duration_s_int > 15:
        raise ValueError("Kling 3.0 I2V duration_s must be between 3 and 15 seconds (inclusive).")

    base_url = os.getenv("KLINGAI_API_BASE_URL", "https://api.klingai.com").strip().rstrip("/")
    endpoint_env = os.getenv("KLINGAI_IMAGE2VIDEO_ENDPOINT", "").strip()
    if endpoint_env:
        endpoint_env = endpoint_env.strip()

    # Some Kling deployments expose the image-to-video endpoint with different prefixes.
    # If the first choice 404s, we retry a common alternative without the extra `klingai/` prefix.
    default_primary = "/klingai/v1/videos/image2video"
    default_alternative = "/v1/videos/image2video"

    primary_endpoint = endpoint_env or default_primary
    alternative_endpoint = default_alternative if primary_endpoint != default_alternative else ""
    endpoint_candidates = [primary_endpoint] + ([alternative_endpoint] if alternative_endpoint else [])

    # Kling implementations generally follow an image URL + text prompt schema.
    # We keep the payload keys aligned with what kling-style APIs commonly use,
    # and allow overrides via env vars if needed later.
    # Kling direct API expects fully-qualified model ids for v3, e.g.:
    #   klingai/kling-v3.0-i2v
    model_name = os.getenv("KLINGAI_MODEL_NAME", "klingai/kling-v3.0-i2v").strip()
    mode = os.getenv("KLINGAI_MODE", "pro").strip()
    aspect_ratio = os.getenv("KLINGAI_ASPECT_RATIO", "16:9").strip()

    model_name_norm = model_name.strip().lower()

    # Some Kling v3 gateways do not support negative_prompt in the request body.
    negative_prompt_supported = "kling-v3" not in model_name_norm

    payload: Dict[str, Any] = {
        "model_name": model_name,
        # Kling direct APIs typically expect the key name `image`.
        "image": start_image_url,
        "prompt": prompt,
        "duration": duration_s_int,
        "mode": mode,
        "generate_audio": bool(generate_audio),
    }

    # v3 gateways often derive aspect ratio from the input image; remove to avoid validation issues.
    if "kling-v3" not in model_name_norm:
        payload["aspect_ratio"] = aspect_ratio

    if negative_prompt_supported and negative_prompt:
        # Only include if the selected model supports it.
        payload["negative_prompt"] = negative_prompt

    # Keep element images consistent (optional).
    # Kling often supports element placeholders like @Element1 / @Element2 inside the prompt.
    elements: List[Dict[str, Any]] = []
    if element_frontal_image_urls:
        for el_url in list(element_frontal_image_urls)[:2]:
            elements.append({"frontal_image_url": el_url})
    if elements:
        payload["elements"] = elements

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Retry strategy for model validation issues:
    # - Some accounts/endpoints don't support certain v3 variants (e.g. "-omni").
    # - Even when a model exists, some gateways reject certain quality tiers via
    #   "model is not supported". For those cases, we try alternate `mode`
    #   values (std/pro) for the same model before moving to a fallback model.
    # Candidate models to try.
    #
    # Your account/region may not expose every "alias" model id. When the API
    # responds with "model is not supported", we try additional likely ids.
    #
    # You can override the whole list via:
    #   KLINGAI_MODEL_CANDIDATES=a,b,c
    env_model_candidates = (os.getenv("KLINGAI_MODEL_CANDIDATES") or "").strip()
    if env_model_candidates:
        model_candidates = [s.strip() for s in env_model_candidates.split(",") if s.strip()]
    else:
        model_candidates = [model_name]
        if "kling-v3" in model_name_norm:
            # Generic v3 model id commonly supported on this gateway.
            extras: List[str] = []
            if model_name_norm != "kling-v3":
                extras.append("kling-v3")
            # Some gateways also support -0 for v3 3-15s.
            allow_fallback_v3_0 = (
                os.getenv("KLINGAI_ALLOW_FALLBACK_KLING_V3_0", "").strip().lower()
                in {"1", "true", "yes", "y", "on"}
            )
            if allow_fallback_v3_0 and model_name_norm != "kling-v3-0":
                extras.append("kling-v3-0")

            for m in extras:
                if m not in model_candidates:
                    model_candidates.append(m)

    last_resp: requests.Response | None = None
    last_url: str = ""
    last_model_used: str = model_name
    mode_primary = (os.getenv("KLINGAI_MODE", mode) or "").strip() or mode
    # Try alternate tiers when model isn't supported under the current mode.
    # Order matters: try the user-selected mode first.
    mode_candidates: List[str] = []
    if mode_primary.lower() == "pro":
        mode_candidates = ["pro", "std"]
    elif mode_primary.lower() == "std":
        mode_candidates = ["std", "pro"]
    else:
        # Unknown mode: try both
        mode_candidates = [mode_primary, "pro", "std"]
    # de-dup while preserving order
    seen_modes: set[str] = set()
    mode_candidates = [m for m in mode_candidates if not (m in seen_modes or seen_modes.add(m))]

    for model_candidate in model_candidates:
        payload["model_name"] = model_candidate
        model_name_norm_candidate = model_candidate.strip().lower()

        negative_prompt_supported_candidate = "kling-v3" not in model_name_norm_candidate
        if negative_prompt_supported_candidate and negative_prompt:
            payload["negative_prompt"] = negative_prompt
        else:
            payload.pop("negative_prompt", None)

        if "kling-v3" in model_name_norm_candidate:
            payload.pop("aspect_ratio", None)
        else:
            payload["aspect_ratio"] = aspect_ratio

        model_supported_under_some_mode = False
        for mode_candidate in mode_candidates:
            payload["mode"] = mode_candidate

            model_supported_under_some_mode = True
            last_mode_used = mode_candidate
            for candidate_endpoint in endpoint_candidates:
                url = f"{base_url}{candidate_endpoint}"
                last_url = url
                resp = requests.post(url, headers=headers, json=payload, timeout=600)
                last_resp = resp
                last_model_used = model_candidate

                if resp.status_code == 404 and candidate_endpoint != endpoint_candidates[-1]:
                    continue

                # If model isn't supported, try next mode (not next endpoint).
                if resp.status_code == 400:
                    try:
                        maybe_json = resp.json()
                    except Exception:
                        maybe_json = {}
                    code = maybe_json.get("code")
                    msg = (maybe_json.get("message") or "").lower()
                    if code == 1201 and "model is not supported" in msg:
                        # Move to next mode_candidate for the same model.
                        break

                # Any non-(model not supported) response ends the retry for this mode.
                break

            # If we got a response that isn't "model not supported", stop retrying.
            if last_resp is not None and last_resp.status_code != 400:
                break
            if last_resp is not None and last_resp.status_code == 400:
                try:
                    maybe_json = last_resp.json()
                except Exception:
                    maybe_json = {}
                code = maybe_json.get("code")
                msg = (maybe_json.get("message") or "").lower()
                if not (code == 1201 and "model is not supported" in msg):
                    break

        # If we succeeded with any mode, stop outer loop.
        if last_resp is not None and last_resp.ok:
            break

    assert last_resp is not None  # for type checkers
    try:
        resp_json = last_resp.json()
    except Exception:
        resp_json = {}

    if last_resp.ok and isinstance(resp_json, dict):
        video_url = _extract_video_url_from_json(resp_json)
        if not video_url:
            # If Kling is async, we may only get a job id.
            data_obj: Any = resp_json.get("data")

            def _pick(d: Any, keys: List[str]) -> Optional[str]:
                if not isinstance(d, dict):
                    return None
                for k in keys:
                    v = d.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                    # Some APIs return numeric ids
                    if isinstance(v, (int, float)) and v:
                        return str(v)
                return None

            job_id = (
                _pick(resp_json, ["job_id", "id", "task_id"])
                or _pick(data_obj, ["job_id", "id", "task_id"])
                # Some responses wrap the task object inside data:
                or (isinstance(data_obj, dict) and _pick(data_obj.get("task"), ["id", "task_id", "job_id"]))
                or (isinstance(data_obj, dict) and _pick(data_obj.get("data"), ["id", "task_id", "job_id"]))
            )
            if job_id:
                # If the API provides an explicit status URL, prefer it.
                status_url = None
                if isinstance(data_obj, dict):
                    status_url = (
                        data_obj.get("status_url")
                        or data_obj.get("query_url")
                        or data_obj.get("task_status_url")
                    )
                # Poll until we get a URL or time out.
                max_attempts = int(os.getenv("KLINGAI_POLL_MAX_ATTEMPTS", "30").strip() or "30")
                sleep_s = float(os.getenv("KLINGAI_POLL_SLEEP_SECONDS", "2").strip() or "2")

                # Status endpoint candidates:
                # - Some gateways only support specific GET paths.
                # - Your network may block some prefixes (we therefore try multiple).
                env_status_candidates = (os.getenv("KLINGAI_STATUS_URL_CANDIDATES") or "").strip()
                status_url_candidates: List[str] = []
                if env_status_candidates:
                    # Comma-separated full URLs or path templates starting with '/'
                    for s in env_status_candidates.split(","):
                        s = s.strip()
                        if not s:
                            continue
                        if s.startswith("http"):
                            status_url_candidates.append(s)
                        else:
                            status_url_candidates.append(f"{base_url}{s}")
                else:
                    if status_url:
                        status_url_candidates.append(status_url)
                    status_template = (
                        os.getenv(
                            "KLINGAI_STATUS_ENDPOINT_TEMPLATE",
                            "/v1/videos/generations/{job_id}",
                        ).strip()
                        or "/v1/videos/generations/{job_id}"
                    )
                    # Primary template from env
                    status_url_candidates.append(
                        f"{base_url}{status_template.format(job_id=str(job_id))}"
                    )
                    # Common alternatives
                    for t in (
                        "/v1/videos/{job_id}",
                        "/v1/videos/image2video/{job_id}",
                        "/klingai/v1/videos/generations/{job_id}",
                        "/klingai/v1/videos/{job_id}",
                    ):
                        status_url_candidates.append(f"{base_url}{t.format(job_id=str(job_id))}")

                # De-dup in order.
                dedup: List[str] = []
                seen: set[str] = set()
                for u in status_url_candidates:
                    if u in seen:
                        continue
                    seen.add(u)
                    dedup.append(u)
                status_url_candidates = dedup
                last_status_json: Any = None
                last_status_url: str = ""
                for _ in range(max_attempts):
                    for candidate_status_url in status_url_candidates:
                        r_status = requests.get(candidate_status_url, headers=headers, timeout=120)
                        # If it's not JSON (e.g., blocked HTML safety page), skip to next candidate.
                        try:
                            status_json = r_status.json()
                        except Exception:
                            status_json = {}
                        last_status_json = status_json
                        last_status_url = candidate_status_url

                        if isinstance(status_json, dict):
                            maybe_url = _extract_video_url_from_json(status_json)
                            if maybe_url:
                                video_url = maybe_url
                                break

                            # If the API provides explicit completion status, stop early on error.
                            status_text = (
                                str(status_json.get("status") or status_json.get("state") or "").lower()
                            )
                            if status_text in {"error", "failed"}:
                                break

                    if video_url:
                        break
                    time.sleep(sleep_s)

            if not video_url:
                raise RuntimeError(
                    "Kling direct API call succeeded but response did not include a video URL "
                    "and we could not poll/resolve it (if async). "
                    f"Response keys: {list(resp_json.keys())}; data={resp_json.get('data')}. "
                    f"Last status url: {last_status_url}. Last status json keys: "
                    f"{list(last_status_json.keys()) if isinstance(last_status_json, dict) else type(last_status_json)}"
                )
    else:
        # Include response text for easier debugging, but avoid dumping secrets.
        msg = last_resp.text[:2000] if isinstance(last_resp.text, str) else str(last_resp.text)
        raise RuntimeError(f"Kling direct API error {last_resp.status_code} at {last_url}: {msg}")

    # Reuse existing downloader from WAN client.
    from wan_client import download_file  # noqa: WPS433

    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(video_url, output_path)
    return output_path


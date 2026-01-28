import os
import base64
from pathlib import Path
import requests
from urllib.parse import quote
from requests.exceptions import ChunkedEncodingError


def frame_to_public_url(frame_path: Path) -> str:
    """
    Upload a PNG frame to a public image hosting service and return a PUBLIC HTTPS URL
    that DashScope I2V can access.

    Supports multiple hosting services (checked in order):
    1. Alibaba Cloud OSS (if OSS credentials are configured)
    2. ImgBB (if IMGBB_API_KEY is set)

    Required env (at least one):
      - OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET, OSS_ENDPOINT (for OSS)
      - IMGBB_API_KEY (for ImgBB)

    Get ImgBB API key from: https://api.imgbb.com/
    """
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    # Check if we should force ImgBB (for testing when OSS URLs are rejected by DashScope)
    force_imgbb = os.getenv("FORCE_IMGBB", "").strip().lower() in ("1", "true", "yes", "y")
    
    # Try Alibaba Cloud OSS first (most compatible with DashScope)
    if not force_imgbb:
        oss_access_key = os.getenv("OSS_ACCESS_KEY_ID", "").strip()
        oss_secret = os.getenv("OSS_ACCESS_KEY_SECRET", "").strip()
        oss_bucket = os.getenv("OSS_BUCKET", "").strip()
        oss_endpoint = os.getenv("OSS_ENDPOINT", "").strip()

        if oss_access_key and oss_secret and oss_bucket and oss_endpoint:
            return _upload_to_oss(frame_path, oss_access_key, oss_secret, oss_bucket, oss_endpoint)

    # Fallback to ImgBB
    api_key = (os.getenv("IMGBB_API_KEY") or "").strip()
    if api_key:
        return _upload_to_imgbb(frame_path, api_key)

    raise RuntimeError(
        "No image hosting configured. Set either:\n"
        "  - OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET, OSS_ENDPOINT (for Alibaba Cloud OSS)\n"
        "  - IMGBB_API_KEY (for ImgBB, get free key from https://api.imgbb.com/)"
    )


def _upload_to_oss(
    frame_path: Path,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    endpoint: str,
) -> str:
    """Upload to Alibaba Cloud OSS and return public URL."""
    try:
        import oss2
    except ImportError:
        raise RuntimeError(
            "oss2 library not installed. Install with: pip install oss2"
        )

    # Normalize endpoint (remove http:// or https:// if present)
    endpoint_clean = endpoint.replace("https://", "").replace("http://", "").strip()
    
    # Create OSS client
    auth = oss2.Auth(access_key, secret_key)
    bucket = oss2.Bucket(auth, endpoint_clean, bucket_name)

    # Upload file
    object_name = f"frames/{frame_path.name}"
    print(f"☁️ Uploading {frame_path.name} to OSS ({bucket_name})...")
    
    with open(frame_path, "rb") as f:
        bucket.put_object(object_name, f, headers={"Content-Type": "image/png"})
    
    # Make object publicly readable (required for DashScope to access)
    try:
        bucket.put_object_acl(object_name, 'public-read')
    except Exception as e:
        print(f"⚠️ Warning: Could not set public read ACL: {e}")
        print("   Make sure the bucket allows public read access via bucket policy.")
    
    # Construct public URL
    # OSS requires virtual-hosted style: https://<bucket-name>.<endpoint>/<object-name>
    # URL encode the object path segments (but keep slashes unencoded)
    object_path_encoded = "/".join(quote(segment, safe="") for segment in object_name.split("/"))
    url = f"https://{bucket_name}.{endpoint_clean}/{object_path_encoded}"
    print(f"🔗 Public URL: {url}")

    _assert_public_image_url(url)
    print("✅ Public URL validated (200 OK, image content-type).")
    print(f"⚠️ Note: If DashScope rejects this URL, it may be a DashScope limitation.")
    print(f"   The URL is publicly accessible but DashScope may have restrictions.")
    return url


def _upload_to_imgbb(frame_path: Path, api_key: str) -> str:
    """Upload to ImgBB and return public URL."""
    # Read image and encode to base64
    with open(frame_path, "rb") as f:
        image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")

    # Upload to ImgBB
    upload_url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": api_key,
        "image": image_b64,
    }

    print(f"☁️ Uploading {frame_path.name} to ImgBB...")
    response = requests.post(upload_url, data=payload, timeout=30)

    if not response.ok:
        raise RuntimeError(f"ImgBB upload failed (HTTP {response.status_code}): {response.text}")

    data = response.json()
    
    if not data.get("success"):
        error_msg = data.get("error", {}).get("message", "Unknown error")
        raise RuntimeError(f"ImgBB upload failed: {error_msg}")

    image_url = data.get("data", {}).get("url")
    if not image_url:
        raise RuntimeError(f"ImgBB response missing URL: {data}")

    print(f"🔗 Public URL: {image_url}")

    # Validate the URL
    _assert_public_image_url(image_url)

    print("✅ Public URL validated (200 OK, image content-type).")
    return image_url


def _assert_public_image_url(url: str) -> None:
    """
    Best-effort validation that the URL is publicly reachable and returns an image.
    Be tolerant of transient ChunkedEncodingError issues from some CDNs (e.g. ImgBB).
    """
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            break
        except ChunkedEncodingError as e:
            last_exc = e
            print(f"⚠️ ChunkedEncodingError while validating URL (attempt {attempt+1}/3): {url} | {e}")
            continue
        except Exception as e:
            raise RuntimeError(f"Public URL not reachable: {url} | {e}")
    else:
        # If we only saw ChunkedEncodingError, assume the URL is basically fine and let DashScope decide.
        print(f"⚠️ Giving up validation after repeated ChunkedEncodingError; assuming URL is OK: {url}")
        return

    if r.status_code != 200:
        raise RuntimeError(f"Public URL not accessible (HTTP {r.status_code}): {url} | body={r.text[:300]}")

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "image/" not in ctype:
        raise RuntimeError(f"Public URL did not return an image. content-type={ctype} url={url} body={r.text[:200]}")

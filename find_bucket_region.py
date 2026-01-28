"""
Try to find which region the bucket is in.
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import oss2
except ImportError:
    print("❌ oss2 not installed")
    exit(1)

access_key = os.getenv("OSS_ACCESS_KEY_ID", "").strip()
secret_key = os.getenv("OSS_ACCESS_KEY_SECRET", "").strip()
bucket_name = "eduvidual27012026"

if not access_key or not secret_key:
    print("❌ Credentials not set")
    exit(1)

# Try common endpoints
endpoints = [
    "oss-cn-hangzhou.aliyuncs.com",
    "oss-cn-shanghai.aliyuncs.com", 
    "oss-cn-beijing.aliyuncs.com",
    "oss-cn-shenzhen.aliyuncs.com",
    "oss-ap-southeast-1.aliyuncs.com",  # Singapore
    "oss-ap-southeast-2.aliyuncs.com",  # Sydney
    "oss-us-west-1.aliyuncs.com",
    "oss-eu-central-1.aliyuncs.com",  # Frankfurt
]

auth = oss2.Auth(access_key, secret_key)

print(f"🔍 Searching for bucket '{bucket_name}' in different regions...\n")

for endpoint in endpoints:
    try:
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        bucket_info = bucket.get_bucket_info()
        print(f"✅ FOUND! Bucket is in: {endpoint}")
        print(f"   Location: {bucket_info.location}")
        print(f"   Creation: {bucket_info.creation_date}")
        break
    except oss2.exceptions.NoSuchBucket:
        print(f"❌ {endpoint} - Bucket not found")
    except oss2.exceptions.SignatureDoesNotMatch:
        print(f"⚠️  {endpoint} - Signature error (credentials issue)")
    except Exception as e:
        print(f"⚠️  {endpoint} - {type(e).__name__}: {str(e)[:100]}")

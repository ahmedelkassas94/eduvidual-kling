"""
Test which endpoint works for your bucket.
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import oss2
except ImportError:
    print("❌ oss2 not installed. Run: pip install oss2")
    exit(1)

access_key = os.getenv("OSS_ACCESS_KEY_ID", "").strip()
secret_key = os.getenv("OSS_ACCESS_KEY_SECRET", "").strip()
bucket_name = "eduvidual27012026"

if not access_key or not secret_key:
    print("❌ OSS credentials not found in .env")
    exit(1)

# Try common endpoints
common_endpoints = [
    "oss-cn-hangzhou.aliyuncs.com",
    "oss-cn-shanghai.aliyuncs.com",
    "oss-cn-beijing.aliyuncs.com",
    "oss-cn-shenzhen.aliyuncs.com",
    "oss-ap-southeast-1.aliyuncs.com",
    "oss-ap-southeast-2.aliyuncs.com",
    "oss-us-west-1.aliyuncs.com",
    "oss-eu-central-1.aliyuncs.com",
]

print(f"🔍 Testing endpoints for bucket: {bucket_name}\n")

auth = oss2.Auth(access_key, secret_key)
working_endpoint = None

for endpoint in common_endpoints:
    try:
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        # Try to get bucket info (this will fail if wrong endpoint)
        bucket_info = bucket.get_bucket_info()
        working_endpoint = endpoint
        print(f"✅ SUCCESS! Working endpoint: {endpoint}")
        print(f"   Bucket location: {bucket_info.location}")
        print(f"\n📝 Add this to your .env file:")
        print(f"OSS_ENDPOINT={endpoint}")
        break
    except oss2.exceptions.NoSuchBucket:
        print(f"❌ {endpoint} - Bucket not found in this region")
    except oss2.exceptions.AccessDenied:
        print(f"⚠️  {endpoint} - Access denied (might be correct region but wrong permissions)")
    except Exception as e:
        # Wrong region or other error
        pass

if not working_endpoint:
    print("\n❌ Could not determine endpoint automatically.")
    print("\n💡 Please check your OSS Console:")
    print("   1. Go to: https://oss.console.aliyun.com/bucket")
    print("   2. Click on bucket: eduvidual27012026")
    print("   3. Find the 'Region' or 'Endpoint' field")
    print("   4. The endpoint format is: oss-<region>.aliyuncs.com")

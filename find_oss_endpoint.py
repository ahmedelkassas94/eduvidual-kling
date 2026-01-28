"""
Quick script to find your OSS bucket endpoint.
This will list all your buckets and show their endpoints.
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

if not access_key or not secret_key:
    print("❌ OSS credentials not found in .env")
    exit(1)

# Try common endpoints to find your buckets
common_endpoints = [
    "oss-cn-hangzhou.aliyuncs.com",      # China (Hangzhou)
    "oss-cn-shanghai.aliyuncs.com",      # China (Shanghai)
    "oss-cn-beijing.aliyuncs.com",       # China (Beijing)
    "oss-cn-shenzhen.aliyuncs.com",      # China (Shenzhen)
    "oss-ap-southeast-1.aliyuncs.com",  # Singapore
    "oss-ap-southeast-2.aliyuncs.com",  # Sydney
    "oss-us-west-1.aliyuncs.com",       # US West
    "oss-eu-central-1.aliyuncs.com",   # Frankfurt
]

print("🔍 Searching for your buckets...\n")

found_buckets = []
auth = oss2.Auth(access_key, secret_key)

for endpoint in common_endpoints:
    try:
        service = oss2.Service(auth, endpoint)
        buckets = service.list_buckets()
        
        for bucket in buckets.buckets:
            bucket_name = bucket.name
            bucket_location = bucket.location
            bucket_endpoint = f"oss-{bucket_location}.aliyuncs.com" if bucket_location else endpoint
            
            found_buckets.append({
                "name": bucket_name,
                "endpoint": bucket_endpoint,
                "location": bucket_location or "unknown"
            })
            
            print(f"✅ Found bucket: {bucket_name}")
            print(f"   Endpoint: {bucket_endpoint}")
            print(f"   Location: {bucket_location or 'unknown'}\n")
            
    except Exception as e:
        # This endpoint doesn't work, try next one
        continue

if not found_buckets:
    print("❌ No buckets found. Possible reasons:")
    print("   1. Your credentials don't have permission to list buckets")
    print("   2. Your region endpoint is not in the common list")
    print("\n💡 Alternative: Check your OSS Console:")
    print("   1. Go to: https://oss.console.aliyun.com/bucket")
    print("   2. Click on your bucket: eduvidual27012026")
    print("   3. Look for 'Region' or 'Endpoint' in the bucket details")
    print("   4. The endpoint format is usually: oss-<region>.aliyuncs.com")
else:
    # Check if our target bucket is in the list
    target_bucket = "eduvidual27012026"
    for bucket in found_buckets:
        if bucket["name"] == target_bucket:
            print(f"\n🎯 Your bucket '{target_bucket}' endpoint is:")
            print(f"   {bucket['endpoint']}")
            print(f"\nAdd this to your .env file:")
            print(f"OSS_ENDPOINT={bucket['endpoint']}")

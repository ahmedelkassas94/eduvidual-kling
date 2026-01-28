"""
Test OSS credentials and connection.
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
bucket_name = os.getenv("OSS_BUCKET", "").strip()
endpoint = os.getenv("OSS_ENDPOINT", "").strip()

print(f"Access Key ID: {access_key[:10]}...")
print(f"Secret Key: {'*' * len(secret_key) if secret_key else 'NOT SET'}")
print(f"Bucket: {bucket_name}")
print(f"Endpoint: {endpoint}")
print()

if not all([access_key, secret_key, bucket_name, endpoint]):
    print("❌ Missing credentials")
    exit(1)

# Normalize endpoint
endpoint_clean = endpoint.replace("https://", "").replace("http://", "").strip()

try:
    auth = oss2.Auth(access_key, secret_key)
    bucket = oss2.Bucket(auth, endpoint_clean, bucket_name)
    
    # Try to get bucket info
    print("🔍 Testing connection...")
    bucket_info = bucket.get_bucket_info()
    print(f"✅ Connection successful!")
    print(f"   Location: {bucket_info.location}")
    print(f"   Creation date: {bucket_info.creation_date}")
    
    # Try a simple list operation
    print("\n🔍 Testing list operation...")
    result = bucket.list_objects(max_keys=1)
    print(f"✅ List operation successful! Found {len(result.object_list)} objects")
    
except oss2.exceptions.SignatureDoesNotMatch as e:
    print(f"❌ Signature error: {e}")
    print("\nPossible causes:")
    print("  1. Wrong Access Key ID or Secret")
    print("  2. Time sync issue (check system clock)")
    print("  3. Wrong endpoint format")
except oss2.exceptions.AccessDenied as e:
    print(f"❌ Access denied: {e}")
    print("\nPossible causes:")
    print("  1. Credentials don't have permission for this bucket")
    print("  2. Bucket doesn't exist in this region")
except oss2.exceptions.NoSuchBucket as e:
    print(f"❌ Bucket not found: {e}")
    print(f"\nBucket '{bucket_name}' doesn't exist in region '{endpoint_clean}'")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

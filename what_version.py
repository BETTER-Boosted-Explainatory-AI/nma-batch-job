# import os
# import tempfile
# import boto3
# import struct
# import json
# import zipfile
# import h5py
# from NMA.utilss.s3_utils import get_datasets_s3_client, get_users_s3_client

# # Get credentials from environment variables
# users_access_key = os.environ.get('AWS_USERS_ACCESS_KEY_ID')
# users_secret_key = os.environ.get('AWS_USERS_SECRET_ACCESS_KEY')
# bucket_name = os.environ.get('S3_USERS_BUCKET_NAME', 'better-xai-users')

# # S3 path details
# user_id = "92457494-3061-700c-8e14-f1ab249392d7"
# model_id = "496105c6-8406-47eb-b2ff-00f0fd532d38"
# model_key = f"{user_id}/{model_id}/resnet50_imagenet.keras"

# # Create temp file
# temp_file = tempfile.NamedTemporaryFile(delete=False)
# temp_file.close()
# print(f"Using custom credentials for bucket: {bucket_name}")
# print(f"Accessing model at: {model_key}")

# # Create S3 client with specific credentials
# s3_client = get_users_s3_client()

# try:
#     s3_client.download_file(bucket_name, model_key, temp_file.name)
    
#     # Try to extract TensorFlow version
#     tf_version = None
#     keras_version = None
    
#     # Check if it's a ZIP file (newer Keras format)
#     if zipfile.is_zipfile(temp_file.name):
#         print("Model is in ZIP format (newer Keras SavedModel)")
#         with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
#             # Look for metadata
#             if 'metadata.json' in zip_ref.namelist():
#                 with zip_ref.open('metadata.json') as f:
#                     metadata = json.load(f)
#                     print(f"Metadata: {json.dumps(metadata, indent=2)}")
#                     keras_version = metadata.get('keras_version', 'unknown')
                    
#             # Check for config.json
#             if 'config.json' in zip_ref.namelist():
#                 with zip_ref.open('config.json') as f:
#                     config = json.load(f)
#                     if 'metadata' in config:
#                         keras_version = config['metadata'].get('keras_version', 'unknown')
                        
#             # Check saved_model.pb for TF version
#             if 'saved_model.pb' in zip_ref.namelist():
#                 print("Found saved_model.pb - this is a TensorFlow SavedModel format")
                
#     # Check if it's HDF5 (older Keras format)
#     else:
#         try:
#             with h5py.File(temp_file.name, 'r') as f:
#                 print("Model is in HDF5 format (older Keras format)")
#                 # Check for keras_version attribute
#                 if 'keras_version' in f.attrs:
#                     keras_version = f.attrs['keras_version'].decode('utf-8')
                    
#                 # Check model_config
#                 if 'model_config' in f.attrs:
#                     model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
#                     if 'keras_version' in model_config:
#                         keras_version = model_config['keras_version']
                        
#         except:
#             print("Not a valid HDF5 file")
    
#     print(f"\nDetected Keras version: {keras_version}")
    
#     # Map Keras version to TensorFlow version
#     if keras_version:
#         if keras_version.startswith('2.4'):
#             tf_version = '2.4.x'
#         elif keras_version.startswith('2.6'):
#             tf_version = '2.6.x'
#         elif keras_version.startswith('2.8'):
#             tf_version = '2.8.x'
#         elif keras_version.startswith('2.9'):
#             tf_version = '2.9.x'
#         elif keras_version.startswith('2.10'):
#             tf_version = '2.10.x'
#         elif keras_version.startswith('2.11'):
#             tf_version = '2.11.x'
#         elif keras_version.startswith('2.12'):
#             tf_version = '2.12.x'
#         elif keras_version.startswith('2.13'):
#             tf_version = '2.13.x'
#         else:
#             print(f"Unknown Keras version mapping for: {keras_version}")
            
#     print(f"Recommended TensorFlow version: {tf_version}")
    
# except Exception as e:
#     print(f"Error: {str(e)}")
# finally:
#     if os.path.exists(temp_file.name):
#         os.unlink(temp_file.name)

import dotenv
dotenv.load_dotenv()

from pprint import pprint
from NMA.services.dataset_service import _get_dataset_config

cfg = _get_dataset_config("imagenet")            # pulls imagenet_info.py from S3
print("📦 keys in config:", list(cfg))

# ─────────────────────────────────────────────────────────────────────────────
# Folder-ID  →  readable-name mapping
mapping = cfg["directory_to_readable"]           # dict[str, str]

print(f"\nTotal synsets in mapping: {len(mapping):,}")
pprint(list(mapping.items())[:20])               # show first 20 pairs

# ─────────────────────────────────────────────────────────────────────────────
# All labels list (if you want to see it)
labels = cfg["directory_labels"]                 # list[str]
print(f"\nNumber of directory_labels: {len(labels):,}")
print("First 15 labels:", labels[:15])

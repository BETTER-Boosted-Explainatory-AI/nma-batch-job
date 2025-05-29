import os
import tempfile
import boto3
import struct
import json

# Get credentials from environment variables
users_access_key = os.environ.get('AWS_USERS_ACCESS_KEY_ID')
users_secret_key = os.environ.get('AWS_USERS_SECRET_ACCESS_KEY')
bucket_name = os.environ.get('S3_USERS_BUCKET_NAME', 'better-xai-users')

# S3 path details
user_id = "92457494-3061-700c-8e14-f1ab249392d7"
model_id = "496105c6-8406-47eb-b2ff-00f0fd532d38"
model_key = f"{user_id}/{model_id}/resnet50_imagenet.keras"

# Create temp file
temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.close()

try:
    print(f"Using custom credentials for bucket: {bucket_name}")
    print(f"Accessing model at: {model_key}")
    
    # Create S3 client with specific credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=users_access_key,
        aws_secret_access_key=users_secret_key
    )
    
    # Download the file
    print(f"Downloading model from S3...")
    s3_client.download_file(bucket_name, model_key, temp_file.name)
    
    # Get file size
    file_size = os.path.getsize(temp_file.name)
    print(f"Model downloaded successfully. File size: {file_size} bytes")
    
    # Examine file signature/magic bytes
    with open(temp_file.name, 'rb') as f:
        header = f.read(8)  # Read first 8 bytes
        
        # Convert bytes to hex for display
        header_hex = ' '.join(f'{b:02x}' for b in header)
        print(f"File header (hex): {header_hex}")
        
        # Try to interpret as various formats
        if header.startswith(b'\x89HDF'):
            print("This appears to be an HDF5 file (standard .h5 format)")
        elif header.startswith(b'PK\x03\x04'):
            print("This appears to be a ZIP file - likely a SavedModel in ZIP format")
            
            # For zip files, we can list contents
            import zipfile
            try:
                with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                    print("\nZIP file contents:")
                    for file_info in zip_ref.infolist():
                        print(f"- {file_info.filename} ({file_info.file_size} bytes)")
                    
                    # Look for config.json or metadata.json
                    if 'config.json' in [file.filename for file in zip_ref.infolist()]:
                        with zip_ref.open('config.json') as config_file:
                            config = json.load(config_file)
                            print("\nModel config:")
                            if 'keras_version' in config:
                                print(f"Keras version: {config['keras_version']}")
                            if 'model_architecture' in config:
                                print(f"Architecture: {config.get('model_architecture', 'unknown')}")
            except zipfile.BadZipFile:
                print("File has ZIP signature but is not a valid ZIP file")
        elif header.startswith(b'\x80\x03'):
            print("This appears to be a Python pickle file")
        else:
            print("Unknown file format")
            
        # If it's a text file (like JSON), try to read the beginning
        f.seek(0)
        try:
            start = f.read(100).decode('utf-8')
            if start.strip().startswith('{'):
                print("\nFile appears to be JSON. First 100 characters:")
                print(start)
        except UnicodeDecodeError:
            print("Not a text file")
            
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Clean up
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
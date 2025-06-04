import boto3
import os

def get_datasets_s3_client():
    """Get S3 client for datasets bucket"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_DATASETS_ACCESS_KEY_ID'),  # This is for datasets
        aws_secret_access_key=os.getenv('AWS_DATASETS_SECRET_ACCESS_KEY')  # This is for datasets
    )

def get_users_s3_client():
    """Get S3 client for users bucket"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_USERS_ACCESS_KEY_ID'),  # This is for users
        aws_secret_access_key=os.getenv('AWS_USERS_SECRET_ACCESS_KEY')  # This is for users
    )
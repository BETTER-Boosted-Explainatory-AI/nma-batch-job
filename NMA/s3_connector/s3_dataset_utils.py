import os
from typing import Dict, List, Tuple, Any
import numpy as np
from utilss.s3_connector.s3_dataset_loader import S3DatasetLoader
import boto3, io, pickle, logging

log = logging.getLogger(__name__)
_s3  = boto3.client("s3")  

def load_dataset_numpy(dataset_str: str, folder_type: str) -> Dict[str, np.ndarray]:

    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_numpy_data(dataset_str, folder_type)

def load_cifar100_adversarial_or_clean(folder_type: str) -> Dict[str, np.ndarray]:
    return load_dataset_numpy('cifar100', folder_type)

def load_imagenet_adversarial_or_clean(folder_type: str) -> Dict[str, np.ndarray]:
    return load_dataset_numpy('imagenet', folder_type)

def get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.get_dataset_info(dataset_str)

def load_dataset_folder(dataset_str: str, folder_type: str) -> List[str]:
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_folder(dataset_str, folder_type)

def load_single_image(image_key: str) -> bytes:
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_single_image(image_key)

def get_image_stream(image_key: str):
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.get_image_stream(image_key)

def load_imagenet_train() -> List[str]:

    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_imagenet_train()

def load_cifar100_as_numpy(folder_type: str) -> Tuple[np.ndarray, np.ndarray]:

    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_cifar100_numpy(folder_type)

def load_cifar100_meta() -> Dict:
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_cifar100_meta()

def load_dataset_split(dataset_str: str, split_type: str) -> List[str]:

    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_dataset_split(dataset_str, split_type)


def unpickle_from_s3(bucket: str, key: str):
    """
    Read a pickle file straight from S3 and return the loaded object.

    Args:
        bucket (str) : S3 bucket name  (e.g. "better-xai-users")
        key    (str) : Object key      (e.g. "cifar100/train")

    Returns:
        Any – whatever was pickled (dict for CIFAR-100)
    """
    obj = _s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()          # bytes
    log.debug("Fetched %s (%d bytes)", key, len(data))
    return pickle.load(io.BytesIO(data), encoding="bytes")

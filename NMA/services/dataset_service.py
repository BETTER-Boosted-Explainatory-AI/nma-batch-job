import os
from typing import Dict, Any, List, Tuple
import numpy as np
from NMA.classes.datasets.dataset_factory import DatasetFactory
from NMA.s3_connector.s3_dataset_loader import S3DatasetLoader

def _get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    """Get dataset configuration based on dataset string from S3."""
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.get_dataset_info(dataset_str)


def _load_dataset(dataset_str: str):
    dataset_config = _get_dataset_config(dataset_str)
    dataset_name = dataset_config["dataset"]
    dataset = DatasetFactory.create_dataset(dataset_name)
    print("dataset", dataset)
    return dataset



def get_dataset_labels(dataset_str: str) -> List[str]:
    dataset_config = _get_dataset_config(dataset_str)
    return dataset_config["labels"]

def _load_dataset_folder(dataset_str: str, folder_type: str):
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_folder(dataset_str, folder_type)


def load_single_image(image_key: str) -> bytes:
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_single_image(image_key)


def load_imagenet_train() -> str:
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_imagenet_train()


def load_cifar100_numpy(folder_type: str) -> Tuple[np.ndarray, np.ndarray]:
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_cifar100_numpy(folder_type)


def load_cifar100_meta() -> Dict:
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_cifar100_meta()


def _load_dataset_split(dataset_str: str, split_type: str) -> str:
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_dataset_split(dataset_str, split_type)
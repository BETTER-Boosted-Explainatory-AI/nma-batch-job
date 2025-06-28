import os
from typing import Dict, Any, List, Tuple
import numpy as np
from NMA.data.datasets.cifar100_info import CIFAR100_INFO
from NMA.data.datasets.imagenet_info import IMAGENET_INFO
from NMA.utilss.enums.datasets_enum import DatasetsEnum
from NMA.classes.datasets.dataset_factory import DatasetFactory

def _get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    """Get dataset configuration based on dataset string."""
    print(dataset_str)
    if dataset_str == DatasetsEnum.CIFAR100.value:
        return CIFAR100_INFO
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        return IMAGENET_INFO
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")

def _load_dataset(dataset_config: Dict[str, Any]):
    """Load the dataset based on configuration."""
    dataset = DatasetFactory.create_dataset(dataset_config["dataset"])
    dataset.load(dataset_config["dataset"])
    
    return dataset

def _get_dataset_labels(dataset_str: str) -> List[str]:
    """Get dataset labels based on dataset string."""
    if dataset_str == DatasetsEnum.CIFAR100.value:
        return CIFAR100_INFO["labels"]
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        return IMAGENET_INFO["labels"]
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")

# def _load_dataset_folder(dataset_str: str, folder_type: str):
#     bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
#     if not bucket_name:
#         raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
#     s3_loader = S3DatasetLoader(bucket_name=bucket_name)
#     return s3_loader.load_folder(dataset_str, folder_type)


# def load_single_image(image_key: str) -> bytes:
#     bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
#     if not bucket_name:
#         raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
#     s3_loader = S3DatasetLoader(bucket_name=bucket_name)
#     return s3_loader.load_single_image(image_key)


# def load_imagenet_train() -> str:
#     bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
#     if not bucket_name:
#         raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
#     s3_loader = S3DatasetLoader(bucket_name=bucket_name)
#     return s3_loader.load_imagenet_train()


# def load_cifar100_numpy(folder_type: str) -> Tuple[np.ndarray, np.ndarray]:
#     bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
#     if not bucket_name:
#         raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
#     s3_loader = S3DatasetLoader(bucket_name=bucket_name)
#     return s3_loader.load_cifar100_numpy(folder_type)


# def load_cifar100_meta() -> Dict:
#     bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
#     if not bucket_name:
#         raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
#     s3_loader = S3DatasetLoader(bucket_name=bucket_name)
#     return s3_loader.load_cifar100_meta()


# def _load_dataset_split(dataset_str: str, split_type: str) -> str:
#     bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
#     if not bucket_name:
#         raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
#     s3_loader = S3DatasetLoader(bucket_name=bucket_name)
#     return s3_loader.load_dataset_split(dataset_str, split_type)
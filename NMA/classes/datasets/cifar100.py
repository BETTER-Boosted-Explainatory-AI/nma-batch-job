
from __future__ import annotations
import os
import pickle
import logging
from typing import Tuple, List
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from NMA.classes.datasets.dataset import Dataset  
from NMA.s3_connector.s3_dataset_utils import unpickle_from_s3 

class Cifar100(Dataset):

    def __init__(self) -> None:
        from NMA.services.dataset_service import _get_dataset_config  

        cfg = _get_dataset_config("cifar100")
        super().__init__(
            cfg["dataset"], cfg["threshold"], cfg["infinity"], cfg["labels"]
        )

        self.log = logging.getLogger(__name__)
        self.x_train: np.ndarray | None = None
        self.y_train: List[str] | None = None
        self.x_test: np.ndarray | None = None
        self.y_test: List[str] | None = None

    def _map_y_labels(self, y: np.ndarray) -> List[str]:
        return [self.label_to_class_name(idx) for idx in y]

    def load(self, name: str = "cifar100") -> bool:  # retained for backward compat
        bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
        if not bucket:
            raise RuntimeError("S3_DATASETS_BUCKET_NAME env‑var must be set")

        train = unpickle_from_s3(bucket, "cifar100/train")
        test = unpickle_from_s3(bucket, "cifar100/test")

        self._process(train, test)
        self.log.info("Loaded CIFAR‑100 from S3 (%d train, %d test)", len(self.x_train), len(self.x_test))
        return True

    def load_from_s3(self, s3_client, bucket: str, prefix: str):
        
        prefix = prefix.rstrip("/")

        if prefix.endswith("train"):
            train_key = prefix
            test_key = prefix[:-5] + "test"  # replace trailing "train" with "test"
        else:
            train_key = f"{prefix}/train"
            test_key = f"{prefix}/test"

        self.log.info("Resolved S3 keys: train=%s  test=%s", train_key, test_key)

        def _unpickle(key: str):
            self.log.debug("Fetching %s …", key)
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            return pickle.load(resp["Body"], encoding="bytes")

        try:
            train = _unpickle(train_key)
            test = _unpickle(test_key)
        except s3_client.exceptions.NoSuchKey as e:
            fallback_train = "cifar100/train"
            self.log.warning("%s missing – falling back to %s", train_key, fallback_train)
            train = _unpickle(fallback_train)
            test = _unpickle("cifar100/test")
        except Exception:
            raise 

        self._process(train, test)
        self.log.info("Loaded CIFAR‑100 (%d train, %d test) from %s", len(self.x_train), len(self.x_test), bucket)
        return self.x_train, self.y_train

    def _process(self, train_pkl: dict, test_pkl: dict) -> None:
        """Common routine used by both load() and load_from_s3()."""
        x_train = train_pkl[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        x_test = test_pkl[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        self.x_train = x_train          
        self.x_test  = x_test
        self.y_train = self._map_y_labels(np.array(train_pkl[b"fine_labels"]))
        self.y_test = self._map_y_labels(np.array(test_pkl[b"fine_labels"]))
        
    def label_to_class_name(self, idx: int) -> str:  
        return self.labels[idx]

    def get_train_image_by_id(self, image_id: int):
        if image_id >= len(self.x_train):
            raise ValueError("Invalid image_id")
        return self.x_train[image_id], self.y_train[image_id]

    def get_test_image_by_id(self, image_id: int):
        if image_id >= len(self.x_test):
            raise ValueError("Invalid image_id")
        return self.x_test[image_id], self.y_test[image_id]

    def get_label_readable_name(self, label):
        return label  

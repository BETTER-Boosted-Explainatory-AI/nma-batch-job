
# import os
# import pickle
# import numpy as np
# from .dataset import Dataset
# import matplotlib.pyplot as plt
# import pickle, logging

# from NMA.classes.datasets.dataset import Dataset 
# from NMA.s3_connector.s3_dataset_utils import unpickle_from_s3


# class Cifar100(Dataset):
#     def __init__(self):
#         from NMA.services.dataset_service import _get_dataset_config
#         super().__init__(_get_dataset_config("cifar100")["dataset"], _get_dataset_config("cifar100")["threshold"], _get_dataset_config("cifar100")["infinity"], _get_dataset_config("cifar100")["labels"])
#         self.x_train = None
#         self.y_train = None
#         self.x_test = None
#         self.y_test = None
        
#     def unpickle(self, file):
#         with open(file, 'rb') as fo:
#             data_dict = pickle.load(fo, encoding='bytes') 
#         return data_dict
    
# class Cifar100(Dataset):
#     def __init__(self):
#         from NMA.services.dataset_service import _get_dataset_config

#         cfg = _get_dataset_config("cifar100")
#         super().__init__(cfg["dataset"], cfg["threshold"], cfg["infinity"], cfg["labels"])
#         self.log = logging.getLogger(__name__)

#     def load(self, name):                         
#         bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
#         if not bucket:
#             raise RuntimeError("S3_DATASETS_BUCKET_NAME env-var must be set")

#         # ① stream both pickles
#         train = unpickle_from_s3(bucket, "cifar100/train")
#         test  = unpickle_from_s3(bucket, "cifar100/test")

#         # ② reshape exactly like before
#         self.x_train = train[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1)
#         self.y_train = np.array(train[b"fine_labels"])

#         self.x_test  = test[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1)
#         self.y_test  = np.array(test[b"fine_labels"])

#         self.y_train = self._map_y_labels(self.y_train)
#         self.y_test  = self._map_y_labels(self.y_test)

#         self.log.info("Loaded CIFAR-100 from S3 (train %s, test %s)",
#                       len(self.x_train), len(self.x_test))
#         return True

    
#     def _map_y_labels(self, y_train):
#         return [self.label_to_class_name(label) for label in y_train]
    
#     def one_hot_to_class_name_auto(self, one_hot_vector):
#         return self.labels[np.argmax(one_hot_vector)] 

#     def label_to_class_name(self, label):
#         return self.labels[label]

#     def get_train_image_by_id(self, image_id):
#         # Check if the image_id is within the range of training data
#         if image_id < len(self.x_train):
#             image = self.x_train[image_id]
#             label = self.y_train[image_id]
#             print(f"Train image ID {image_id}: label {label}") 
#         else:
#             raise ValueError("Invalid image_id")

#         return image, label
    
#     def get_test_image_by_id(self, image_id):
#         if image_id < len(self.x_test):
#             image = self.x_test[image_id]
#             label = self.y_test[image_id]
#             print(f"Test image ID {image_id}: label {label}")
#         else:
#             raise ValueError("Invalid image_id")

#         return image, label
        
#     def get_label_readable_name(self, label):
#         return label
    
    
#     def load_from_s3(self, s3_client, bucket, prefix):
#         """
#         Load CIFAR-100 dataset from S3 directly using the provided S3 client.
#         This method is specifically required by the NMA class.
        
#         Args:
#             s3_client: boto3 S3 client
#             bucket: S3 bucket name
#             prefix: Path prefix in the bucket (e.g. 'cifar100/' or 'datasets/cifar100/')
        
#         Returns:
#             Tuple of (x_train, y_train)
#         """
#         import io
#         import pickle
#         import numpy as np
        
#         self.log.info(f"Loading CIFAR-100 from S3: {bucket}/{prefix}")
        
#         # Normalize the prefix to ensure it has the right format
#         if not prefix.endswith('/'):
#             prefix = prefix + '/'
        
#         # Determine the train file path based on the prefix
#         if prefix.endswith('cifar100/') or prefix.endswith('CIFAR100/'):
#             train_key = f"{prefix}train"
#         elif prefix.endswith('train/'):
#             train_key = prefix.rstrip('/')
#         else:
#             # Try adding train to the prefix
#             train_key = f"{prefix}train"
        
#         self.log.info(f"Looking for train data at: {train_key}")
        
#         # Function to load pickle from S3 using the provided client
#         def unpickle_with_client(key):
#             try:
#                 response = s3_client.get_object(Bucket=bucket, Key=key)
#                 return pickle.load(response['Body'], encoding='bytes')
#             except Exception as e:
#                 self.log.error(f"Error loading data from {key}: {str(e)}")
#                 raise
        
#         # Load the train data
#         try:
#             train_data = unpickle_with_client(train_key)
#             self.log.info(f"Successfully loaded train data from {train_key}")
#         except Exception as e:
#             # Try alternative path
#             alternative_key = f"cifar100/train"
#             self.log.info(f"Trying alternative path: {alternative_key}")
#             try:
#                 train_data = unpickle_with_client(alternative_key)
#                 self.log.info(f"Successfully loaded train data from {alternative_key}")
#             except Exception as e2:
#                 self.log.error(f"Error loading train data from {alternative_key}: {str(e2)}")
#                 raise ValueError(f"Could not load CIFAR-100 train data from S3: {str(e)} | {str(e2)}")
        
#         # Process the train data - using the same approach as your existing load method
#         try:
#             # CIFAR-100 pickle format: dict with keys b'data', b'fine_labels', etc.
#             x_train = train_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#             y_train_raw = np.array(train_data[b'fine_labels'])
            
#             # Map labels to class names using your existing method
#             y_train = self._map_y_labels(y_train_raw)
            
#             self.log.info(f"Processed train data: {x_train.shape} images, {len(y_train)} labels")
            
#             # Store in instance variables for future use
#             self.x_train = x_train
#             self.y_train = y_train
            
#             # For completeness, try to load the test data as well
#             # This isn't required by NMA but keeps your class consistent
#             try:
#                 test_key = train_key.replace('train', 'test')
#                 test_data = unpickle_with_client(test_key)
                
#                 self.x_test = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#                 self.y_test = np.array(test_data[b'fine_labels'])
#                 self.y_test = self._map_y_labels(self.y_test)
                
#                 self.log.info(f"Also loaded test data: {self.x_test.shape} images, {len(self.y_test)} labels")
#             except Exception as e:
#                 self.log.warning(f"Could not load test data: {str(e)}")
#                 # This is not critical for NMA, so we can continue
            
#             return x_train, y_train
        
#         except Exception as e:
#             self.log.error(f"Error processing CIFAR-100 data: {str(e)}")
#             raise ValueError(f"Error processing CIFAR-100 data: {str(e)}")


# cifar100_dataset.py – fully corrected with 224×224 preprocessing
# --------------------------------------------------------------
# Drop‑in replacement for your existing NMA.classes.datasets.cifar100
# module.  It automatically resizes 32×32 CIFAR‑100 images to
# 224×224 and applies the same preprocessing Keras' ResNet50
# expects, so the downstream model no longer raises shape errors.
#
# 2025‑06‑06 — v2
#   • load_from_s3(): robust prefix handling so it works whether the
#     caller passes "cifar100/", "cifar100", "cifar100/train", etc.
#   • If the first key is missing, fall back to a small set of
#     alternatives before raising NoSuchKey.

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

# ################################## FOR TESTING ##################################

#         y_train_raw = np.array(train_pkl[b"fine_labels"])          # <-- add this

#         # --- tiny-set filter (keep first 5 per class) -----------------------
#         SAMPLES = 1
#         if SAMPLES is not None:
#             idx = np.hstack([np.where(y_train_raw == c)[0][:SAMPLES]   # 0…99 classes
#                             for c in range(100)])
#             x_train     = x_train[idx]
#             y_train_raw = y_train_raw[idx]
#         # -------------------------------------------------------------------


#         self.x_train = x_train          
#         self.x_test  = x_test
#         self.y_train = self._map_y_labels(y_train_raw)
#         self.y_test = self._map_y_labels(np.array(test_pkl[b"fine_labels"]))

# ################################################################################

## TODO: uncomment this for NOT testsing 

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

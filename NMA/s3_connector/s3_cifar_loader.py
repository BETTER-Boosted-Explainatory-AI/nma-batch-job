import os
from typing import Dict, Tuple, List, Any
import numpy as np
from NMA.s3_connector.s3_handler import S3Handler

class S3CifarLoader:
    def __init__(self, s3_handler=None, bucket_name=None):
        self.bucket_name = bucket_name or os.environ.get('S3_DATASETS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name not specified")
        self.s3_handler = s3_handler or S3Handler(bucket_name=self.bucket_name)
    
    def list_cifar100_files(self, folder_type: str) -> List[str]:
        return self.s3_handler.get_folder_contents('cifar100', folder_type)
    
    def load_cifar100_as_numpy(self, folder_type: str) -> Tuple[np.ndarray, np.ndarray]:
        if folder_type not in ['adversarial', 'clean', 'test', 'train']:
            raise ValueError(f"Invalid folder type for CIFAR-100 numpy: {folder_type}")
        return self.s3_handler.get_cifar100_as_numpy(folder_type)
    
    def load_cifar100_numpy_files(self, folder_type: str) -> Dict[str, np.ndarray]:
        if folder_type not in ['clean', 'adversarial']:
            raise ValueError(f"Invalid folder type for numpy data: {folder_type}")
        return self.s3_handler.get_numpy_data('cifar100', folder_type)
    
    def load_cifar100_meta(self) -> Dict:
        return self.s3_handler.get_cifar100_meta()
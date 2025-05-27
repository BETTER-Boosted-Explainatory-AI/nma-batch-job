import os
from typing import Dict, List, Any
import numpy as np
from utilss.s3_connector.s3_handler import S3Handler

class S3ImagenetLoader:
    def __init__(self, s3_handler=None, bucket_name=None):
        """Initialize with an S3 handler."""
        self.bucket_name = bucket_name or os.environ.get('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name not specified")
            
        self.s3_handler = s3_handler or S3Handler(bucket_name=self.bucket_name)
    
    def load_imagenet_folder(self, folder_type: str) -> List[str]:
        """List all files in an ImageNet folder"""
        return self.s3_handler.get_folder_contents('imagenet', folder_type)
    
    def list_test_images(self) -> List[str]:
        """Return S3 keys for every test-set image."""
        return self.s3_handler.get_folder_contents('imagenet', 'test')
    
    def get_imagenet_classes(self) -> List[str]:
        """Get all ImageNet class directories from train folder"""
        return self.s3_handler.get_imagenet_classes()
    
    def get_class_images(self, class_dir: str) -> List[str]:
        """Get all images for a specific ImageNet class"""
        return self.s3_handler.get_imagenet_class_data(class_dir)
    
    def get_image_data(self, image_key: str) -> bytes:
        """Get image data for a specific image"""
        return self.s3_handler.get_object_data(image_key)
    
    def get_image_stream(self, image_key: str):
        """Get image as a stream for processing without downloading"""
        return self.s3_handler.get_object_stream(image_key)
    
    def load_imagenet_numpy_files(self, folder_type: str) -> Dict[str, np.ndarray]:
        """Load NumPy (.npy) files from ImageNet clean/adversarial folders"""
        if folder_type not in ['clean', 'adversarial']:
            raise ValueError(f"Invalid folder type for numpy data: {folder_type}")
            
        return self.s3_handler.get_numpy_data('imagenet', folder_type)
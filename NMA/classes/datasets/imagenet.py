import os
import time
import io
from PIL import Image
from .dataset import Dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
# from data.datasets.imagenet_info import IMAGENET_INFO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from NMA.s3_connector.s3_dataset_loader import S3DatasetLoader
from NMA.s3_connector.s3_imagenet_loader import S3ImagenetLoader
from NMA.utilss.s3_utils import get_datasets_s3_client

import logging
logger = logging.getLogger(__name__)

class ImageNet(Dataset):
    def __init__(self):
        from NMA.services.dataset_service import _get_dataset_config
        config = _get_dataset_config("imagenet")

        super().__init__(config["dataset"], config["threshold"], config["infinity"], config["directory_labels"])
        self.x_train = None
        self.y_train = None
        self.directory_labels = config["directory_labels"]
        self.s3_loader = S3ImagenetLoader()
        

    def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
        """
        Returns:
        --------
        images : np.array
            Array of preprocessed images with shape (n_samples, height, width, channels)
        labels : np.array
            Array of labels as folder names (e.g., 'n01440764')
        """
       
        norm_path = dataset_path.replace("\\", "/")
        if norm_path.startswith('data/datasets/'):
            split = os.path.basename(dataset_path)  # This will give 'train' or 'test'
        else:
            split = os.path.basename(dataset_path)
        print(f"Loading mini ImageNet for split: {split} from {dataset_path}")

        if split == 'train':
            class_names = self.s3_loader.get_imagenet_classes()
        else:
            print(f"Test split loading from S3 not yet implemented")
            return np.array([]), np.array([])
        
        if not class_names:
            raise ValueError(f"No subdirectories found in {dataset_path}")
        
        class_names = [d for d in class_names if d.startswith('n') and len(d) > 1 and d[1:].replace('_', '').isdigit()]
        
        images = []
        labels = []
        
        print(f"Loading images from {len(class_names)} classes...")
        
        def load_class_images(class_name):
            class_images = []
            class_labels = []
            
            try:
                image_keys = self.s3_loader.get_class_images(class_name)
                img_files = [k for k in image_keys if k.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for image_key in img_files:
                    try:
                        image_data = self.s3_loader.get_image_data(image_key)
                        if image_data:
                            img = Image.open(io.BytesIO(image_data))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img = img.resize(img_size, Image.Resampling.LANCZOS)
                            img_array = np.array(img, dtype=np.float32)
                            class_images.append(img_array)
                            class_labels.append(class_name)  # Use folder name directly as label
                    except Exception as e:
                        print(f"Error loading {image_key}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error loading class {class_name}: {e}")
            return class_images, class_labels
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(load_class_images, class_name): class_name for class_name in class_names}
            for future in as_completed(futures):
                class_name = futures[future]
                try:
                    class_images, class_labels = future.result()
                    if class_images:
                        images.extend(class_images)
                        labels.extend(class_labels)
                except Exception as e:
                    print(f"Error processing class {class_name}: {e}")
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    
    def load(self, name):
        from NMA.services.dataset_service import _get_dataset_config
        config = _get_dataset_config("imagenet")
        bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
        if not bucket:
            raise RuntimeError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        s3_client = get_datasets_s3_client()
        train_prefix = f"{name}/train"
        self.x_train, self.y_train = self.load_from_s3(s3_client, bucket, train_prefix)
        self.directory_labels = config["directory_labels"]
        print(f"Loaded {len(self.x_train)} train images")


    def get_train_image_by_id(self, image_id):
        if image_id < len(self.x_train):
            image = self.x_train[image_id]
            label = self.y_train[image_id]
            print(f"Train image ID {image_id}: label {label}") 
        else:
            raise ValueError("Invalid image_id")

        return image, label
    
    def get_test_image_by_id(self, image_id):
        if image_id < len(self.x_test):
            image = self.x_test[image_id]
            label = self.y_test[image_id]
            print(f"Test image ID {image_id}: label {label}")
        else:
            raise ValueError("Invalid image_id")

        return image, label


    def directory_to_labels_conversion(self, label):
        from NMA.services.dataset_service import _get_dataset_config
        dir_to_readable = _get_dataset_config("imagenet")["directory_to_readable"]
        return dir_to_readable[label]
    
    def get_label_readable_name(self, label):
        return self.directory_to_labels_conversion(label)
    
        
    def load_from_s3(self, s3_client, bucket, prefix):
        logger.info(f"Loading ImageNet from S3: {bucket}/{prefix}")
        from NMA.services.dataset_service import _get_dataset_config
        config = _get_dataset_config("imagenet")
        
        if not hasattr(self, 'directory_labels') or not self.directory_labels:
            self.directory_labels = config["directory_labels"]
        
        if not prefix.endswith('/'):
            prefix = prefix + '/'
        
        logger.info(f"Using prefix: {prefix}")
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter='/'
        )
        
        class_folders = []
        if 'CommonPrefixes' in response:
            class_folders = [p['Prefix'] for p in response['CommonPrefixes']]
        
        if not class_folders:
            logger.error(f"No class folders found at {prefix} in bucket {bucket}")
            raise ValueError(f"No class folders found at {prefix} in bucket {bucket}")
        
        logger.info(f"Found {len(class_folders)} class folders")
        x_train = []
        y_train = []
        
        max_classes = 1000 
        max_images_per_class = 10 
        
        processed_classes = 0
        for folder in class_folders:
            if processed_classes >= max_classes:
                break
            folder_name = folder.rstrip('/').split('/')[-1]
            if not folder_name.startswith('n') or len(folder_name) < 5:
                continue
            if folder_name not in self.directory_labels:
                logger.debug(f"Skipping folder {folder_name} - not in directory_labels")
                continue
            
            logger.info(f"Processing class folder: {folder_name}")
            try:
                images_response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=folder,
                    MaxKeys=max_images_per_class
                )
                
                if 'Contents' not in images_response:
                    logger.warning(f"No images found in {folder}")
                    continue
                
                images_processed = 0
                for item in images_response['Contents']:
                    if images_processed >= max_images_per_class:
                        break
                        
                    if not item['Key'].lower().endswith(('.jpeg', '.jpg', '.png')):
                        continue
                    
                    # Download and process image
                    try:
                        img_response = s3_client.get_object(Bucket=bucket, Key=item['Key'])
                        img_data = img_response['Body'].read()
                        
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(img_data))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img = img.resize((224, 224)) 
                        img_array = np.array(img) 
                        
                        # Ensure correct shape
                        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                            x_train.append(img_array)
                            y_train.append(folder_name)  # Use folder name (e.g., 'n01440764')
                            images_processed += 1
                            
                    except Exception as e:
                        logger.warning(f"Error processing image {item['Key']}: {str(e)}")
                        continue
                
                if images_processed > 0:
                    logger.info(f"Loaded {images_processed} images from class {folder_name}")
                    processed_classes += 1
                    
            except Exception as e:
                logger.warning(f"Error listing images in {folder}: {str(e)}")
                continue
        
        if not x_train:
            logger.error("No images were processed successfully")
            raise ValueError("Failed to load any images from S3")
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        self.x_train = x_train
        self.y_train = y_train
        
        logger.info(f"Successfully loaded ImageNet: {x_train.shape} images, {len(y_train)} labels")
        logger.info(f"Processed {processed_classes} classes")
        logger.info(f"Sample labels: {y_train[:5]}")
        
        return x_train, y_train
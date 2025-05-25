import os
import time
import io
from PIL import Image
from .dataset import Dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
# from data.datasets.imagenet_info import IMAGENET_INFO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utilss.s3_connector.s3_dataset_loader import S3DatasetLoader
from utilss.s3_connector.s3_imagenet_loader import S3ImagenetLoader


class ImageNet(Dataset):
    def __init__(self):
        from services.dataset_service import _get_dataset_config
        super().__init__(_get_dataset_config("imagenet")["dataset"], _get_dataset_config("imagenet")["threshold"], _get_dataset_config("imagenet")["infinity"], _get_dataset_config("imagenet")["labels"])
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.directory_labels = None
        
    # def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
    #     """
    #     Load mini ImageNet dataset from a directory structure where:
    #     - Each subdirectory name in dataset_path is a class label
    #     - Each subdirectory contains images of that class
        
    #     Parameters:
    #     -----------
    #     dataset_path : str
    #         Path to the dataset directory (e.g., 'imagenet/train')
    #     img_size : tuple
    #         Size to resize the images to (height, width)
            
    #     Returns:
    #     --------
    #     images : np.array
    #         Array of preprocessed images with shape (n_samples, height, width, channels)
    #     labels : np.array
    #         Array of labels as folder names (e.g., 'n01440764')
    #     """
    #     if not os.path.exists(dataset_path):
    #         raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
    #     # Get all class names (subdirectories)
    #     class_names = sorted([
    #         d for d in os.listdir(dataset_path)
    #         if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('n') and d[1:].isdigit()
    #     ])
        
    #     if not class_names:
    #         raise ValueError(f"No subdirectories found in {dataset_path}")

    #     # Prepare lists to hold images and labels
    #     images = []
    #     labels = []
        
    #     # Load images from each class
    #     print(f"Loading images from {len(class_names)} classes...")
    #     for class_name in class_names:
    #         class_path = os.path.join(dataset_path, class_name)
    #         # Get all image files in the class directory
    #         img_files = [f for f in os.listdir(class_path) 
    #                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
    #         for img_file in img_files:
    #             img_path = os.path.join(class_path, img_file)
    #             try:
    #                 # Load and preprocess image
    #                 img = load_img(img_path, target_size=img_size)
    #                 img_array = img_to_array(img)
                    
    #                 # Properly preprocess for ResNet50
    #                 # img_array = preprocess_input(img_array)
                    
    #                 # Add to our collections
    #                 images.append(img_array)
    #                 labels.append(class_name)  # Use folder name directly as label

    #             except Exception as e:
    #                 print(f"Error loading {img_path}: {e}")
        
    #     # Convert lists to numpy arrays
    #     images = np.array(images)
    #     labels = np.array(labels)

    #     return images, labels

    def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
        """
        Returns:
        --------
        images : np.array
            Array of preprocessed images with shape (n_samples, height, width, channels)
        labels : np.array
            Array of labels as folder names (e.g., 'n01440764')
        """
        # Initialize S3 loader if not already done
        if self.s3_loader is None:
            self.s3_loader = S3ImagenetLoader()
        
        # Extract split type from dataset_path (e.g., 'imagenet/train' -> 'train')
        # Handle both local-style paths and S3-style paths
        if dataset_path.startswith('data/datasets/'):
            # Local-style path: 'data/datasets/imagenet/train'
            parts = dataset_path.split('/')
            split = parts[-1]  # 'train' or 'test'
        else:
            # S3-style path: 'imagenet/train'
            split = dataset_path.split('/')[-1]
        
        # Get all class names from S3
        if split == 'train':
            class_names = self.s3_loader.get_imagenet_classes()
        else:
            # For test split, we need to handle differently
            # For now, return empty arrays as in original code
            print(f"Test split loading from S3 not yet implemented")
            return np.array([]), np.array([])
        
        if not class_names:
            raise ValueError(f"No subdirectories found in {dataset_path}")
        
        # Filter to only valid ImageNet synsets (starting with 'n')
        class_names = [d for d in class_names if d.startswith('n') and len(d) > 1 and d[1:].replace('_', '').isdigit()]
        class_names = sorted(class_names)
        
        # Prepare lists to hold images and labels
        images = []
        labels = []
        
        # Load images from each class
        print(f"Loading images from {len(class_names)} classes...")
        
        # Use ThreadPoolExecutor for parallel loading
        def load_class_images(class_name):
            class_images = []
            class_labels = []
            
            try:
                # Get all images for this class
                image_keys = self.s3_loader.get_class_images(class_name)
                
                # Filter only image files
                img_files = [k for k in image_keys if k.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for image_key in img_files:
                    try:
                        # Get image data from S3
                        image_data = self.s3_loader.get_image_data(image_key)
                        if image_data:
                            # Load image from bytes using PIL
                            img = Image.open(io.BytesIO(image_data))
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            # Resize
                            img = img.resize(img_size, Image.Resampling.LANCZOS)
                            # Convert to array (matching keras img_to_array behavior)
                            img_array = np.array(img, dtype=np.float32)
                            
                            # Add to our collections
                            class_images.append(img_array)
                            class_labels.append(class_name)  # Use folder name directly as label
                    except Exception as e:
                        print(f"Error loading {image_key}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error loading class {class_name}: {e}")
                
            return class_images, class_labels
        
        # Load images with some parallelism for better performance
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(load_class_images, class_name): class_name 
                      for class_name in class_names}
            
            for future in as_completed(futures):
                class_name = futures[future]
                try:
                    class_images, class_labels = future.result()
                    if class_images:
                        images.extend(class_images)
                        labels.extend(class_labels)
                except Exception as e:
                    print(f"Error processing class {class_name}: {e}")
        
        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels

    # def load(self, name):
    #     from services.dataset_service import _get_dataset_config

    #     dataset_path = os.path.join("data", "datasets", name)
    #     train_path = os.path.join(dataset_path, "train")
    #     test_path = os.path.join(dataset_path, "test")

    #     self.x_train, self.y_train = self.load_mini_imagenet(train_path)
    # #    self.x_test, self.y_test = self.load_mini_imagenet(test_path)
    
    #     self.directory_labels = _get_dataset_config("imagenet")["directory_labels"]
    #     print("loaded imagenet dataset")
    
    
    def load(self, name):
        from services.dataset_service import _get_dataset_config
        
        # Check if S3 bucket is configured
        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            raise RuntimeError("S3_BUCKET_NAME environment variable must be set")
        
        # Construct paths that mimic the local structure but for S3
        # Original: data/datasets/imagenet/train
        # S3: imagenet/train
        dataset_path = os.path.join("data", "datasets", name) 
        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")

        # Load using the same function but it will now fetch from S3
        self.x_train, self.y_train = self.load_mini_imagenet(train_path)
        # self.x_test, self.y_test = self.load_mini_imagenet(test_path)
        
        self.directory_labels = _get_dataset_config("imagenet")["directory_labels"]
        print("loaded imagenet dataset")


    # def get_train_image_by_id(self, image_id):
    #     # Implement the logic to get an image by its ID from the ImageNet dataset
        
    #     print(f"Getting image with ID: {image_id}")
    #     data = self.x_train
    #     batch_index = image_id // len(data[0][0])
    #     image_index = image_id % len(data[0][0])
    #     image = data[batch_index][0][image_index]
    #     label = data[batch_index][1][image_index]
    #     return image, label
    
    # def get_test_image_by_id(self, image_id):
    #     # Implement the logic to get an image by its ID from the ImageNet dataset
    #     print(f"Getting image with ID: {image_id}")
    #     data = self.x_test
    #     batch_index = image_id // len(data[0][0])
    #     image_index = image_id % len(data[0][0])
    #     image = data[batch_index][0][image_index]
    #     label = data[batch_index][1][image_index]
    #     return image, label

    def get_train_image_by_id(self, image_id):
        # Check if the image_id is within the range of training data
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
        from services.dataset_service import _get_dataset_config
        dir_to_readable = _get_dataset_config("imagenet")["directory_to_readable"]
        return dir_to_readable[label]
    
    def get_label_readable_name(self, label):
        return self.directory_to_labels_conversion(label)
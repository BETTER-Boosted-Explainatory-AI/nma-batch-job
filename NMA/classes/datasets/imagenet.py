# # import os
# # import time
# # import io
# # from PIL import Image
# # from .dataset import Dataset
# # import numpy as np
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # # from data.datasets.imagenet_info import IMAGENET_INFO
# # from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from NMA.s3_connector.s3_dataset_loader import S3DatasetLoader



# # class ImageNet(Dataset):
# #     def __init__(self):
# #         from dataset_service import _get_dataset_config
# #         super().__init__(_get_dataset_config("imagenet")["dataset"], _get_dataset_config("imagenet")["threshold"], _get_dataset_config("imagenet")["infinity"], _get_dataset_config("imagenet")["labels"])
# #         self.x_train = None
# #         self.y_train = None
# #         self.x_test = None
# #         self.y_test = None
# #         self.directory_labels = None
        
# #     # def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
# #     #     """
# #     #     Load mini ImageNet dataset from a directory structure where:
# #     #     - Each subdirectory name in dataset_path is a class label
# #     #     - Each subdirectory contains images of that class
        
# #     #     Parameters:
# #     #     -----------
# #     #     dataset_path : str
# #     #         Path to the dataset directory (e.g., 'imagenet/train')
# #     #     img_size : tuple
# #     #         Size to resize the images to (height, width)
            
# #     #     Returns:
# #     #     --------
# #     #     images : np.array
# #     #         Array of preprocessed images with shape (n_samples, height, width, channels)
# #     #     labels : np.array
# #     #         Array of labels as folder names (e.g., 'n01440764')
# #     #     """
# #     #     if not os.path.exists(dataset_path):
# #     #         raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
# #     #     # Get all class names (subdirectories)
# #     #     class_names = sorted([
# #     #         d for d in os.listdir(dataset_path)
# #     #         if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('n') and d[1:].isdigit()
# #     #     ])
        
# #     #     if not class_names:
# #     #         raise ValueError(f"No subdirectories found in {dataset_path}")

# #     #     # Prepare lists to hold images and labels
# #     #     images = []
# #     #     labels = []
        
# #     #     # Load images from each class
# #     #     print(f"Loading images from {len(class_names)} classes...")
# #     #     for class_name in class_names:
# #     #         class_path = os.path.join(dataset_path, class_name)
# #     #         # Get all image files in the class directory
# #     #         img_files = [f for f in os.listdir(class_path) 
# #     #                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
# #     #         for img_file in img_files:
# #     #             img_path = os.path.join(class_path, img_file)
# #     #             try:
# #     #                 # Load and preprocess image
# #     #                 img = load_img(img_path, target_size=img_size)
# #     #                 img_array = img_to_array(img)
                    
# #     #                 # Properly preprocess for ResNet50
# #     #                 # img_array = preprocess_input(img_array)
                    
# #     #                 # Add to our collections
# #     #                 images.append(img_array)
# #     #                 labels.append(class_name)  # Use folder name directly as label

# #     #             except Exception as e:
# #     #                 print(f"Error loading {img_path}: {e}")
        
# #     #     # Convert lists to numpy arrays
# #     #     images = np.array(images)
# #     #     labels = np.array(labels)

# #     #     return images, labels

# #     def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
# #         """
# #         Returns:
# #         --------
# #         images : np.array
# #             Array of preprocessed images with shape (n_samples, height, width, channels)
# #         labels : np.array
# #             Array of labels as folder names (e.g., 'n01440764')
# #         """
# #         # Initialize S3 loader if not already done
# #         if self.s3_loader is None:
# #             self.s3_loader = S3ImagenetLoader()
        
# #         # Extract split type from dataset_path (e.g., 'imagenet/train' -> 'train')
# #         # Handle both local-style paths and S3-style paths
# #         if dataset_path.startswith('data/datasets/'):
# #             # Local-style path: 'data/datasets/imagenet/train'
# #             parts = dataset_path.split('/')
# #             split = parts[-1]  # 'train' or 'test'
# #         else:
# #             # S3-style path: 'imagenet/train'
# #             split = dataset_path.split('/')[-1]
        
# #         # Get all class names from S3
# #         if split == 'train':
# #             class_names = self.s3_loader.get_imagenet_classes()
# #         else:
# #             # For test split, we need to handle differently
# #             # For now, return empty arrays as in original code
# #             print(f"Test split loading from S3 not yet implemented")
# #             return np.array([]), np.array([])
        
# #         if not class_names:
# #             raise ValueError(f"No subdirectories found in {dataset_path}")
        
# #         # Filter to only valid ImageNet synsets (starting with 'n')
# #         class_names = [d for d in class_names if d.startswith('n') and len(d) > 1 and d[1:].replace('_', '').isdigit()]
# #         class_names = sorted(class_names)
        
# #         # Prepare lists to hold images and labels
# #         images = []
# #         labels = []
        
# #         # Load images from each class
# #         print(f"Loading images from {len(class_names)} classes...")
        
# #         # Use ThreadPoolExecutor for parallel loading
# #         def load_class_images(class_name):
# #             class_images = []
# #             class_labels = []
            
# #             try:
# #                 # Get all images for this class
# #                 image_keys = self.s3_loader.get_class_images(class_name)
                
# #                 # Filter only image files
# #                 img_files = [k for k in image_keys if k.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
# #                 for image_key in img_files:
# #                     try:
# #                         # Get image data from S3
# #                         image_data = self.s3_loader.get_image_data(image_key)
# #                         if image_data:
# #                             # Load image from bytes using PIL
# #                             img = Image.open(io.BytesIO(image_data))
# #                             # Convert to RGB if necessary
# #                             if img.mode != 'RGB':
# #                                 img = img.convert('RGB')
# #                             # Resize
# #                             img = img.resize(img_size, Image.Resampling.LANCZOS)
# #                             # Convert to array (matching keras img_to_array behavior)
# #                             img_array = np.array(img, dtype=np.float32)
                            
# #                             # Add to our collections
# #                             class_images.append(img_array)
# #                             class_labels.append(class_name)  # Use folder name directly as label
# #                     except Exception as e:
# #                         print(f"Error loading {image_key}: {e}")
# #                         continue
                        
# #             except Exception as e:
# #                 print(f"Error loading class {class_name}: {e}")
                
# #             return class_images, class_labels
        
# #         # Load images with some parallelism for better performance
# #         with ThreadPoolExecutor(max_workers=10) as executor:
# #             futures = {executor.submit(load_class_images, class_name): class_name 
# #                       for class_name in class_names}
            
# #             for future in as_completed(futures):
# #                 class_name = futures[future]
# #                 try:
# #                     class_images, class_labels = future.result()
# #                     if class_images:
# #                         images.extend(class_images)
# #                         labels.extend(class_labels)
# #                 except Exception as e:
# #                     print(f"Error processing class {class_name}: {e}")
        
# #         # Convert lists to numpy arrays
# #         images = np.array(images)
# #         labels = np.array(labels)
        
# #         return images, labels

# #     # def load(self, name):
# #     #     from dataset_service import _get_dataset_config

# #     #     dataset_path = os.path.join("data", "datasets", name)
# #     #     train_path = os.path.join(dataset_path, "train")
# #     #     test_path = os.path.join(dataset_path, "test")

# #     #     self.x_train, self.y_train = self.load_mini_imagenet(train_path)
# #     # #    self.x_test, self.y_test = self.load_mini_imagenet(test_path)
    
# #     #     self.directory_labels = _get_dataset_config("imagenet")["directory_labels"]
# #     #     print("loaded imagenet dataset")
    
    
# #     def load(self, name):
# #         from dataset_service import _get_dataset_config
        
# #         # Check if S3 bucket is configured
# #         bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
# #         if not bucket:
# #             raise RuntimeError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        
# #         # Construct paths that mimic the local structure but for S3
# #         # Original: data/datasets/imagenet/train
# #         # S3: imagenet/train
# #         dataset_path = os.path.join("data", "datasets", name) 
# #         train_path = os.path.join(dataset_path, "train")
# #         test_path = os.path.join(dataset_path, "test")

# #         # Load using the same function but it will now fetch from S3
# #         self.x_train, self.y_train = self.load_mini_imagenet(train_path)
# #         # self.x_test, self.y_test = self.load_mini_imagenet(test_path)
        
# #         self.directory_labels = _get_dataset_config("imagenet")["directory_labels"]
# #         print("loaded imagenet dataset")


# #     # def get_train_image_by_id(self, image_id):
# #     #     # Implement the logic to get an image by its ID from the ImageNet dataset
        
# #     #     print(f"Getting image with ID: {image_id}")
# #     #     data = self.x_train
# #     #     batch_index = image_id // len(data[0][0])
# #     #     image_index = image_id % len(data[0][0])
# #     #     image = data[batch_index][0][image_index]
# #     #     label = data[batch_index][1][image_index]
# #     #     return image, label
    
# #     # def get_test_image_by_id(self, image_id):
# #     #     # Implement the logic to get an image by its ID from the ImageNet dataset
# #     #     print(f"Getting image with ID: {image_id}")
# #     #     data = self.x_test
# #     #     batch_index = image_id // len(data[0][0])
# #     #     image_index = image_id % len(data[0][0])
# #     #     image = data[batch_index][0][image_index]
# #     #     label = data[batch_index][1][image_index]
# #     #     return image, label

# #     def get_train_image_by_id(self, image_id):
# #         # Check if the image_id is within the range of training data
# #         if image_id < len(self.x_train):
# #             image = self.x_train[image_id]
# #             label = self.y_train[image_id]
# #             print(f"Train image ID {image_id}: label {label}") 
# #         else:
# #             raise ValueError("Invalid image_id")

# #         return image, label
    
# #     def get_test_image_by_id(self, image_id):
# #         if image_id < len(self.x_test):
# #             image = self.x_test[image_id]
# #             label = self.y_test[image_id]
# #             print(f"Test image ID {image_id}: label {label}")
# #         else:
# #             raise ValueError("Invalid image_id")

# #         return image, label


# #     def directory_to_labels_conversion(self, label):
# #         from dataset_service import _get_dataset_config
# #         dir_to_readable = _get_dataset_config("imagenet")["directory_to_readable"]
# #         return dir_to_readable[label]
    
# #     def get_label_readable_name(self, label):
# #         return self.directory_to_labels_conversion(label)



# import os
# import time
# import io
# from PIL import Image
# from .dataset import Dataset
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# # from data.datasets.imagenet_info import IMAGENET_INFO
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from NMA.s3_connector.s3_dataset_loader import S3DatasetLoader
# from NMA.s3_connector.s3_imagenet_loader import S3ImagenetLoader

# import logging
# logger = logging.getLogger(__name__)

# class ImageNet(Dataset):
#     def __init__(self):
#         from dataset_service import _get_dataset_config
#         config = _get_dataset_config("imagenet")

#         super().__init__(config["dataset"], config["threshold"], config["infinity"], config["directory_labels"])
#         self.x_train = None
#         self.y_train = None
#         self.x_test = None
#         self.y_test = None
#         self.directory_labels = config["directory_labels"]
        
#     # def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
#     #     """
#     #     Load mini ImageNet dataset from a directory structure where:
#     #     - Each subdirectory name in dataset_path is a class label
#     #     - Each subdirectory contains images of that class
        
#     #     Parameters:
#     #     -----------
#     #     dataset_path : str
#     #         Path to the dataset directory (e.g., 'imagenet/train')
#     #     img_size : tuple
#     #         Size to resize the images to (height, width)
            
#     #     Returns:
#     #     --------
#     #     images : np.array
#     #         Array of preprocessed images with shape (n_samples, height, width, channels)
#     #     labels : np.array
#     #         Array of labels as folder names (e.g., 'n01440764')
#     #     """
#     #     if not os.path.exists(dataset_path):
#     #         raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
#     #     # Get all class names (subdirectories)
#     #     class_names = sorted([
#     #         d for d in os.listdir(dataset_path)
#     #         if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('n') and d[1:].isdigit()
#     #     ])
        
#     #     if not class_names:
#     #         raise ValueError(f"No subdirectories found in {dataset_path}")

#     #     # Prepare lists to hold images and labels
#     #     images = []
#     #     labels = []
        
#     #     # Load images from each class
#     #     print(f"Loading images from {len(class_names)} classes...")
#     #     for class_name in class_names:
#     #         class_path = os.path.join(dataset_path, class_name)
#     #         # Get all image files in the class directory
#     #         img_files = [f for f in os.listdir(class_path) 
#     #                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
#     #         for img_file in img_files:
#     #             img_path = os.path.join(class_path, img_file)
#     #             try:
#     #                 # Load and preprocess image
#     #                 img = load_img(img_path, target_size=img_size)
#     #                 img_array = img_to_array(img)
                    
#     #                 # Properly preprocess for ResNet50
#     #                 # img_array = preprocess_input(img_array)
                    
#     #                 # Add to our collections
#     #                 images.append(img_array)
#     #                 labels.append(class_name)  # Use folder name directly as label

#     #             except Exception as e:
#     #                 print(f"Error loading {img_path}: {e}")
        
#     #     # Convert lists to numpy arrays
#     #     images = np.array(images)
#     #     labels = np.array(labels)

#     #     return images, labels

#     def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
#         """
#         Returns:
#         --------
#         images : np.array
#             Array of preprocessed images with shape (n_samples, height, width, channels)
#         labels : np.array
#             Array of labels as folder names (e.g., 'n01440764')
#         """
#         # Initialize S3 loader if not already done
#         if self.s3_loader is None:
#             self.s3_loader = S3ImagenetLoader()
        
#         # Extract split type from dataset_path (e.g., 'imagenet/train' -> 'train')
#         # Handle both local-style paths and S3-style paths
#         if dataset_path.startswith('data/datasets/'):
#             # Local-style path: 'data/datasets/imagenet/train'
#             parts = dataset_path.split('/')
#             split = parts[-1]  # 'train' or 'test'
#         else:
#             # S3-style path: 'imagenet/train'
#             split = dataset_path.split('/')[-1]
        
#         # Get all class names from S3
#         if split == 'train':
#             class_names = self.s3_loader.get_imagenet_classes()
#         else:
#             # For test split, we need to handle differently
#             # For now, return empty arrays as in original code
#             print(f"Test split loading from S3 not yet implemented")
#             return np.array([]), np.array([])
        
#         if not class_names:
#             raise ValueError(f"No subdirectories found in {dataset_path}")
        
#         # Filter to only valid ImageNet synsets (starting with 'n')
#         class_names = [d for d in class_names if d.startswith('n') and len(d) > 1 and d[1:].replace('_', '').isdigit()]
#         class_names = sorted(class_names)
        
#         # Prepare lists to hold images and labels
#         images = []
#         labels = []
        
#         # Load images from each class
#         print(f"Loading images from {len(class_names)} classes...")
        
#         # Use ThreadPoolExecutor for parallel loading
#         def load_class_images(class_name):
#             class_images = []
#             class_labels = []
            
#             try:
#                 # Get all images for this class
#                 image_keys = self.s3_loader.get_class_images(class_name)
                
#                 # Filter only image files
#                 img_files = [k for k in image_keys if k.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
#                 for image_key in img_files:
#                     try:
#                         # Get image data from S3
#                         image_data = self.s3_loader.get_image_data(image_key)
#                         if image_data:
#                             # Load image from bytes using PIL
#                             img = Image.open(io.BytesIO(image_data))
#                             # Convert to RGB if necessary
#                             if img.mode != 'RGB':
#                                 img = img.convert('RGB')
#                             # Resize
#                             img = img.resize(img_size, Image.Resampling.LANCZOS)
#                             # Convert to array (matching keras img_to_array behavior)
#                             img_array = np.array(img, dtype=np.float32)
                            
#                             # Add to our collections
#                             class_images.append(img_array)
#                             class_labels.append(class_name)  # Use folder name directly as label
#                     except Exception as e:
#                         print(f"Error loading {image_key}: {e}")
#                         continue
                        
#             except Exception as e:
#                 print(f"Error loading class {class_name}: {e}")
                
#             return class_images, class_labels
        
#         # Load images with some parallelism for better performance
#         with ThreadPoolExecutor(max_workers=10) as executor:
#             futures = {executor.submit(load_class_images, class_name): class_name 
#                       for class_name in class_names}
            
#             for future in as_completed(futures):
#                 class_name = futures[future]
#                 try:
#                     class_images, class_labels = future.result()
#                     if class_images:
#                         images.extend(class_images)
#                         labels.extend(class_labels)
#                 except Exception as e:
#                     print(f"Error processing class {class_name}: {e}")
        
#         # Convert lists to numpy arrays
#         images = np.array(images)
#         labels = np.array(labels)
        
#         return images, labels

#     # def load(self, name):
#     #     from dataset_service import _get_dataset_config

#     #     dataset_path = os.path.join("data", "datasets", name)
#     #     train_path = os.path.join(dataset_path, "train")
#     #     test_path = os.path.join(dataset_path, "test")

#     #     self.x_train, self.y_train = self.load_mini_imagenet(train_path)
#     # #    self.x_test, self.y_test = self.load_mini_imagenet(test_path)
    
#     #     self.directory_labels = _get_dataset_config("imagenet")["directory_labels"]
#     #     print("loaded imagenet dataset")
    
    
#     def load(self, name):
#         from dataset_service import _get_dataset_config
        
#         # Check if S3 bucket is configured
#         bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
#         if not bucket:
#             raise RuntimeError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        
#         # Construct paths that mimic the local structure but for S3
#         # Original: data/datasets/imagenet/train
#         # S3: imagenet/train
#         dataset_path = os.path.join("data", "datasets", name) 
#         train_path = os.path.join(dataset_path, "train")
#         test_path = os.path.join(dataset_path, "test")

#         # Load using the same function but it will now fetch from S3
#         self.x_train, self.y_train = self.load_mini_imagenet(train_path)
#         # self.x_test, self.y_test = self.load_mini_imagenet(test_path)
        
#         self.directory_labels = _get_dataset_config("imagenet")["directory_labels"]
#         print("loaded imagenet dataset")


#     # def get_train_image_by_id(self, image_id):
#     #     # Implement the logic to get an image by its ID from the ImageNet dataset
        
#     #     print(f"Getting image with ID: {image_id}")
#     #     data = self.x_train
#     #     batch_index = image_id // len(data[0][0])
#     #     image_index = image_id % len(data[0][0])
#     #     image = data[batch_index][0][image_index]
#     #     label = data[batch_index][1][image_index]
#     #     return image, label
    
#     # def get_test_image_by_id(self, image_id):
#     #     # Implement the logic to get an image by its ID from the ImageNet dataset
#     #     print(f"Getting image with ID: {image_id}")
#     #     data = self.x_test
#     #     batch_index = image_id // len(data[0][0])
#     #     image_index = image_id % len(data[0][0])
#     #     image = data[batch_index][0][image_index]
#     #     label = data[batch_index][1][image_index]
#     #     return image, label

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


#     def directory_to_labels_conversion(self, label):
#         from dataset_service import _get_dataset_config
#         dir_to_readable = _get_dataset_config("imagenet")["directory_to_readable"]
#         return dir_to_readable[label]
    
#     def get_label_readable_name(self, label):
#         return self.directory_to_labels_conversion(label)
    
        
#     def load_from_s3(self, s3_client, bucket, prefix):
#         """
#         Load ImageNet dataset from S3 directly using the provided S3 client.
#         """
#         logger.info(f"Loading ImageNet from S3: {bucket}/{prefix}")
        
#         # Ensure we have the directory_labels loaded
#         from dataset_service import _get_dataset_config
#         config = _get_dataset_config("imagenet")
        
#         # Load required attributes if not already loaded
#         if not hasattr(self, 'directory_labels') or not self.directory_labels:
#             self.directory_labels = config["directory_labels"]
        
#         # Ensure prefix ends with / for proper directory listing
#         if not prefix.endswith('/'):
#             prefix = prefix + '/'
        
#         logger.info(f"Using prefix: {prefix}")
        
#         # List all class folders (n01440764/, n01443537/, etc.)
#         response = s3_client.list_objects_v2(
#             Bucket=bucket,
#             Prefix=prefix,
#             Delimiter='/'
#         )
        
#         class_folders = []
#         if 'CommonPrefixes' in response:
#             class_folders = [p['Prefix'] for p in response['CommonPrefixes']]
        
#         if not class_folders:
#             logger.error(f"No class folders found at {prefix} in bucket {bucket}")
#             raise ValueError(f"No class folders found at {prefix} in bucket {bucket}")
        
#         logger.info(f"Found {len(class_folders)} class folders")
        
#         # Initialize data lists
#         x_train = []
#         y_train = []
        
#         # Process each class folder (increased limits for real use)
#         max_classes = 20  # ✅ INCREASED: Process more classes
#         max_images_per_class = 50  # ✅ INCREASED: More images per class
        
#         processed_classes = 0
#         for folder in class_folders:
#             if processed_classes >= max_classes:
#                 break
                
#             # Extract folder name (e.g., 'n01440764' from 'imagenet/train/n01440764/')
#             folder_name = folder.rstrip('/').split('/')[-1]
            
#             # Skip non-class folders like 'LOC_synset_mapping.txt' directory
#             if not folder_name.startswith('n') or len(folder_name) < 5:
#                 continue
                
#             # Only process folders that exist in our directory_labels
#             if folder_name not in self.directory_labels:
#                 logger.debug(f"Skipping folder {folder_name} - not in directory_labels")
#                 continue
            
#             logger.info(f"Processing class folder: {folder_name}")
            
#             # List images in this class folder
#             try:
#                 images_response = s3_client.list_objects_v2(
#                     Bucket=bucket,
#                     Prefix=folder,
#                     MaxKeys=max_images_per_class
#                 )
                
#                 if 'Contents' not in images_response:
#                     logger.warning(f"No images found in {folder}")
#                     continue
                
#                 images_processed = 0
#                 # Process each image
#                 for item in images_response['Contents']:
#                     if images_processed >= max_images_per_class:
#                         break
                        
#                     if not item['Key'].lower().endswith(('.jpeg', '.jpg', '.png')):
#                         continue
                    
#                     # Download and process image
#                     try:
#                         img_response = s3_client.get_object(Bucket=bucket, Key=item['Key'])
#                         img_data = img_response['Body'].read()
                        
#                         from PIL import Image
#                         import io
#                         img = Image.open(io.BytesIO(img_data))
#                         if img.mode != 'RGB':
#                             img = img.convert('RGB')
#                         img = img.resize((224, 224)) 
#                         img_array = np.array(img) 
                        
#                         # Ensure correct shape
#                         if len(img_array.shape) == 3 and img_array.shape[2] == 3:
#                             x_train.append(img_array)
#                             y_train.append(folder_name)  # Use folder name (e.g., 'n01440764')
#                             images_processed += 1
                            
#                     except Exception as e:
#                         logger.warning(f"Error processing image {item['Key']}: {str(e)}")
#                         continue
                
#                 if images_processed > 0:
#                     logger.info(f"Loaded {images_processed} images from class {folder_name}")
#                     processed_classes += 1
                    
#             except Exception as e:
#                 logger.warning(f"Error listing images in {folder}: {str(e)}")
#                 continue
        
#         # Convert lists to numpy arrays
#         if not x_train:
#             logger.error("No images were processed successfully")
#             raise ValueError("Failed to load any images from S3")
        
#         x_train = np.array(x_train)
#         y_train = np.array(y_train)
        
#         # Store in instance variables
#         self.x_train = x_train
#         self.y_train = y_train
        
#         logger.info(f"Successfully loaded ImageNet: {x_train.shape} images, {len(y_train)} labels")
#         logger.info(f"Processed {processed_classes} classes")
#         logger.info(f"Sample labels: {y_train[:5]}")
        
#         return x_train, y_train



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

        # Extract split type from dataset_path (e.g., 'imagenet/train' -> 'train')
        # Handle both local-style paths and S3-style paths
        # if dataset_path.startswith('data/datasets/'):
        #     # Local-style path: 'data/datasets/imagenet/train'
        #     parts = dataset_path.split('/')
        #     split = parts[-1]  # 'train' or 'test'
        # else:
        #     # S3-style path: 'imagenet/train'
        #     split = dataset_path.split('/')[-1]
        
        norm_path = dataset_path.replace("\\", "/")
        if norm_path.startswith('data/datasets/'):
            split = os.path.basename(dataset_path)  # This will give 'train' or 'test'
        else:
            split = os.path.basename(dataset_path)
        print(f"Loading mini ImageNet for split: {split} from {dataset_path}")

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
    #     from dataset_service import _get_dataset_config

    #     dataset_path = os.path.join("data", "datasets", name)
    #     train_path = os.path.join(dataset_path, "train")
    #     test_path = os.path.join(dataset_path, "test")

    #     self.x_train, self.y_train = self.load_mini_imagenet(train_path)
    # #    self.x_test, self.y_test = self.load_mini_imagenet(test_path)
    
    #     self.directory_labels = _get_dataset_config("imagenet")["directory_labels"]
    #     print("loaded imagenet dataset")
    
    
    def load(self, name):
        from NMA.services.dataset_service import _get_dataset_config

        # 1. pull in your dataset config (for labels, thresholds, etc.)
        config = _get_dataset_config("imagenet")

        # 2. make sure the bucket is set
        bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
        if not bucket:
            raise RuntimeError("S3_DATASETS_BUCKET_NAME environment variable must be set")

        # 3. create an actual S3 client
        s3_client = get_datasets_s3_client()

        # 4. define the S3 key prefixes (not local paths!)
        train_prefix = f"{name}/train"
        # test_prefix  = f"{name}/test"

        # 5. load train & test splits via your helper
        #    note: load_from_s3(self, s3_client, bucket, prefix)
        self.x_train, self.y_train = self.load_from_s3(s3_client, bucket, train_prefix)
        # self.x_test,  self.y_test  = self.load_from_s3(s3_client, bucket, test_prefix)

        # 6. restore any other config you need
        self.directory_labels = config["directory_labels"]

        # 7. done!
        print(f"Loaded {len(self.x_train)} train images")



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
        # if self.x_train is None or self.y_train is None:
        #     self.load("imagenet")
            
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
        """
        Load ImageNet dataset from S3 directly using the provided S3 client.
        """
        logger.info(f"Loading ImageNet from S3: {bucket}/{prefix}")
        
        # Ensure we have the directory_labels loaded
        from NMA.services.dataset_service import _get_dataset_config
        config = _get_dataset_config("imagenet")
        
        # Load required attributes if not already loaded
        if not hasattr(self, 'directory_labels') or not self.directory_labels:
            self.directory_labels = config["directory_labels"]
        
        # Ensure prefix ends with / for proper directory listing
        if not prefix.endswith('/'):
            prefix = prefix + '/'
        
        logger.info(f"Using prefix: {prefix}")
        
        # List all class folders (n01440764/, n01443537/, etc.)
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
        
        # Initialize data lists
        x_train = []
        y_train = []
        
        max_classes = 1000
        max_images_per_class = 10  
        
        processed_classes = 0
        for folder in class_folders:
            if processed_classes >= max_classes:
                break
                
            # Extract folder name (e.g., 'n01440764' from 'imagenet/train/n01440764/')
            folder_name = folder.rstrip('/').split('/')[-1]
            
            # Skip non-class folders like 'LOC_synset_mapping.txt' directory
            if not folder_name.startswith('n') or len(folder_name) < 5:
                continue
                
            # Only process folders that exist in our directory_labels
            if folder_name not in self.directory_labels:
                logger.debug(f"Skipping folder {folder_name} - not in directory_labels")
                continue
            
            logger.info(f"Processing class folder: {folder_name}")
            
            # List images in this class folder
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
                # Process each image
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
        
        # Convert lists to numpy arrays
        if not x_train:
            logger.error("No images were processed successfully")
            raise ValueError("Failed to load any images from S3")
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        # Store in instance variables
        self.x_train = x_train
        self.y_train = y_train
        
        logger.info(f"Successfully loaded ImageNet: {x_train.shape} images, {len(y_train)} labels")
        logger.info(f"Processed {processed_classes} classes")
        logger.info(f"Sample labels: {y_train[:5]}")
        
        return x_train, y_train
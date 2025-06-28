import os
import time
from PIL import Image
from .dataset import Dataset
import numpy as np
from NMA.data.datasets.imagenet_info import IMAGENET_INFO
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageNet(Dataset):
    def __init__(self):
        super().__init__(IMAGENET_INFO["dataset"], IMAGENET_INFO["threshold"], IMAGENET_INFO["infinity"], IMAGENET_INFO["labels"])
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.directory_labels = None
        
    def load_mini_imagenet(self, dataset_path, img_size=(224, 224)):
        """
        Load mini ImageNet dataset from a directory structure where:
        - Each subdirectory name in dataset_path is a class label
        - Each subdirectory contains images of that class
        
        Parameters:
        -----------
        dataset_path : str
            Path to the dataset directory (e.g., 'imagenet/train')
        img_size : tuple
            Size to resize the images to (height, width)
            
        Returns:
        --------
        images : np.array
            Array of preprocessed images with shape (n_samples, height, width, channels)
        labels : np.array
            Array of labels as folder names (e.g., 'n01440764')
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Get all class names (subdirectories)
        class_names = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('n') and d[1:].isdigit()
        ])
        
        if not class_names:
            raise ValueError(f"No subdirectories found in {dataset_path}")

        
        # Prepare lists to hold images and labels
        images = []
        labels = []
        
        # Load images from each class
        print(f"Loading images from {len(class_names)} classes...")
        for class_name in class_names:
            class_path = os.path.join(dataset_path, class_name)
            # Get all image files in the class directory
            img_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Load and preprocess image
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img)
                    
                    # Properly preprocess for ResNet50
                    # img_array = preprocess_input(img_array)
                    
                    # Add to our collections
                    images.append(img_array)
                    labels.append(class_name)  # Use folder name directly as label

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def load(self, name):
        dataset_path = os.path.join("data", "datasets", name)
        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")

        self.x_train, self.y_train = self.load_mini_imagenet(train_path)
    #    self.x_test, self.y_test = self.load_mini_imagenet(test_path)
    
        self.directory_labels = IMAGENET_INFO["directory_labels"]
        print("loaded imagenet dataset")

    def get_train_image_by_id(self, image_id):
        # Implement the logic to get an image by its ID from the ImageNet dataset
        
        print(f"Getting image with ID: {image_id}")
        data = self.train_data
        batch_index = image_id // len(data[0][0])
        image_index = image_id % len(data[0][0])
        image = data[batch_index][0][image_index]
        label = data[batch_index][1][image_index]
        return image, label
    
    def get_test_image_by_id(self, image_id):
        # Implement the logic to get an image by its ID from the ImageNet dataset
        print(f"Getting image with ID: {image_id}")
        data = self.test_data
        batch_index = image_id // len(data[0][0])
        image_index = image_id % len(data[0][0])
        image = data[batch_index][0][image_index]
        label = data[batch_index][1][image_index]
        return image, label

    def directory_to_labels_conversion(self, label):
        dir_to_readable = IMAGENET_INFO["directory_to_readable"]
        return dir_to_readable[label]
    
    def get_label_readable_name(self, label):
        return self.directory_to_labels_conversion(label)


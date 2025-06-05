from sklearn.model_selection import train_test_split
import numpy as np
from NMA.services.adversarial_files.score_calculator import ScoreCalculator
from NMA.services.dataset_service import get_dataset_labels
from NMA.s3_connector.s3_dataset_utils import load_dataset_numpy
from NMA.utilss.files_utils import preprocess_numpy_image
import tensorflow as tf
import os, tempfile
from dotenv import load_dotenv
load_dotenv() 
from NMA.utilss.s3_utils import get_users_s3_client 

class AdversarialDataset:
    def __init__(self, Z_file, clean_images, adversarial_images, model_filename, dataset):
        self.Z_matrix = Z_file
        self.dataset = dataset
        
        # Initialize S3 client
        s3_client = get_users_s3_client()
        s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
        if not s3_bucket:
            raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
        
        # Load the model from S3
        try:
            # Check if model_filename is a full S3 URI or just a path
            if model_filename.startswith('s3://'):
                parts = model_filename.replace('s3://', '').split('/', 1)
                bucket = parts[0]
                key = parts[1]
            else:
                # Assume it's a relative path in the default bucket
                bucket = s3_bucket
                key = model_filename
            
            print(f"Loading model from S3: {bucket}/{key}")
            
            # Check if the file exists in S3
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                print(f"Model file exists in S3: {bucket}/{key}")
            except Exception as e:
                raise FileNotFoundError(f"Model file '{bucket}/{key}' not found in S3: {str(e)}")
            
            # Load model directly from memory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_path = os.path.join(temp_dir, 'model.keras')
                s3_client.download_file(bucket, key, temp_model_path)
                self.model = tf.keras.models.load_model(temp_model_path)
                print(f"Model loaded successfully from S3: '{bucket}/{key}'.")
                
        except Exception as e:
            raise ValueError(f"Error loading model from '{model_filename}': {e}")
        
        # Load images from S3 with model-aware preprocessing
        if clean_images is None and adversarial_images is None:
            # Load from default S3 locations
            print(f"Loading clean images for {dataset} from S3...")
            clean_data = load_dataset_numpy(dataset, 'clean')
            
            print(f"Loading adversarial images for {dataset} from S3...")
            adversarial_data = load_dataset_numpy(dataset, 'adversarial')
            
            # Convert S3 dictionary format and apply model-aware preprocessing
            self.clear_images = self._process_s3_data_with_model(clean_data)
            self.adversarial_images = self._process_s3_data_with_model(adversarial_data)
            
        else:
            # Handle custom paths - extract folder type from path
            if clean_images:
                if 'clean' in clean_images.lower():
                    clean_data = load_dataset_numpy(dataset, 'clean')
                    self.clear_images = self._process_s3_data_with_model(clean_data)
                else:
                    # Handle other custom logic if needed
                    raise ValueError(f"Unsupported clean images path: {clean_images}")
            
            if adversarial_images:
                if 'adversarial' in adversarial_images.lower():
                    adversarial_data = load_dataset_numpy(dataset, 'adversarial')
                    self.adversarial_images = self._process_s3_data_with_model(adversarial_data)
                else:
                    # Handle other custom logic if needed
                    raise ValueError(f"Unsupported adversarial images path: {adversarial_images}")

        print(f"Loaded {len(self.clear_images)} clean images")
        print(f"Loaded {len(self.adversarial_images)} adversarial images")

        self.labels = get_dataset_labels(dataset)
        if self.labels is None:
            raise ValueError(f"Info file for the dataset {dataset} not found.")      

        # Create ScoreCalculator with S3 support
        self.score_calculator = ScoreCalculator(self.Z_matrix, self.labels)


    def _process_s3_data_with_model(self, s3_data):
        """
        Convert S3 dictionary of individual image files to list of images
        with the same preprocessing as the original load_numpy_from_directory.
        """
        if isinstance(s3_data, dict):
            if not s3_data:
                return []
            
            # Sort by filename to maintain consistent order
            sorted_items = sorted(s3_data.items(), key=lambda x: x[0])
            
            processed_images = []
            for filename, image_array in sorted_items:
                try:
                    # Apply the same preprocessing as the original function
                    preprocess_image = preprocess_numpy_image(self.model, image_array)
                    processed_images.append(preprocess_image)
                except Exception as e:
                    print(f"Error preprocessing image {filename}: {e}")
                    continue
            
            return processed_images
        
        elif isinstance(s3_data, list):
            return s3_data
        else:
            print(f"Warning: Unexpected data type from S3: {type(s3_data)}")
            return []
        
        
    def create_logistic_regression_dataset(self):
        scores = []
        labels = []

        print("getting preprocess function...")

        try:
            for image in self.clear_images[:50]:
                score = self.score_calculator.calculate_adversarial_score(self.model.predict(image))
                scores.append(score)
                labels.append(0)
        except Exception as e:
            print(f"Error processing clean image: {e}")
        
        # Generate features for PGD attacks
        print("Generating attack features...")
        try:
            for adv_image in self.adversarial_images[:50]:
                score = self.score_calculator.calculate_adversarial_score(self.model.predict(adv_image))
                scores.append(score)
                labels.append(1)
        except Exception as e:
            print(f"Error processing PGD attack on image: {e}")

        print("labels:", labels)
        print("scores:", scores)
        
        # Convert to numpy arrays
        X = np.array(scores)
        y = np.array(labels)

        # Reshape X to ensure it is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Clean samples: {sum(y_train == 0)}, Adversarial samples: {sum(y_train == 1)}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Clean samples: {sum(y_test == 0)}, Adversarial samples: {sum(y_test == 1)}")

        return X_train, y_train, X_test, y_test

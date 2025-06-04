
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics import f1_score
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time
import logging
import boto3
from botocore.exceptions import ClientError
import tempfile
from NMA.utilss.s3_utils import get_users_s3_client, get_datasets_s3_client
import io 

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, model, top_k, min_confidence, model_filename, dataset):
        MODELS_PATH = os.getenv("MODELS_PATH")
        model_file_path = f'{MODELS_PATH}/{model_filename}.keras'
        
        self.model = model
        self.model_filename = model_file_path
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.size = (256, 256)
        self.accuracy = -1
        self.loss = -1
        self.f1score = -1
        self.dataset = dataset
        
        ### S3 implementation ### 
        
        # Get S3 bucket from environment variable
        self.s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
        if not self.s3_bucket:
            raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
        
        # Handle S3 paths
        if model_filename.startswith('s3://'):
            # Extract bucket and key from S3 URI
            parts = model_filename.replace('s3://', '').split('/', 1)
            self.s3_bucket = parts[0]  # Override bucket if specified in path
            self.s3_key = parts[1]
            self.model_filename = model_filename
        else:
            # Use the provided key with the bucket from environment
            self.s3_key = model_filename
            self.model_filename = f"s3://{self.s3_bucket}/{model_filename}"
        
        logger.info(f"Model initialized with S3 path: {self.model_filename}")


### original implemetation ###

    # def load_model(self):
    #     if os.path.exists(self.model_filename):
    #         self.model = tf.keras.models.load_model(self.model_filename)
    #         print(f'Model {self.model_filename} has been loaded')
    #     else:
    #         print(f'Model {self.model_filename} does not exist')


### S3 implementation ### 
    def save_model(self):
        """Save the model to S3"""
        logger.info(f"Saving model to S3: {self.model_filename}")
        # Create a temporary file to save the model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = os.path.join(temp_dir, 'model.keras')
            
            # Save the model to the temporary file
            self.model.save(temp_model_path)
            
            # Upload to S3
            s3_client = get_users_s3_client()
            s3_client.upload_file(temp_model_path, self.s3_bucket, self.s3_key)
            logger.info(f'Model has been saved to S3: {self.model_filename}')
      
### original implemetation ###   
    # def save_model(self):
    #     self.model.save(self.model_filename)
    #     print(f'Model has been saved: {self.model_filename}') 
    
    
### S3 implementation ### 
    def save_model(self):
        """Save the model to S3"""
        logger.info(f"Saving model to S3: {self.model_filename}")
        # Create a temporary file to save the model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = os.path.join(temp_dir, 'model.keras')
            
            # Save the model to the temporary file
            self.model.save(temp_model_path)
            
            # Upload to S3
            s3_client = get_users_s3_client()
            s3_client.upload_file(temp_model_path, self.s3_bucket, self.s3_key)
            logger.info(f'Model has been saved to S3: {self.model_filename}')
            
    def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
        """Check if a file exists in S3"""
        s3_client = get_users_s3_client()
        print(f"Checking if file exists in S3: {bucket_name}/{s3_key}")
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            print(f"File found: {bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"File not found: {bucket_name}/{s3_key}, Error: {str(e)}")
            return False
        
    def model_evaluation(self, x_test, y_test):
        if self.accuracy != -1 & self.loss != -1:
            return self.accuracy
        batch_size = 64
        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.loss, self.accuracy = self.model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
        print(f'Model accuracy: {self.accuracy:.4f}')
        print(f'Model loss: {self.loss:.4f}')
        
        return self.accuracy
 
### S3 implementation ### 
    def model_evaluate_imagenet(self, test_data_s3_path):
        """Evaluate model on ImageNet data from S3"""
        # Extract bucket and prefix
        if test_data_s3_path.startswith('s3://'):
            parts = test_data_s3_path.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            prefix = parts[1]
        else:
            bucket = self.s3_bucket
            prefix = test_data_s3_path
        
        # Create a temporary directory to store the test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the test dataset from S3
            s3_client = get_datasets_s3_client()
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # Get relative path
                    key = obj['Key']
                    relative_path = key[len(prefix):].lstrip('/')
                    if not relative_path:
                        continue
                    
                    # Create local directory structure
                    local_path = os.path.join(temp_dir, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download file
                    s3_client.download_file(bucket, key, local_path)
            
            # Now we have the test data locally, proceed with evaluation
            img_height, img_width = 224, 224
            batch_size = 32

            test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

            test_generator = test_datagen.flow_from_directory(
                temp_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

            self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            # Evaluate the model
            start_time = time.time()

            self.loss, self.accuracy = self.model.evaluate(
                test_generator,
                steps=test_generator.samples // batch_size,
                verbose=1
            )

            elapsed_time = time.time() - start_time

            # Print results
            logger.info(f"Loss: {self.loss:.4f}")
            logger.info(f"Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
            logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    
    
    def model_accuracy_selected(self, dataset_instance, selected_labels):       
        # works for cifar100 dataset 
        x_test = dataset_instance.x_test
        y_test = dataset_instance.y_test

        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)

        # validation for specific labels accuracy using mask
        selected_indices = [selected_labels.index(label) for label in selected_labels]

        # Create masks for train and test sets
        test_mask = np.isin(np.argmax(y_test, axis=1), selected_indices)

        # Filter x_test, and y_test
        x_test_filtered = x_test[test_mask]
        y_test_filtered = y_test[test_mask]

        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        batch_size = 320  # Adjust the batch size as needed
        test_fitlered_loss, test_filtered_accuracy = self.model.evaluate(x_test_filtered, y_test_filtered, batch_size=batch_size,verbose=0)
        print(f'selected Labels accuracy: {test_filtered_accuracy:.4f}')
        print(f'selected Labels loss: {test_fitlered_loss:.4f}')

        return test_filtered_accuracy, test_fitlered_loss


    def model_f1score(self, x_test, y_test):
        # works for cifar100 dataset
        if self.f1score != -1:
            return self.f1score
        
        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)

        y_test_pred = self.model.predict(x_test)

        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_test_true_classes = np.argmax(y_test, axis=1)

        self.f1score = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        print(f'F1 Score (full dataset): {self.f1score:.4f}')
    
    def model_f1score_selected(self, dataset_instance, selected_labels): 
        # works for cifar100 dataset
        x_test = dataset_instance.x_test
        y_test = dataset_instance.y_test

        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)

        # validation for specific labels accuracy using mask
        selected_indices = [selected_labels.index(label) for label in selected_labels]

        # Create masks for train and test sets
        test_mask = np.isin(np.argmax(y_test, axis=1), selected_indices)

        # Filter x_test, and y_test
        x_test_filtered = x_test[test_mask]
        y_test_filtered = y_test[test_mask]

        y_test_filtered_pred = self.model.predict(x_test_filtered)
        y_test_filtered_pred_classes = np.argmax(y_test_filtered_pred, axis=1)
        y_test_filtered_true_classes = np.argmax(y_test_filtered, axis=1)

        f1_filtered = f1_score(y_test_filtered_true_classes, y_test_filtered_pred_classes, average='weighted')
        print(f'F1 Score (selected labels): {f1_filtered:.4f}')
        return f1_filtered

    def predict(self, image_input):
        if "cifar100" in self.dataset:
            return self.predict_cifar100(image_input)
        elif "imagenet" in self.dataset:
            return self.predict_imagenet(image_input)
        else:
            print("Model not recognized")
    

### S3 implementation ### 
    def predict_cifar100(self, image_input):
        """Make predictions for CIFAR-100 images from S3"""
        expected_size = (32, 32)
        
        # Extract bucket and key from S3 path
        if image_input.startswith('s3://'):
            parts = image_input.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            s3_key = parts[1]
        else:
            bucket = self.s3_bucket
            s3_key = image_input
        
        # Download image directly into memory
        s3_client = get_datasets_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        image_bytes = response['Body'].read()
        
        # Load image from bytes
        with io.BytesIO(image_bytes) as image_buffer:
            img = tf.keras.preprocessing.image.load_img(image_buffer, target_size=expected_size)
        
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = self.model.predict(img_array)

        # Get the indices of the top k predictions
        top_indices = np.argsort(predictions[0])[-self.top_k:][::-1]
        
        # Get the probabilities for those indices
        top_probabilities = predictions[0][top_indices]
        
        # Combine indices and probabilities
        results = [(idx, prob) for idx, prob in zip(top_indices, top_probabilities)]
        
        return results

    
### S3 implementation ### 
    def predict_imagenet(self, image_path):
        """Make predictions for ImageNet images from S3"""
        # Extract bucket and key from S3 path
        if image_path.startswith('s3://'):
            parts = image_path.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            s3_key = parts[1]
        else:
            bucket = self.s3_bucket
            s3_key = image_path
        
        # Download image directly into memory
        s3_client = get_datasets_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        image_bytes = response['Body'].read()
        
        # Load image from bytes
        with io.BytesIO(image_bytes) as image_buffer:
            img = tf.keras.preprocessing.image.load_img(image_buffer, target_size=(224, 224))
        
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=self.top_k)[0]

        return decoded_predictions
     
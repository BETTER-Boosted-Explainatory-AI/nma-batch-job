import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import numpy as np
import logging
from botocore.exceptions import ClientError
import io
from NMA.utilss.s3_utils import get_users_s3_client 
from datetime import datetime, timezone
logger = logging.getLogger(__name__)
S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
if not S3_BUCKET:
    raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")


class AdversarialDetector:
    def __init__(self, model_folder, detector_filename="logistic_regression_model.pkl", threshold=0.5):
        
        self.model_folder = model_folder
        self.threshold = threshold
    
        self.s3_detector_key = f'{model_folder}/{detector_filename}'
        self.detector = None
        
        if s3_file_exists(S3_BUCKET, self.s3_detector_key):
            self._load_detector_from_s3()
        else:
            logger.info(f"No existing detector found at s3://{S3_BUCKET}/{self.s3_detector_key}")


    def does_detector_exist(self):
        return self.detector is not None

    def predict(self, X):
        if self.detector is None:
            raise ValueError("Detector model is not trained or loaded.")
        if X is None or len(X) == 0:
            raise ValueError("Input data is empty or None.")
        
        y_pred_proba = self.detector.predict_proba(X)[:, 1]
        return (y_pred_proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        if self.detector is None:
            raise ValueError("Detector model is not trained or loaded.")
        if X is None or len(X) == 0:
            raise ValueError("Input data is empty or None.")
        return self.detector.predict_proba(X)

    def train_adversarial_detector(self, dataset):
        X_train, y_train = dataset['X_train'], dataset['y_train']
        print("Training adversarial detector...")
       
        detector = LogisticRegression(max_iter=1000, class_weight='balanced')
        detector.fit(X_train, y_train)
        
        self.detector = detector
        X_test, y_test = dataset['X_test'], dataset['y_test']
        y_pred_proba = self.predict_proba(X_test)[:, 1]
       
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        self.threshold = optimal_threshold
        
        print("Training completed. Saving to S3...")
        self._save_detector_to_s3(detector, optimal_threshold)
        return detector
    
    def _load_detector_from_s3(self):
        try:
            s3_client = get_users_s3_client()
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=self.s3_detector_key)
            detector_bytes = response['Body'].read()
            
            with io.BytesIO(detector_bytes) as buffer:
                self.detector, self.threshold = joblib.load(buffer)
                
            logger.info(f"Detector model loaded from 's3://{S3_BUCKET}/{self.s3_detector_key}'")
        except Exception as e:
            logger.error(f"Error loading detector from S3: {e}")
            self.detector = None
            
    def _save_detector_to_s3(self, detector, threshold):
        """Save detector model directly to S3 from memory"""
        try:
            s3_client = get_users_s3_client()
            buffer = io.BytesIO()
            joblib.dump((detector, threshold), buffer)
            buffer.seek(0)  # Reset buffer position to the beginning

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            s3_detector_key_new = f'{self.model_folder}/logistic_regression_model_{timestamp}.pkl'
            s3_client.upload_fileobj(
                buffer, 
                S3_BUCKET, 
                s3_detector_key_new
            )
            logger.info(f"Detector model saved to 's3://{S3_BUCKET}/{s3_detector_key_new}'")
      
        except Exception as e:
            logger.error(f"Error saving detector to S3: {e}")
            raise


def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
    s3_client = get_users_s3_client()
    print(f"Checking if file exists in S3: {bucket_name}/{s3_key}")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File found: {bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"File not found: {bucket_name}/{s3_key}, Error: {str(e)}")
        return False

def upload_file_to_s3(local_file_path: str, bucket_name: str, s3_key: str):
    s3_client = get_users_s3_client()
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        logger.info(f"File uploaded to s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        raise

def download_file_from_s3(bucket_name: str, s3_key: str, local_file_path: str):
    s3_client = get_users_s3_client()
    try:
        s3_client.download_file(bucket_name, s3_key, local_file_path)
        logger.info(f"File downloaded from s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise
    

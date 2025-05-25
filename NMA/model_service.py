from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import tempfile
import zipfile
import io
import boto3
from botocore.exceptions import ClientError
from contextlib import contextmanager
from typing import Dict, Any, Optional
import tensorflow as tf 
import logging
from classes.model import Model
from classes.dendrogram import Dendrogram
# from utilss.photos_utils import preprocess_loaded_image
from NMA.dataset_service import get_dataset_labels
import json
from NMA.utilss.enums.datasets import DatasetsEnum


# Set up logging
logger = logging.getLogger(__name__)

# S3 Configuration - Always use S3
S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
if not S3_BUCKET:
    raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")

### original implemetation ###
def get_model():
    return None

### S3 implementation ### 
def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
    """
    Get the model path in S3 based on user_id and model_id
    """
    s3_client = boto3.client('s3')
    
    # Construct the S3 prefix (equivalent to directory path in S3)
    s3_prefix = f"{str(user_id)}/{str(model_id)}/"
    
    # Check if this prefix exists in S3
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=s3_prefix,
            MaxKeys=1 
        )
        
        # If there are objects with this prefix, the "path" exists
        if 'Contents' in response:
            return s3_prefix
        else:
            logger.debug(f"S3 prefix {s3_prefix} does not exist in bucket {S3_BUCKET}")
            return None
    except ClientError as e:
        logger.error(f"Error checking S3 prefix {s3_prefix}: {e}")
        return None


### S3 implementation ### 
def _get_model_filename(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
    """Get S3 model filename"""
    model_path = _get_model_path(user_id, model_id)
    if model_path is None:
        ValueError(f"Model path for user {user_id} and model {model_id} does not exist in S3")
    
    # Find .keras file in S3 prefix
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=model_path
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.keras'):
                    return obj['Key']   
        
        return None
    except ClientError as e:
        logger.error(f"Error listing S3 objects: {e}")
        return None
        

### S3 implementation ### 
def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
    """Load model from S3 with minimal memory usage"""
    logger.info(f"Loading model {model_path} for dataset {dataset_str}")
    
    if dataset_str != DatasetsEnum.IMAGENET.value and dataset_str != DatasetsEnum.CIFAR100.value:
        raise ValueError(f"Invalid dataset: {dataset_str}")
    
    # Always load from S3
    if model_path.startswith('s3://'):
        # S3 path - extract bucket and key
        parts = model_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        s3_key = parts[1]
        model = load_model_from_s3(bucket, s3_key)
        effective_path = model_path
    else:
        # S3 key without s3:// prefix
        model = load_model_from_s3(S3_BUCKET, model_path)
        effective_path = f"s3://{S3_BUCKET}/{model_path}"
    
    print(f"Model {effective_path} has been loaded")
    
    return Model(
        model, 
        dataset_config["top_k"], 
        dataset_config["min_confidence"], 
        effective_path, 
        dataset_config["dataset"]
    )
    
### S3 implementation ### 
@contextmanager
def streaming_zip_extraction(bucket_name: str, s3_key: str):
    """Stream a zip file from S3 and extract it to a temporary directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        s3_client = boto3.client('s3')
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            buffer = io.BytesIO(response['Body'].read())
            with zipfile.ZipFile(buffer) as zip_ref:
                zip_ref.extractall(temp_dir)
            yield temp_dir
        except ClientError as e:
            logger.error(f"Error streaming zip from S3 ({bucket_name}/{s3_key}): {e}")
            raise

def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
    """Check if a file exists in S3"""
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except ClientError:
        return False

def read_json_from_s3(bucket_name: str, s3_key: str) -> Any:
    """Read a JSON file from S3"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        logger.error(f"Error reading JSON from S3 ({bucket_name}/{s3_key}): {e}")
        raise

def load_model_from_s3(bucket_name: str, s3_key: str):
    """Load Keras model from S3"""
    # Check if it's a .keras file
    if s3_key.endswith('.keras'):
        with streaming_zip_extraction(bucket_name, s3_key) as temp_dir:
            model = tf.keras.models.load_model(temp_dir)
            return model
    else:
        # Assume it's a SavedModel directory
        with tempfile.TemporaryDirectory() as temp_dir:
            s3_client = boto3.client('s3')
            # List all objects with the prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_key)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # Get relative path
                    key = obj['Key']
                    relative_path = key[len(s3_key):].lstrip('/')
                    if not relative_path:
                        continue
                    
                    # Create local directory structure
                    local_path = os.path.join(temp_dir, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download file
                    s3_client.download_file(bucket_name, key, local_path)
            
            model = tf.keras.models.load_model(temp_dir)
            return model
        

# ### S3 implementation ### 
def construct_model(model_path: str, dataset_config: Dict[str, Any]) -> Model:
    """Load model from S3 and construct Model object"""
    logger.info(f"Loading model {model_path} for dataset {dataset_config['dataset']}")
    
    # Check if model_path is an S3 path or just a key
    if model_path.startswith('s3://'):
        bucket_name = model_path.replace('s3://', '').split('/')[0]
        s3_key = '/'.join(model_path.replace('s3://', '').split('/')[1:])
    else:
        bucket_name = S3_BUCKET
        s3_key = model_path
    
    # Check if model exists in S3
    if not s3_file_exists(bucket_name, s3_key):
        raise FileNotFoundError(f'Model s3://{bucket_name}/{s3_key} does not exist')
    
    # Load model from S3
    resnet_model = load_model_from_s3(bucket_name, s3_key)
    
    return Model(
        resnet_model, 
        dataset_config["top_k"], 
        dataset_config["min_confidence"],
        f"s3://{bucket_name}/{s3_key}", 
        dataset_config["dataset"]
    )
    
def delete_model():
    return None


# ### S3 implementation ### 
def query_model(top_label, model_id, graph_type, user):
    """Query model using dendrogram from S3"""
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user.get_user_folder(), model_info, graph_type)
        model_path = model_files["model_graph_folder"]
        dendrogram_s3_key = f'{model_path}/dendrogram'
    
    # We need to modify the Dendrogram class to work with S3
    # For now, let's download the dendrogram file to a temporary location
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dendrogram = os.path.join(temp_dir, 'dendrogram')
        s3_client = boto3.client('s3')
        
        # Download dendrogram files if they exist
        dendrogram_json_key = f"{dendrogram_s3_key}.json"
        dendrogram_pkl_key = f"{dendrogram_s3_key}.pkl"
        
        if s3_file_exists(S3_BUCKET, dendrogram_json_key):
            s3_client.download_file(S3_BUCKET, dendrogram_json_key, f"{temp_dendrogram}.json")
        
        if s3_file_exists(S3_BUCKET, dendrogram_pkl_key):
            s3_client.download_file(S3_BUCKET, dendrogram_pkl_key, f"{temp_dendrogram}.pkl")
        
        dendrogram = Dendrogram(temp_dendrogram)
        dendrogram.load_dendrogram()
        
        consistency = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
        if consistency is None:
            raise ValueError(f"Label {top_label} not found in dendrogram")
        else:
            logger.debug(f"Top label: {top_label}")
            logger.debug(f"Consistency: {consistency}")
        
        return consistency


# ### S3 implementation ### 
# def query_predictions(model_id, graph_type, image, user):
#     """Query predictions using model from S3"""
#     model_info = get_user_models_info(user, model_id)
#     if model_info is None:
#         raise ValueError(f"Model ID {model_id} not found in models.json")
#     else:
#         model_files = get_model_files(user.get_user_folder(), model_info, graph_type)

#     dataset = model_info["dataset"]
#     labels = get_dataset_labels(dataset)
#     model_s3_key = model_files["model_file"]
    
#     if s3_file_exists(S3_BUCKET, model_s3_key):
#         # Load model from S3
#         current_model = load_model_from_s3(S3_BUCKET, model_s3_key)
#         logger.debug(f"Model loaded successfully from 's3://{S3_BUCKET}/{model_s3_key}'.")
#     else:
#         raise ValueError(f"Model file s3://{S3_BUCKET}/{model_s3_key} does not exist")
    
#     preprocessed_image = preprocess_loaded_image(current_model, image)
#     predictions = get_top_k_predictions(current_model, preprocessed_image, labels)
#     top_label = predictions[0][0]  # Top label
#     top_3_predictions = predictions[:3]  # Top 3 predictions
    
#     return top_label, top_3_predictions


# ### S3 implementation ### 
def get_user_models_info(user, model_id):
    """Get model info from models.json in S3"""
    # Assuming user object has a method to get the models.json path in S3
    # If not, we'll need to construct it
    s3_models_json_key = f"{user.get_user_folder()}/models.json"
    
    if s3_file_exists(S3_BUCKET, s3_models_json_key):
        models_data = read_json_from_s3(S3_BUCKET, s3_models_json_key)
        logger.debug(f"Models data loaded from s3://{S3_BUCKET}/{s3_models_json_key}")
    else:
        models_data = []
        raise ValueError(f"Models metadata file 's3://{S3_BUCKET}/{s3_models_json_key}' not found.")
    
    if model_id is None:
        return models_data
    else:
        return get_model_info(models_data, model_id)


### S3 implementation ### 
def get_model_files(user_folder: str, model_info: dict, graph_type: str):
    """Get model file paths in S3"""
    logger.info(f"Getting model files for user folder: {user_folder}, model info: {model_info}, graph type: {graph_type}")
    
    # Construct S3 paths
    model_subfolder = f"{user_folder}/{model_info['model_id']}"
    model_file = f"{model_subfolder}/{model_info['file_name']}"
    
    if not s3_file_exists(S3_BUCKET, model_file):
        model_file = None
        raise ValueError(f"Model file s3://{S3_BUCKET}/{model_file} does not exist")
    
    model_graph_folder = f"{model_subfolder}/{graph_type}"
    
    # Check if the graph folder "exists" by listing objects with this prefix
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=model_graph_folder,
        MaxKeys=1
    )
    
    if 'Contents' not in response:
        model_graph_folder = None
        raise ValueError(f"Model graph folder s3://{S3_BUCKET}/{model_graph_folder} does not exist")
    
    # Define file paths
    Z_file = f"{model_graph_folder}/dendrogram.pkl"
    Z_file_exists = s3_file_exists(S3_BUCKET, Z_file)
    if not Z_file_exists:
        Z_file = None
        logger.debug(f"Z file s3://{S3_BUCKET}/{Z_file} does not exist")
    
    dendrogram_file = f"{model_graph_folder}/dendrogram.json"
    dendrogram_file_exists = s3_file_exists(S3_BUCKET, dendrogram_file)
    if not dendrogram_file_exists:
        dendrogram_file = None
        logger.debug(f"Dendrogram file s3://{S3_BUCKET}/{dendrogram_file} does not exist")
    
    detector_filename = f"{model_graph_folder}/logistic_regression_model.pkl"
    detector_file_exists = s3_file_exists(S3_BUCKET, detector_filename)
    if not detector_file_exists:
        detector_filename = None
        logger.debug(f"Detector model file s3://{S3_BUCKET}/{detector_filename} does not exist")
    
    dataframe_filename = f"{model_graph_folder}/edges_df.csv"
    dataframe_file_exists = s3_file_exists(S3_BUCKET, dataframe_filename)
    if not dataframe_file_exists:
        dataframe_filename = None
        logger.debug(f"Dataframe file s3://{S3_BUCKET}/{dataframe_filename} does not exist")
    
    return {
        "model_file": model_file, 
        "Z_file": Z_file, 
        "dendrogram": dendrogram_file, 
        "detector_filename": detector_filename, 
        "dataframe": dataframe_filename, 
        "model_graph_folder": model_graph_folder
    }
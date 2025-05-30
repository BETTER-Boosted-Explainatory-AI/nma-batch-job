# from dotenv import load_dotenv
# load_dotenv()

# import os
# import numpy as np
# import tempfile
# import zipfile
# import io
# import boto3
# from botocore.exceptions import ClientError
# from contextlib import contextmanager
# from typing import Dict, Any, Optional
# import tensorflow as tf 
# import logging
# from classes.model import Model
# from classes.dendrogram import Dendrogram
# # from NMA.utilss.photos_utils import preprocess_loaded_image
# from NMA.dataset_service import get_dataset_labels
# import json
# from NMA.utilss.enums.datasets import DatasetsEnum


# # Set up logging
# logger = logging.getLogger(__name__)

# # S3 Configuration - Always use S3
# S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
# if not S3_BUCKET:
#     raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")

# ### original implemetation ###
# def get_model():
#     return None

# ### S3 implementation ### 
# def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
#     """
#     Get the model path in S3 based on user_id and model_id
#     """
#     s3_client = boto3.client('s3')
    
#     # Construct the S3 prefix (equivalent to directory path in S3)
#     s3_prefix = f"{str(user_id)}/{str(model_id)}/"
    
#     # Check if this prefix exists in S3
#     try:
#         response = s3_client.list_objects_v2(
#             Bucket=S3_BUCKET,
#             Prefix=s3_prefix,
#             MaxKeys=1 
#         )
        
#         # If there are objects with this prefix, the "path" exists
#         if 'Contents' in response:
#             return s3_prefix
#         else:
#             logger.debug(f"S3 prefix {s3_prefix} does not exist in bucket {S3_BUCKET}")
#             return None
#     except ClientError as e:
#         logger.error(f"Error checking S3 prefix {s3_prefix}: {e}")
#         return None


# ### S3 implementation ### 
# def _get_model_filename(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
#     """Get S3 model filename"""
#     model_path = _get_model_path(user_id, model_id)
#     if model_path is None:
#         ValueError(f"Model path for user {user_id} and model {model_id} does not exist in S3")
    
#     # Find .keras file in S3 prefix
#     s3_client = boto3.client('s3')
    
#     try:
#         response = s3_client.list_objects_v2(
#             Bucket=S3_BUCKET,
#             Prefix=model_path
#         )
        
#         if 'Contents' in response:
#             for obj in response['Contents']:
#                 if obj['Key'].endswith('.keras'):
#                     return obj['Key']   
        
#         return None
#     except ClientError as e:
#         logger.error(f"Error listing S3 objects: {e}")
#         return None
        

# ### S3 implementation ### 
# def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
#     """Load model from S3 with minimal memory usage"""
#     logger.info(f"Loading model {model_path} for dataset {dataset_str}")
    
#     if dataset_str != DatasetsEnum.IMAGENET.value and dataset_str != DatasetsEnum.CIFAR100.value:
#         raise ValueError(f"Invalid dataset: {dataset_str}")
    
#     # Always load from S3
#     if model_path.startswith('s3://'):
#         # S3 path - extract bucket and key
#         parts = model_path.replace('s3://', '').split('/', 1)
#         bucket = parts[0]
#         s3_key = parts[1]
#         model = load_model_from_s3(bucket, s3_key)
#         effective_path = model_path
#     else:
#         # S3 key without s3:// prefix
#         model = load_model_from_s3(S3_BUCKET, model_path)
#         effective_path = f"s3://{S3_BUCKET}/{model_path}"
    
#     print(f"Model {effective_path} has been loaded")
    
#     return Model(
#         model, 
#         dataset_config["top_k"], 
#         dataset_config["min_confidence"], 
#         effective_path, 
#         dataset_config["dataset"]
#     )
    
# ### S3 implementation ### 
# @contextmanager
# def streaming_zip_extraction(bucket_name: str, s3_key: str):
#     """Stream a zip file from S3 and extract it to a temporary directory"""
#     with tempfile.TemporaryDirectory() as temp_dir:
#         s3_client = boto3.client('s3')
#         try:
#             response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
#             buffer = io.BytesIO(response['Body'].read())
#             with zipfile.ZipFile(buffer) as zip_ref:
#                 zip_ref.extractall(temp_dir)
#             yield temp_dir
#         except ClientError as e:
#             logger.error(f"Error streaming zip from S3 ({bucket_name}/{s3_key}): {e}")
#             raise

# def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
#     """Check if a file exists in S3"""
#     s3_client = boto3.client('s3')
#     try:
#         s3_client.head_object(Bucket=bucket_name, Key=s3_key)
#         return True
#     except ClientError:
#         return False

# def read_json_from_s3(bucket_name: str, s3_key: str) -> Any:
#     """Read a JSON file from S3"""
#     s3_client = boto3.client('s3')
#     try:
#         response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
#         content = response['Body'].read().decode('utf-8')
#         return json.loads(content)
#     except ClientError as e:
#         logger.error(f"Error reading JSON from S3 ({bucket_name}/{s3_key}): {e}")
#         raise

# def load_model_from_s3(bucket_name: str, s3_key: str):
#     """Load Keras model from S3"""
#     # Check if it's a .keras file
#     if s3_key.endswith('.keras'):
#         with streaming_zip_extraction(bucket_name, s3_key) as temp_dir:
#             model = tf.keras.models.load_model(temp_dir)
#             return model
#     else:
#         # Assume it's a SavedModel directory
#         with tempfile.TemporaryDirectory() as temp_dir:
#             s3_client = boto3.client('s3')
#             # List all objects with the prefix
#             paginator = s3_client.get_paginator('list_objects_v2')
#             pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_key)
            
#             for page in pages:
#                 if 'Contents' not in page:
#                     continue
                
#                 for obj in page['Contents']:
#                     # Get relative path
#                     key = obj['Key']
#                     relative_path = key[len(s3_key):].lstrip('/')
#                     if not relative_path:
#                         continue
                    
#                     # Create local directory structure
#                     local_path = os.path.join(temp_dir, relative_path)
#                     os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
#                     # Download file
#                     s3_client.download_file(bucket_name, key, local_path)
            
#             model = tf.keras.models.load_model(temp_dir)
#             return model
        

# # ### S3 implementation ### 
# def construct_model(model_path: str, dataset_config: Dict[str, Any]) -> Model:
#     """Load model from S3 and construct Model object"""
#     logger.info(f"Loading model {model_path} for dataset {dataset_config['dataset']}")
    
#     # Check if model_path is an S3 path or just a key
#     if model_path.startswith('s3://'):
#         bucket_name = model_path.replace('s3://', '').split('/')[0]
#         s3_key = '/'.join(model_path.replace('s3://', '').split('/')[1:])
#     else:
#         bucket_name = S3_BUCKET
#         s3_key = model_path
    
#     # Check if model exists in S3
#     if not s3_file_exists(bucket_name, s3_key):
#         raise FileNotFoundError(f'Model s3://{bucket_name}/{s3_key} does not exist')
    
#     # Load model from S3
#     resnet_model = load_model_from_s3(bucket_name, s3_key)
    
#     return Model(
#         resnet_model, 
#         dataset_config["top_k"], 
#         dataset_config["min_confidence"],
#         f"s3://{bucket_name}/{s3_key}", 
#         dataset_config["dataset"]
#     )
    
# def delete_model():
#     return None


# # ### S3 implementation ### 
# def query_model(top_label, model_id, graph_type, user):
#     """Query model using dendrogram from S3"""
#     model_info = get_user_models_info(user, model_id)
#     if model_info is None:
#         raise ValueError(f"Model ID {model_id} not found in models.json")
#     else:
#         model_files = get_model_files(user.get_user_folder(), model_info, graph_type)
#         model_path = model_files["model_graph_folder"]
#         dendrogram_s3_key = f'{model_path}/dendrogram'
    
#     # We need to modify the Dendrogram class to work with S3
#     # For now, let's download the dendrogram file to a temporary location
#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_dendrogram = os.path.join(temp_dir, 'dendrogram')
#         s3_client = boto3.client('s3')
        
#         # Download dendrogram files if they exist
#         dendrogram_json_key = f"{dendrogram_s3_key}.json"
#         dendrogram_pkl_key = f"{dendrogram_s3_key}.pkl"
        
#         if s3_file_exists(S3_BUCKET, dendrogram_json_key):
#             s3_client.download_file(S3_BUCKET, dendrogram_json_key, f"{temp_dendrogram}.json")
        
#         if s3_file_exists(S3_BUCKET, dendrogram_pkl_key):
#             s3_client.download_file(S3_BUCKET, dendrogram_pkl_key, f"{temp_dendrogram}.pkl")
        
#         dendrogram = Dendrogram(temp_dendrogram)
#         dendrogram.load_dendrogram()
        
#         consistency = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
#         if consistency is None:
#             raise ValueError(f"Label {top_label} not found in dendrogram")
#         else:
#             logger.debug(f"Top label: {top_label}")
#             logger.debug(f"Consistency: {consistency}")
        
#         return consistency


# # ### S3 implementation ### 
# # def query_predictions(model_id, graph_type, image, user):
# #     """Query predictions using model from S3"""
# #     model_info = get_user_models_info(user, model_id)
# #     if model_info is None:
# #         raise ValueError(f"Model ID {model_id} not found in models.json")
# #     else:
# #         model_files = get_model_files(user.get_user_folder(), model_info, graph_type)

# #     dataset = model_info["dataset"]
# #     labels = get_dataset_labels(dataset)
# #     model_s3_key = model_files["model_file"]
    
# #     if s3_file_exists(S3_BUCKET, model_s3_key):
# #         # Load model from S3
# #         current_model = load_model_from_s3(S3_BUCKET, model_s3_key)
# #         logger.debug(f"Model loaded successfully from 's3://{S3_BUCKET}/{model_s3_key}'.")
# #     else:
# #         raise ValueError(f"Model file s3://{S3_BUCKET}/{model_s3_key} does not exist")
    
# #     preprocessed_image = preprocess_loaded_image(current_model, image)
# #     predictions = get_top_k_predictions(current_model, preprocessed_image, labels)
# #     top_label = predictions[0][0]  # Top label
# #     top_3_predictions = predictions[:3]  # Top 3 predictions
    
# #     return top_label, top_3_predictions


# # ### S3 implementation ### 
# def get_user_models_info(user, model_id):
#     """Get model info from models.json in S3"""
#     # Assuming user object has a method to get the models.json path in S3
#     # If not, we'll need to construct it
#     s3_models_json_key = f"{user.get_user_folder()}/models.json"
    
#     if s3_file_exists(S3_BUCKET, s3_models_json_key):
#         models_data = read_json_from_s3(S3_BUCKET, s3_models_json_key)
#         logger.debug(f"Models data loaded from s3://{S3_BUCKET}/{s3_models_json_key}")
#     else:
#         models_data = []
#         raise ValueError(f"Models metadata file 's3://{S3_BUCKET}/{s3_models_json_key}' not found.")
    
#     if model_id is None:
#         return models_data
#     else:
#         return get_model_info(models_data, model_id)


# ### S3 implementation ### 
# def get_model_files(user_folder: str, model_info: dict, graph_type: str):
#     """Get model file paths in S3"""
#     logger.info(f"Getting model files for user folder: {user_folder}, model info: {model_info}, graph type: {graph_type}")
    
#     # Construct S3 paths
#     model_subfolder = f"{user_folder}/{model_info['model_id']}"
#     model_file = f"{model_subfolder}/{model_info['file_name']}"
    
#     if not s3_file_exists(S3_BUCKET, model_file):
#         model_file = None
#         raise ValueError(f"Model file s3://{S3_BUCKET}/{model_file} does not exist")
    
#     model_graph_folder = f"{model_subfolder}/{graph_type}"
    
#     # Check if the graph folder "exists" by listing objects with this prefix
#     s3_client = boto3.client('s3')
#     response = s3_client.list_objects_v2(
#         Bucket=S3_BUCKET,
#         Prefix=model_graph_folder,
#         MaxKeys=1
#     )
    
#     if 'Contents' not in response:
#         model_graph_folder = None
#         raise ValueError(f"Model graph folder s3://{S3_BUCKET}/{model_graph_folder} does not exist")
    
#     # Define file paths
#     Z_file = f"{model_graph_folder}/dendrogram.pkl"
#     Z_file_exists = s3_file_exists(S3_BUCKET, Z_file)
#     if not Z_file_exists:
#         Z_file = None
#         logger.debug(f"Z file s3://{S3_BUCKET}/{Z_file} does not exist")
    
#     dendrogram_file = f"{model_graph_folder}/dendrogram.json"
#     dendrogram_file_exists = s3_file_exists(S3_BUCKET, dendrogram_file)
#     if not dendrogram_file_exists:
#         dendrogram_file = None
#         logger.debug(f"Dendrogram file s3://{S3_BUCKET}/{dendrogram_file} does not exist")
    
#     detector_filename = f"{model_graph_folder}/logistic_regression_model.pkl"
#     detector_file_exists = s3_file_exists(S3_BUCKET, detector_filename)
#     if not detector_file_exists:
#         detector_filename = None
#         logger.debug(f"Detector model file s3://{S3_BUCKET}/{detector_filename} does not exist")
    
#     dataframe_filename = f"{model_graph_folder}/edges_df.csv"
#     dataframe_file_exists = s3_file_exists(S3_BUCKET, dataframe_filename)
#     if not dataframe_file_exists:
#         dataframe_filename = None
#         logger.debug(f"Dataframe file s3://{S3_BUCKET}/{dataframe_filename} does not exist")
    
#     return {
#         "model_file": model_file, 
#         "Z_file": Z_file, 
#         "dendrogram": dendrogram_file, 
#         "detector_filename": detector_filename, 
#         "dataframe": dataframe_filename, 
#         "model_graph_folder": model_graph_folder
#     }



from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import zipfile
import io
import boto3
from botocore.exceptions import ClientError
from contextlib import contextmanager
from typing import Dict, Any, Optional
# import tensorflow_io as tfio  
import tensorflow as tf 
import logging
from NMA.classes.model import Model
from NMA.classes.dendrogram import Dendrogram
from NMA.utilss.photos_utils import preprocess_loaded_image
from NMA.dataset_service import get_dataset_labels
# from fastapi import HTTPException, status
import json
from NMA.utilss.enums.datasets_enum import DatasetsEnum
import shutil
import time
import tempfile
from NMA.utilss.s3_utils import get_users_s3_client

# Set up logging
logger = logging.getLogger(__name__)

# S3 Configuration - Always use S3
S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
if not S3_BUCKET:
    raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")

def get_top_k_predictions(model, image, class_names, top_k=5):
    # Get predictions from the model
    predictions = model.predict(image)

    # Flatten the predictions array if it's 2D (e.g., shape (1, num_classes))
    if len(predictions.shape) == 2:
        predictions = predictions[0]  # Extract the first (and only) batch

    # Get the indices of the top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    # Get the top-k probabilities and corresponding labels
    top_probs = predictions[top_indices]
    top_labels = [class_names[i] for i in top_indices]

    logger.debug(f"Top {top_k} predictions: {list(zip(top_labels, top_probs))}")

    return list(zip(top_labels, top_probs))

def get_model():
    return None

def _check_model_path(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
    if user_id is None:
        raise ("user_id is required")
    
    if model_id is None:
        raise ("model_id is required")
    
    if graph_type is None:
        raise ValueError("graph_type is required")
    
    logger.info(f"Checking model path for user_id: {user_id}, model_id: {model_id}, graph_type: {graph_type}")

    model_path = _get_model_path(user_id, model_id)

    logger.debug(f"Model path: {model_path}")

    if model_path is None:
        raise ValueError("Could not find model directory")
    return model_path

    
def delete_model():
    return None


# ### original implemetation ###
# def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
#     # Construct the model path based on user_id and model_id
#     BASE_DIR = os.getenv("USERS_PATH", "users")
#     model_path = os.path.join(BASE_DIR, str(user_id), str(model_id))
    
#     # Check if the model path exists
#     if os.path.exists(model_path):
#         return model_path
#     else:
#         logger.debug(f"Model path {model_path} does not exist")
#         return None



## S3 implementation ### 
def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
    """
    Get the model path in S3 based on user_id and model_id
    """
    s3_client = get_users_s3_client()
    
    # Construct the S3 prefix (equivalent to directory path in S3)
    s3_prefix = f"{str(user_id)}/{str(model_id)}"
    
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
    

# ### original implemetation ###
# def _get_model_filename(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
#     model_path = _get_model_path(user_id, model_id)
#     if model_path is None:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
#             detail="Could not find model directory"
#         )
#     for file_name in os.listdir(model_path):
#         if file_name.endswith(".keras"):
#             model_filename = os.path.join(model_path, file_name)
#             return model_filename

# current_user.user_id, model_id, dataset, graph_type, min_confidence, top_k

## S3 implementation ### 
def _get_model_filename(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
    """Get S3 model filename"""
    model_path = _get_model_path(user_id, model_id)
    if model_path is None:
        raise ValueError("Could not find model directory")
    
    s3_client =  get_users_s3_client() 
    
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
        
# ### original implemetation ###
# def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
#     logger.info(f"Loading model {model_path} for dataset {dataset_str}")

#     if dataset_str != DatasetsEnum.IMAGENET.value and dataset_str != DatasetsEnum.CIFAR100.value:
#         raise ValueError(f"Invalid dataset: {dataset_str}")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f'Model {model_path} does not exist')
    
#     model = tf.keras.models.load_model(model_path)
    
#     print(f"Model {model_path} has been loaded")
#     return Model(
#         model, 
#         dataset_config["top_k"], 
#         dataset_config["min_confidence"], 
#         model_path, 
#         dataset_config["dataset"]
#     )

## S3 implementation ### 
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
    
        
def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
    """Check if a file exists in S3"""
    s3_client =  get_users_s3_client() 
    print(f"Checking if file exists in S3: {bucket_name}/{s3_key}")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File found: {bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"File not found: {bucket_name}/{s3_key}, Error: {str(e)}")
        return False

def read_json_from_s3(bucket_name: str, s3_key: str) -> Any:
    """Read a JSON file from S3"""
    s3_client =  get_users_s3_client() 
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        logger.error(f"Error reading JSON from S3 ({bucket_name}/{s3_key}): {e}")
        raise

# def load_model_from_s3(bucket_name: str, s3_key: str):
#     """Load Keras model from S3"""
#     # Check if s3_key is actually a full S3 URI
#     if s3_key.startswith('s3://'):
#         parts = s3_key.replace('s3://', '').split('/', 1)
#         bucket = parts[0]
#         key = parts[1] if len(parts) > 1 else ''
#     else:
#         bucket = bucket_name
#         key = s3_key
    
#     s3_client =  get_users_s3_client() 
    
#     # Use a temporary directory instead of a temporary file
#     # This approach is more reliable on Windows
#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_model_path = os.path.join(temp_dir, 'model.keras')
        
#         try:
#             # Download the model file to the temp directory
#             s3_client.download_file(bucket, key, temp_model_path)
            
#             # Load the model from the temp file
#             model = tf.keras.models.load_model(temp_model_path)
#             return model
            
#         except Exception as e:
#             logger.error(f"Error loading model from S3 ({bucket}/{key}): {e}")
#             raise



# batch implementation
def load_model_from_s3(bucket_name: str, s3_key: str):
    """Load Keras model from S3 with version compatibility handling"""
    # Check if s3_key is actually a full S3 URI
    if s3_key.startswith('s3://'):
        parts = s3_key.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        bucket = bucket_name
        key = s3_key
    
    s3_client = get_users_s3_client() 
    
    # Use a temporary directory instead of a temporary file
    # This approach is more reliable on Windows
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'model.keras')
        
        try:
            # Download the model file to the temp directory
            logger.info(f"Downloading model from S3: {bucket}/{key}")
            s3_client.download_file(bucket, key, temp_model_path)
            
            # First try loading with custom object scope to handle version differences
            try:
                logger.info("Attempting to load model with custom object scope...")
                with tf.keras.utils.custom_object_scope({'Functional': tf.keras.Model}):
                    model = tf.keras.models.load_model(temp_model_path, compile=False)
                logger.info("Model loaded successfully with custom object scope")
                return model
            except Exception as custom_error:
                logger.warning(f"Could not load with custom object scope: {str(custom_error)}")
                
                # Check if this is a keras.src module error
                if 'keras.src.models.functional' in str(custom_error):
                    logger.warning("This model was saved with TensorFlow 2.13+ but you're using an older version")
                    logger.warning("Attempting alternative loading approaches...")
                    
                    # Try using a fallback model
                    if key.lower().find('resnet50') >= 0:
                        logger.info("Creating ResNet50 fallback model...")
                        fallback_model = tf.keras.applications.ResNet50(weights=None)
                        logger.info("Using ResNet50 fallback model")
                        return fallback_model
                    else:
                        # For other models, try a different loading approach
                        try:
                            # Try loading with h5py directly if it's an H5 file
                            import h5py
                            if temp_model_path.endswith(('.h5', '.keras')):
                                logger.info("Trying to load model weights directly...")
                                with h5py.File(temp_model_path, 'r') as h5file:
                                    # Check if it's a weights-only file
                                    if 'model_weights' in h5file:
                                        # Create a base model with matching architecture
                                        logger.info("Found weights file, creating compatible model...")
                                        base_model = tf.keras.Sequential([
                                            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                                            tf.keras.applications.ResNet50(include_top=True, weights=None)
                                        ])
                                        base_model.load_weights(temp_model_path)
                                        logger.info("Model weights loaded successfully")
                                        return base_model
                        except Exception as h5_error:
                            logger.warning(f"Could not load weights directly: {str(h5_error)}")
                    
                    # Last resort: create a new model from scratch
                    logger.warning("All loading attempts failed. Creating a new model as fallback.")
                    logger.warning("Please upgrade to TensorFlow 2.13+ to properly load this model.")
                    return tf.keras.applications.ResNet50(weights='imagenet')
                
                # If it's not a keras.src error or fallbacks failed, try standard loading
                logger.info("Attempting standard model loading...")
                model = tf.keras.models.load_model(temp_model_path, compile=False)
                return model
                
        except Exception as e:
            logger.error(f"Error loading model from S3 ({bucket}/{key})")
        

# ### original implemetation ###
# def construct_model(model_path: str, dataset_config: Dict[str, Any]) -> Model:
#         logger.info(f"Loading model {model_path} for dataset {dataset_config['dataset']}")

#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f'Model {model_path} does not exist')
#         resnet_model = tf.keras.models.load_model(model_path)
#         return Model(resnet_model, dataset_config["top_k"], dataset_config["min_confidence"], model_path, dataset_config["dataset"])




### S3 implementation ### 
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
    
    
    



## original implemetation ###
# def query_model(top_label, model_id, graph_type, user):
#     model_info = get_user_models_info(user, model_id)
#     if model_info is None:
#         raise ValueError(f"Model ID {model_id} not found in models.json")
#     else:
#         model_files = get_model_files(user.get_user_folder(), model_info, graph_type)
#         model_path = model_files["model_graph_folder"]
#         dendrogram_filename = f'{model_path}/dendrogram'
#     dendrogram = Dendrogram(dendrogram_filename)
#     dendrogram.load_dendrogram()
#     consistensy = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
#     if consistensy is None:
#         raise ValueError(f"Label {top_label} not found in dendrogram")
#     else:
#         logger.debug(f"Top label: {top_label}")
#         logger.debug(f"Consistency: {consistensy}")
#     return consistensy


# ### S3 implementation ### 
def query_model(top_label, model_id, graph_type, user):
    """Query model using dendrogram from S3 without local disk usage"""
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user.get_user_folder(), model_info, graph_type)
        model_path = model_files["model_graph_folder"]
        dendrogram_s3_key = f'{model_path}/dendrogram'
    
    # Initialize S3 client
    s3_client =  get_users_s3_client() 
    
    # Create S3 parameters for Dendrogram class
    s3_params = {
        "s3_client": s3_client,
        "s3_bucket": S3_BUCKET,
        "s3_key_prefix": dendrogram_s3_key
    }
    
    # Use the Dendrogram class with S3 support
    dendrogram = Dendrogram(dendrogram_s3_key)
    dendrogram.load_dendrogram()
    
    consistency = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
    if consistency is None:
        raise ValueError(f"Label {top_label} not found in dendrogram")
    else:
        logger.debug(f"Top label: {top_label}")
        logger.debug(f"Consistency: {consistency}")
    
    return consistency
    
    
## original implemetation ###
# def query_predictions(model_id, graph_type, image, user):
#     model_info = get_user_models_info(user, model_id)
#     if model_info is None:
#         raise ValueError(f"Model ID {model_id} not found in models.json")
#     else:
#         model_files = get_model_files(user.get_user_folder(), model_info, graph_type)

#     dataset = model_info["dataset"]
#     labels = get_dataset_labels(dataset)
#     model_filename = model_files["model_file"]
#     if os.path.exists(model_filename):
#         current_model = tf.keras.models.load_model(model_filename)
#         logger.debug(f"Model loaded successfully from '{model_filename}'.")
#     else:
#         raise ValueError(f"Model file {model_filename} does not exist")
#     preprocessed_image = preprocess_loaded_image(current_model, image)
#     predictions = get_top_k_predictions(current_model, preprocessed_image, labels)
#     top_label = predictions[0][0]  # Top label
#     top_3_predictions = predictions[:3]  # Top 3 predictions
#     return top_label, top_3_predictions


# ### S3 implementation ### 
def query_predictions(model_id, graph_type, image, user):
    """Query predictions using model from S3"""
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user.get_user_folder(), model_info, graph_type)

    dataset = model_info["dataset"]
    labels = get_dataset_labels(dataset)
    model_s3_key = model_files["model_file"]
    
    if s3_file_exists(S3_BUCKET, model_s3_key):
        # Load model from S3
        current_model = load_model_from_s3(S3_BUCKET, model_s3_key)
        logger.debug(f"Model loaded successfully from 's3://{S3_BUCKET}/{model_s3_key}'.")
    else:
        raise ValueError(f"Model file s3://{S3_BUCKET}/{model_s3_key} does not exist")
    
    preprocessed_image = preprocess_loaded_image(current_model, image)
    predictions = get_top_k_predictions(current_model, preprocessed_image, labels)
    top_label = predictions[0][0]  # Top label
    top_3_predictions = predictions[:3]  # Top 3 predictions
    
    return top_label, top_3_predictions




## original implemetation ###
# def get_user_models_info(user, model_id):
#     models_json_path = user.get_models_json_path()

#     if os.path.exists(models_json_path):
#         with open(models_json_path, "r") as json_file:
#             models_data = json.load(json_file)
#         logger.debug(f"Models data loaded from {models_json_path}")
#     else:
#         models_data = []
#         raise ValueError(f"Models metadata file '{models_json_path}' not found.")
    
#     if model_id is None:
#         return models_data
#     else:
#         return get_model_info(models_data, model_id)


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


## original implemetation ###
def get_model_info(models_data, model_id):
    for model in models_data:
        if str(model["model_id"]) == str(model_id):
            return {
                "model_id": model["model_id"],
                "file_name": model["file_name"],
                "dataset": model["dataset"],
                "graph_type": model["graph_type"],
            }
            
    # If no match is found, return None
    logger.debug(f"Model ID {model_id} not found in models data.")
    return None



### original implemetation ###
# def get_model_files(user_folder: str, model_info: dict, graph_type: str):
#         logger.info(f"Getting model files for user folder: {user_folder}, model info: {model_info}, graph type: {graph_type}")

#         model_subfolder = os.path.join(user_folder, model_info["model_id"])
#         model_file = os.path.join(model_subfolder, model_info["file_name"])
#         if not os.path.exists(model_file):
#             model_file = None
#             raise ValueError(f"Model file {model_file} does not exist")
        
#         model_graph_folder = os.path.join(model_subfolder, graph_type)
#         if not os.path.exists(model_graph_folder):
#             model_graph_folder = None
#             raise ValueError(f"Model graph folder {model_graph_folder} does not exist")

#         Z_file = os.path.join(model_graph_folder, "dendrogram.pkl")
#         if not os.path.exists(Z_file):
#             Z_file = None
#             logger.debug(f"Z file {Z_file} does not exist")

#         dendrogram_file = os.path.join(model_graph_folder, 'dendrogram.json')
#         if not os.path.exists(dendrogram_file):
#             dendrogram_file = None
#             logger.debug(f"Dendrogram file {dendrogram_file} does not exist")

#         detector_filename = os.path.join(model_graph_folder, 'logistic_regression_model.pkl')
#         if not os.path.exists(detector_filename):
#             detector_filename = None
#             logger.debug(f"Detector model file {detector_filename} does not exist")

#         dataframe_filename = os.path.join(model_graph_folder, 'edges_df.csv')
#         if not os.path.exists(dataframe_filename):
#             dataframe_filename = None
#             logger.debug(f"Dataframe file {dataframe_filename} does not exist")

#         return {"model_file": model_file, "Z_file": Z_file, "dendrogram": dendrogram_file, "detector_filename": detector_filename, "dataframe": dataframe_filename, "model_graph_folder": model_graph_folder}
        

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
    s3_client = get_users_s3_client() 
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
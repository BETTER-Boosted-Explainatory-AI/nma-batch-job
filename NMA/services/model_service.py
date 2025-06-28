from dotenv import load_dotenv
load_dotenv()
import os
from botocore.exceptions import ClientError
from typing import Dict, Any, Optional
import tensorflow as tf 
import logging
from NMA.classes.model import Model
import json
from NMA.utilss.enums.datasets_enum import DatasetsEnum
import tempfile
from NMA.utilss.s3_utils import get_users_s3_client
logger = logging.getLogger(__name__)
S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
if not S3_BUCKET:
    raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")


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


def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
    s3_client = get_users_s3_client()
    s3_prefix = f"{str(user_id)}/{str(model_id)}"
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=s3_prefix,
            MaxKeys=1 
        )

        print(f"Checking S3 prefix: {s3_prefix} in bucket: {S3_BUCKET}")
        print("object list response: ",response)

        if 'Contents' in response:
            return s3_prefix
        else:
            logger.debug(f"S3 prefix {s3_prefix} does not exist in bucket {S3_BUCKET}")
            return None
    except ClientError as e:
        logger.error(f"Error checking S3 prefix {s3_prefix}: {e}")
        return None
    

def _get_model_filename(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
    model_path = _get_model_path(user_id, model_id)
    if model_path is None:
        raise ValueError("Could not find model directory")

    s3_client = get_users_s3_client() 
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

def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
    logger.info(f"Loading model {model_path} for dataset {dataset_str}")
    if dataset_str != DatasetsEnum.IMAGENET.value and dataset_str != DatasetsEnum.CIFAR100.value:
        raise ValueError(f"Invalid dataset: {dataset_str}")
    
    if model_path.startswith('s3://'):
        parts = model_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        s3_key = parts[1]
        model = load_model_from_s3(bucket, s3_key)
        effective_path = model_path
    else:
        model = load_model_from_s3(S3_BUCKET, model_path)
        effective_path = f"{S3_BUCKET}/{model_path}"
    
        
    return Model(
        model, 
        dataset_config["top_k"], 
        dataset_config["min_confidence"], 
        effective_path, 
        dataset_config["dataset"]
    )
    
        
def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
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
    s3_client =  get_users_s3_client() 
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        logger.error(f"Error reading JSON from S3 ({bucket_name}/{s3_key}): {e}")
        raise

def load_model_from_s3(bucket_name: str, s3_key: str):
    """Load Keras model from S3 with version compatibility handling"""
    if s3_key.startswith('s3://'):
        parts = s3_key.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        bucket = bucket_name
        key = s3_key
    
    s3_client = get_users_s3_client() 
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'model.keras')
        try:
            logger.info(f"Downloading model from S3: {bucket}/{key}")
            s3_client.download_file(bucket, key, temp_model_path)

            model = tf.keras.models.load_model(temp_model_path)
            return model
                
        except Exception as e:
            logger.error(f"Error loading model from S3 ({bucket}/{key}): {str(e)}")
            raise
    
# # ### S3 implementation ### 
def get_user_models_info(user_id, model_id):
    """Get model info from models.json in S3"""
    # Assuming user object has a method to get the models.json path in S3
    # If not, we'll need to construct it
    s3_models_json_key = f"{user_id}/models.json"
    
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


def get_model_info(models_data, model_id):
    for model in models_data:
        if str(model["model_id"]) == str(model_id):
            return {
                "model_id": model["model_id"],
                "file_name": model["file_name"],
                "dataset": model["dataset"],
                "graph_type": model["graph_type"],
            }
    logger.debug(f"Model ID {model_id} not found in models data.")
    return None

def get_model_files(user_folder: str, model_info: dict, graph_type: str):
    """Get model file paths in S3"""
    logger.info(f"Getting model files for user folder: {user_folder}, model info: {model_info}, graph type: {graph_type}")
    model_subfolder = f"{user_folder}/{model_info['model_id']}"
    model_file = f"{model_subfolder}/{model_info['file_name']}"
    
    if not s3_file_exists(S3_BUCKET, model_file):
        model_file = None
        raise ValueError(f"Model file s3://{S3_BUCKET}/{model_file} does not exist")
    model_graph_folder = f"{model_subfolder}/{graph_type}"
    
    s3_client = get_users_s3_client() 
    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=model_graph_folder,
        MaxKeys=1
    )
    
    if 'Contents' not in response:
        model_graph_folder = None
        raise ValueError(f"Model graph folder s3://{S3_BUCKET}/{model_graph_folder} does not exist")
    
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
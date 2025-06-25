
import os
# import boto3
# from fastapi import UploadFile
import json
import uuid
import io
import numpy as np
import tensorflow as tf
from ..utilss.photos_utils import preprocess_numpy_image
from ..utilss.s3_utils import get_users_s3_client
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

def _update_model_metadata(current_user, model_id, model_filename, dataset, graph_type, min_confidence, top_k, job_id, job_status="submitted"):
    from ..utilss.s3_utils import get_users_s3_client
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    user_folder = current_user.get_user_folder()
    models_json_path = f"{user_folder}.json"
    
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=models_json_path)
        models_data = json.loads(response['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        models_data = []
    if model_filename is None:
        for model in models_data:
            if model.get("model_id") == model_id:
                model_filename = model.get("file_name")
                break

    job_metadata = {
        "job_id": job_id,
        "job_graph_type": graph_type,
        "job_status": job_status,
        "timestamp": datetime.utcnow().isoformat()
    }

    return save_model_metadata(
        models_data,
        models_json_path,
        model_id,
        model_filename,
        dataset,
        graph_type,
        min_confidence,
        top_k,
        job_metadata
    )


def save_model_metadata(
    models_data, 
    models_json_path, 
    model_id, 
    model_filename, 
    dataset, 
    graph_type, 
    min_confidence, 
    top_k, 
    job_metadata=None
) -> bool:  
    
    from ..utilss.s3_utils import get_users_s3_client
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    model_metadata = {
        "model_id": model_id,
        "file_name": model_filename,
        "dataset": dataset,
        "graph_type": [graph_type],
        "min_confidence": min_confidence,
        "top_k": top_k,
        "batch_jobs": [job_metadata] if job_metadata else []
    }

    for model in models_data:
        if model["model_id"] == model_id:
            if not isinstance(model["graph_type"], list):
                model["graph_type"] = [model["graph_type"]]

            if graph_type not in model["graph_type"]:
                model["graph_type"].append(graph_type)
                print(f"Adding '{graph_type}' to graph_type for file '{model_filename}'.")
                
            else:
                raise ValueError(f"Graph type '{graph_type}' for model ID '{model_id}' already exists.")
            if "batch_jobs" not in model or not isinstance(model["batch_jobs"], list):
                model["batch_jobs"] = []
            if job_metadata:
                model["batch_jobs"].append(job_metadata)
            break
 
        if "batch_jobs" not in model or not isinstance(model["batch_jobs"], list):
            model["batch_jobs"] = []
            if "batch_jobs" not in model or not isinstance(model["batch_jobs"], list):
                model["batch_jobs"] = []
            if job_metadata:
                model["batch_jobs"].append(job_metadata)
            break
    else:
        models_data.append(model_metadata)
        print(
            f"Adding new metadata for file '{model_filename}' with graph type '{graph_type}'."
        )
    models_json = json.dumps(models_data, indent=4)
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=models_json_path,
        Body=models_json)
    return True


def check_models_metadata(models_data, model_id, graph_type):
    for model in models_data:
        if model.get("model_id") == model_id:
            if graph_type in model.get("graph_type", []):
                raise ValueError(f"Graph type '{graph_type}' for model ID '{model_id}' already exists.")
            return model_id
    return str(uuid.uuid4())


def user_has_job_running(current_user):
    current_user.load_models()
    for model in current_user.models:
        batch_jobs = model.get("batch_jobs", [])
        for job in batch_jobs:
            if job.get("job_status") not in ("succeeded", "failed"):
                return True
    return False


def load_numpy_from_directory(model, source):
    images = []
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if isinstance(source, str):  # Case 1: S3 path or directory path
        print(f"Loading images from source: {source}")
        if s3_bucket and source.startswith(f"{s3_bucket}/"):
            prefix = source.replace(f"{s3_bucket}/", "")
            response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
            
            for item in response.get('Contents', []):
                if item['Key'].endswith(".npy"):
                    file_obj = s3_client.get_object(Bucket=s3_bucket, Key=item['Key'])
                    file_content = file_obj['Body'].read()
                    image = np.load(io.BytesIO(file_content))
                    preprocess_image = preprocess_numpy_image(model, image)
                    images.append(preprocess_image)
        else:
            for filename in os.listdir(source):
                if filename.endswith(".npy"):
                    file_path = os.path.join(source, filename)
                    image = np.load(file_path)
                    preprocess_image = preprocess_numpy_image(model, image)
                    images.append(preprocess_image)

    elif isinstance(source, list): 
        print(f"Loading images from user-provided files: {len(source)} files")
        for upload_file in source:
            if upload_file.filename.endswith(".npy"):
                file_content = upload_file.file.read()
                image = np.load(io.BytesIO(file_content))
                preprocess_image = preprocess_numpy_image(model, image)
                images.append(preprocess_image)

    else:
        raise ValueError("Source must be a directory path (str) or a list of UploadFile objects.")

    return images


def load_raw_image(file_path):
    img_example = np.load(file_path)
    return tf.convert_to_tensor(img_example, dtype=tf.float32)


def user_has_job_running(current_user):
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    user_folder = current_user.get_user_folder()
    models_json_path = f"{user_folder}.json"
    
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=models_json_path)
        models_data = json.loads(response['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        return False
    
    for model in models_data:
        batch_jobs = model.get("batch_jobs", [])
        for job in batch_jobs:
            if job.get("job_status") not in ("succeeded", "failed"):
                return True
    return False


def update_current_model(user_id, model_id, graph_type, model_filename, dataset, min_confidence, top_k):
    if hasattr(dataset, 'dataset'):
        dataset_name = dataset.dataset
    elif hasattr(dataset, '__class__'):
        dataset_name = dataset.__class__.__name__.lower()
    else:
        dataset_name = str(dataset)
    
    model_metadata = {
        "model_id": model_id,
        "file_name": model_filename,
        "dataset": dataset,
        "graph_type": graph_type,
        "min_confidence": min_confidence,
        "top_k": top_k
    }
    
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    models_json_key = f"{user_id}/models.json"
    
    try:
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=models_json_key)
            models_data = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            models_data = []
        
        model_found = False
        for i, model in enumerate(models_data):
            if str(model.get("model_id")) == str(model_id):
                models_data[i] = model_metadata
                model_found = True
                break
        if not model_found:
            models_data.append(model_metadata)
        
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=models_json_key,
            Body=json.dumps(models_data, indent=4),
            ContentType='application/json'
        )
        logger.info(f"Updated current model for user {user_id}")
        
    except Exception as e:
        logging.error(f"Error updating job status: {e}")
        raise

def update_job_status(user_id, model_id, new_status):
    job_id = os.getenv("AWS_BATCH_JOB_ID")
    print("This job's ID is:", job_id)

    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    # Define models.json key
    models_json_key = f"{user_id}/models.json"
    
    try:
        # Try to get existing models.json
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=models_json_key)
            models_data = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            # Create new models.json if it doesn't exist
            models_data = []
        
        # Update the job status
        updated = False
        for model in models_data:
            if model['model_id'] == model_id:
                for job in model.get('batch_jobs', []):
                    if job['job_id'] == job_id:
                        job['job_status'] = new_status
                        updated = True
                        break

        if updated:
            # Write back to S3
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=models_json_key,
                Body=json.dumps(models_data, indent=4).encode('utf-8')
            )
            print(f"Updated job status for job_id={job_id} to '{new_status}' in {models_json_key}")
        else:
            print(f"No matching job_id={job_id} found for model_id={model_id}")

    except Exception as e:
        logging.error(f"Error updating job status: {e}")
        raise
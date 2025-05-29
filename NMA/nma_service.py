import boto3
import os
from botocore.exceptions import ClientError
from NMA.utilss.enums.graph_types import GraphTypes
from NMA.utilss.enums.datasets_enum import DatasetsEnum
from NMA.classes.dendrogram import Dendrogram
from NMA.classes.edges_dataframe import EdgesDataframe
from NMA.classes.nma import NMA
from NMA.dataset_service import _get_dataset_config, _load_dataset
from NMA.model_service import _get_model_filename, _load_model
from NMA.utilss.files_utils import update_current_model

# Get S3 bucket names from environment variables
S3_DATASETS_BUCKET_NAME = os.environ.get('S3_DATASETS_BUCKET_NAME', 'better-datasets')
S3_USERS_BUCKET_NAME = os.environ.get('S3_USERS_BUCKET_NAME', 'better-xai-users')

# Configure S3 client
s3_client = boto3.client('s3')

def _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k):
    if user_id is None:
        raise ValueError("user_id cannot be None")
    
    if model_id is None:
        raise ValueError("model_id cannot be None")
        
    if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
        raise ValueError("Graph type must be either 'similarity', 'dissimilarity', or 'count'")
    
    # Get S3 object key for the model
    model_key = _get_model_filename(user_id, model_id, graph_type)
    if model_key is None:
        raise ValueError("Model key cannot be None")
    
    # Create the full S3 URI for the model
    model_path = f"s3://{S3_USERS_BUCKET_NAME}/{model_key}"
    
    # Get dataset config from S3 or wherever the function is configured to look
    dataset_config = _get_dataset_config(dataset)
    
    
    # The following commented code would need to be updated similarly for S3 operations
    try:
        # Load model from S3
        loaded_model = _load_model(dataset, model_path, dataset_config)
        dataset_obj = _load_dataset(dataset)
        
        print("dataset_obj", type(dataset_obj))
        print("loaded_model", type(loaded_model))
        
        # # Get the S3 prefix for this model instead of a local directory
        # model_s3_prefix = f"{user_id}/models/{model_path}"
        
        # # Define S3 keys for dataframe and dendrogram instead of local paths
        # dataframe_s3_key = f"{model_s3_prefix}/{graph_type}/edges_df.csv"
        # dendrogram_s3_key = f"{model_s3_prefix}/{graph_type}/dendrogram"
        
        # labels = dataset_obj.labels
        # if dataset == DatasetsEnum.IMAGENET.value:
        #     labels = dataset_obj.directory_labels
        
        # nma = NMA(
        #     loaded_model.model,
        #     dataset_obj,
        #     labels,
        #     graph_type=graph_type,
        #     top_k=top_k,
        #     min_confidence=min_confidence           
        # )
        
        # # Save dataframe to S3
        # edges_df_obj = EdgesDataframe(model_id, dataframe_s3_key, nma.edges_df)
        # edges_df_obj.save_dataframe_to_s3(S3_USERS_BUCKET_NAME, dataframe_s3_key)
        # print(nma.Z)            
        
        # # Save dendrogram to S3
        # dendrogram = Dendrogram(dendrogram_s3_key, nma.Z)
        # dendrogram._build_tree_hierarchy(nma.Z, dataset_obj.labels)
        # dendrogram.save_dendrogram_to_s3(S3_USERS_BUCKET_NAME, dendrogram_s3_key, nma.Z)
        
        # init_sub_z = dendrogram.get_sub_dendrogram_formatted(dataset_config["init_selected_labels"])
        # print(init_sub_z)
        
        # # Update model metadata in database or S3
        # update_current_model(
        #     user_id,
        #     model_id,
        #     graph_type,
        #     os.path.basename(model_key),
        #     dataset,
        #     min_confidence,
        #     top_k,
        # )
        # return init_sub_z
    
    except Exception as e:
        import traceback
        print("🛠 Original NMA error:")
        traceback.print_exc()
        # either re-raise exactly as is:
        raise
    
    # # Extract just the filename from the S3 path
    # model_filename = os.path.basename(model_key)
    # print(model_filename)
    
    # return model_path
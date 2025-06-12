# import boto3
# import os
# from botocore.exceptions import ClientError
# from NMA.utilss.enums.graph_types import GraphTypes
# from NMA.utilss.enums.datasets_enum import DatasetsEnum
# from NMA.classes.dendrogram import Dendrogram
# from NMA.classes.edges_dataframe import EdgesDataframe
# from NMA.classes.nma import NMA
# from NMA.dataset_service import _get_dataset_config, _load_dataset
# from NMA.model_service import _get_model_filename, _load_model
# from NMA.utilss.files_utils import update_current_model
# from NMA.utilss.debug import assert_acyclic, CycleFound


# # Get S3 bucket names from environment variables
# S3_DATASETS_BUCKET_NAME = os.environ.get('S3_DATASETS_BUCKET_NAME', 'better-datasets')
# S3_USERS_BUCKET_NAME = os.environ.get('S3_USERS_BUCKET_NAME', 'better-xai-users')

# # Configure S3 client
# s3_client = boto3.client('s3')

# def _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k):
#     if user_id is None:
#         raise ValueError("user_id cannot be None")
    
#     if model_id is None:
#         raise ValueError("model_id cannot be None")
        
#     if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
#         raise ValueError("Graph type must be either 'similarity', 'dissimilarity', or 'count'")
    
#     # Get S3 object key for the model
#     model_key = _get_model_filename(user_id, model_id, graph_type)
#     if model_key is None:
#         raise ValueError("Model key cannot be None")
    
#     # Create the full S3 URI for the model
#     model_path = f"s3://{S3_USERS_BUCKET_NAME}/{model_key}"
    
#     # Get dataset config from S3 or wherever the function is configured to look
#     dataset_config = _get_dataset_config(dataset)
    
    
#     # The following commented code would need to be updated similarly for S3 operations
#     try:
#         # Load model from S3
#         loaded_model = _load_model(dataset, model_path, dataset_config)
#         dataset_obj = _load_dataset(dataset)
        
#         print("dataset_obj", type(dataset_obj))
#         print("loaded_model", type(loaded_model))
        
#         # Get the S3 prefix for this model instead of a local directory
#         model_s3_prefix = f"{user_id}/models/{model_path}"
        
#         # Define S3 keys for dataframe and dendrogram instead of local paths
#         dataframe_s3_key = f"{model_s3_prefix}/{graph_type}/edges_df.csv"
#         dendrogram_s3_key = f"{model_s3_prefix}/{graph_type}/dendrogram"
        
#         print("dataframe_s3_key", dataframe_s3_key)
#         print("dendrogram_s3_key", dendrogram_s3_key)
        
        
#         labels = dataset_obj.labels
#         if dataset == DatasetsEnum.IMAGENET.value:
#             labels = dataset_obj.directory_labels
        
        
#         nma = NMA(
#             loaded_model.model,
#             dataset_obj,
#             labels,
#             graph_type=graph_type,
#             top_k=top_k,
#             min_confidence=min_confidence           
#         )
        
#         # Save dataframe to S3
#         edges_df_obj = EdgesDataframe(model_id, dataframe_s3_key, nma.edges_df)
#         edges_df_obj.save_dataframe()
#         print(nma.Z)            
        
#         # Save dendrogram to S3
#         dendrogram = Dendrogram(dendrogram_s3_key, nma.Z)
#         dendrogram._build_tree_hierarchy(nma.Z, dataset_obj.labels)
#         dendrogram.save_dendrogram(nma.Z)
        
#         init_sub_z = dendrogram.get_sub_dendrogram_formatted(dataset_config["init_selected_labels"])
#         print(init_sub_z)
        
#         # Update model metadata in database or S3
#         update_current_model(
#             user_id,
#             model_id,
#             graph_type,
#             os.path.basename(model_key),
#             dataset,
#             min_confidence,
#             top_k,
#         )
#         return init_sub_z
    
#     except Exception as e:
#         import traceback
#         print("🛠 Original NMA error:")
#         traceback.print_exc()
#         # either re-raise exactly as is:
#         raise
    
#     # # Extract just the filename from the S3 path
#     # model_filename = os.path.basename(model_key)
#     # print(model_filename)
    
#     # return model_path



# create_nma.py
import os
import time
import sys
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Iterable

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm                           # pip install tqdm

from utilss.enums.graph_types import GraphTypes
from utilss.enums.datasets_enum import DatasetsEnum
from classes.dendrogram import Dendrogram
from classes.edges_dataframe import EdgesDataframe
from classes.nma import NMA
from services.dataset_service import _get_dataset_config, _load_dataset
from services.model_service import _get_model_filename, _load_model
from utilss.files_utils import update_current_model
from utilss.s3_utils import  get_users_s3_client


sys.setrecursionlimit(10_000)  
# --------------------------------------------------------------------------- #
# 1) LOGGING CONFIGURATION – ONE PLACE, EARLY
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()           # export LOG_LEVEL=DEBUG for more noise

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"_create_nma.log", mode="w"),
        logging.StreamHandler(),
    ],
)

for noisy in ("boto3", "botocore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 2) CONSTANTS
# --------------------------------------------------------------------------- #
S3_DATASETS_BUCKET_NAME = os.environ.get('S3_DATASETS_BUCKET_NAME', 'better-datasets')
S3_USERS_BUCKET_NAME    = os.environ.get('S3_USERS_BUCKET_NAME',    'better-xai-users')

s3_client = s3_client = get_users_s3_client


# --------------------------------------------------------------------------- #
# 3) UTILS
# --------------------------------------------------------------------------- #
@contextmanager
def timed(msg: str):
    """Context manager to log execution time of a code block."""
    t0 = time.perf_counter()
    logger.info("🔄 %s – started", msg)
    try:
        yield
    finally:
        logger.info("✅ %s – finished in %.2fs", msg, time.perf_counter() - t0)

def iterable_length(xs: Iterable):
    """Return len(xs) if xs supports len(); otherwise None."""
    try:
        return len(xs)
    except TypeError:
        return None

# --------------------------------------------------------------------------- #
# 4) MAIN FUNCTION
# --------------------------------------------------------------------------- #
# def _create_nma(
#     user_id: str,
#     model_id: str,
#     graph_type: str,
#     dataset: str,
#     min_confidence: float,
#     top_k: int,
# ):
#     """
#     Build graph + dendrogram for `model_id`, save artefacts to S3 and return
#     JSON with the initial sub-dendrogram requested by config.

#     All progress is logged to console and to create_nma_YYYYMMDD_HHMMSS.log.
#     """
#     t_global = time.perf_counter()

#     # --------------------------------------------------------------------- #
#     # Basic validation
#     # --------------------------------------------------------------------- #
#     if not user_id:
#         raise ValueError("user_id cannot be None")
#     if not model_id:
#         raise ValueError("model_id cannot be None")
#     if graph_type not in (
#         GraphTypes.SIMILARITY.value,
#         GraphTypes.DISSIMILARITY.value,
#         GraphTypes.COUNT.value,
#     ):
#         raise ValueError("graph_type must be 'similarity', 'dissimilarity' or 'count'")

#     logger.info("🏁 create_nma(): user=%s, model=%s, graph_type=%s, dataset=%s",
#                 user_id, model_id, graph_type, dataset)

#     try:
#         # ----------------------------------------------------------------- #
#         # Model + dataset
#         # ----------------------------------------------------------------- #
#         model_key  = _get_model_filename(user_id, model_id, graph_type)
#         if not model_key:
#             raise ValueError("Could not resolve model key for S3")

#         model_uri  = f"s3://{S3_USERS_BUCKET_NAME}/{model_key}"
#         logger.info("Model path resolved → %s", model_uri)

#         dataset_cfg = _get_dataset_config(dataset)

#         with timed("Load Keras model from S3"):
#             loaded_model = _load_model(dataset, model_uri, dataset_cfg)

#         with timed("Load dataset"):
#             dataset_obj = _load_dataset(dataset)

#         labels = dataset_obj.directory_labels if dataset == DatasetsEnum.IMAGENET.value else dataset_obj.labels
#         logger.info("Dataset loaded – %d samples, %d labels", iterable_length(dataset_obj) or -1, len(labels))

#         nma = NMA(
#             loaded_model.model,
#             dataset_obj,
#             labels,
#             graph_type=graph_type,
#             top_k=top_k,
#             min_confidence=min_confidence,
#         )

#         logger.debug("Linkage matrix Z shape: %s", None if nma.Z is None else nma.Z.shape)

#         # ----------------------------------------------------------------- #
#         # S3 paths for artefacts
#         # ----------------------------------------------------------------- #
#         base_prefix = f"{user_id}/{model_id}"  # Use just user_id and model_id
#         dataframe_key = f"{base_prefix}/{graph_type}/edges_df_test.csv"
#         dendrogram_key = f"{base_prefix}/{graph_type}/dendrogram_test"
#         logger.info("S3 targets → edges: %s | dendrogram: %s", dataframe_key, dendrogram_key)

#         # ----------------------------------------------------------------- #
#         # Save edges dataframe
#         # ----------------------------------------------------------------- #
#         if nma.edges_df is not None:
#             with timed("Upload edges_df.csv to S3"):
#                 EdgesDataframe(model_id, dataframe_key, nma.edges_df).save_dataframe()
#         else:
#             logger.warning("edges_df is None – nothing uploaded")

#         # ----------------------------------------------------------------- #
#         # Build & save dendrogram
#         # ----------------------------------------------------------------- #
#         dendro = Dendrogram(dendrogram_key, nma.Z)

#         with timed("Build tree hierarchy + upload dendrogram JSON"):
#             if dataset == "imagenet":
#                 # Get the mapping dictionary
#                 dir_to_readable = _get_dataset_config(dataset)["directory_to_readable"]
                
#                 # Create readable labels by mapping the WordNet IDs
#                 readable_labels = []
#                 for label in dataset_obj.labels:
#                     if label in dir_to_readable:
#                         readable_labels.append(dir_to_readable[label])
#                     else:
#                         # Keep original if no mapping exists
#                         readable_labels.append(label)
                
#                 # Use readable labels for building the tree hierarchy
#             else: 
#                 readable_labels = dataset_obj.labels
                
#             dendro._build_tree_hierarchy(nma.Z, readable_labels)

#             # Sanity-check the produced hierarchy
#             try:
#                 assert_acyclic(dendro.Z_tree_format)
#             except CycleFound as e:
#                 logger.error("Cycle detected in dendrogram: %s", e)
#                 raise

#             dendro.save_dendrogram(nma.Z)


#         available_labels = list(set(readable_labels[:100]))[:10] 

#         # Replace the init_selected_labels with actual labels
#         dataset_cfg["init_selected_labels"] = available_labels

#         logger.info(f"Updated init_selected_labels to use actual dataset labels: {available_labels}")

#         # Then continue with existing code
#         init_json = dendro.get_sub_dendrogram_formatted(dataset_cfg["init_selected_labels"])
        
        
        
#         logger.info("dataset_cfg[\"init_selected_labels\"]: %s", dataset_cfg["init_selected_labels"])
        
        
#         logger.debug("Initial sub-tree JSON length=%d", len(init_json))

#         # ----------------------------------------------------------------- #
#         # Update metadata
#         # ----------------------------------------------------------------- #
#         with timed("Update current_model metadata"):
#             update_current_model(
#                 user_id,
#                 model_id,
#                 graph_type,
#                 os.path.basename(model_key),
#                 dataset,
#                 min_confidence,
#                 top_k,
#             )

#         logger.info("🎉 create_nma completed in %.2fs", time.perf_counter() - t_global)
#         return init_json

#     # --------------------------------------------------------------------- #
#     # Error handling – log full traceback then re-raise
#     # --------------------------------------------------------------------- #
#     except Exception:
#         logger.exception("create_nma failed")
#         raise

def _create_nma(
    user_id: str,
    model_id: str,
    graph_type: str,
    dataset: str,
    min_confidence: float,
    top_k: int,
):
    """
    Build graph + dendrogram for `model_id`, save artefacts to S3 and return
    JSON with the initial sub-dendrogram requested by config.
    """
    t_global = time.perf_counter()
    
    # Basic validation
    if not user_id:
        raise ValueError("user_id cannot be None")
    if not model_id:
        raise ValueError("model_id cannot be None")
    if graph_type not in (
        GraphTypes.SIMILARITY.value,
        GraphTypes.DISSIMILARITY.value,
        GraphTypes.COUNT.value,
    ):
        raise ValueError("graph_type must be 'similarity', 'dissimilarity' or 'count'")

    logger.info("🏁 create_nma(): user=%s, model=%s, graph_type=%s, dataset=%s",
                user_id, model_id, graph_type, dataset)
    
    try:
        # Model + dataset
        model_key = _get_model_filename(user_id, model_id, graph_type)
        if not model_key:
            raise ValueError("Could not resolve model key for S3")
        
        model_uri = f"s3://{S3_USERS_BUCKET_NAME}/{model_key}"
        logger.info("Model path resolved → %s", model_uri)
        
        dataset_cfg = _get_dataset_config(dataset)
        
        with timed("Load Keras model from S3"):
            loaded_model = _load_model(dataset, model_uri, dataset_cfg)
        
        with timed("Load dataset"):
            dataset_obj = _load_dataset(dataset)
        
        # Use ORIGINAL WordNet labels (folder names) throughout processing
        labels = dataset_obj.directory_labels if dataset == DatasetsEnum.IMAGENET.value else dataset_obj.labels
        
        logger.info("Dataset loaded – %d samples, %d labels", iterable_length(dataset_obj) or -1, len(labels))
        
        # Create NMA with WordNet folder names
        nma = NMA(
            loaded_model.model,
            dataset_obj,
            labels,  # Using folder names like 'n01440764'
            graph_type=graph_type,
            top_k=top_k,
            min_confidence=min_confidence,
        )
        
        logger.debug("Linkage matrix Z shape: %s", None if nma.Z is None else nma.Z.shape)
        
        # S3 paths for artefacts
        base_prefix = f"{user_id}/{model_id}"
        dataframe_key = f"{base_prefix}/{graph_type}/edges_df_test.csv"
        dendrogram_key = f"{base_prefix}/{graph_type}/dendrogram_test"
        logger.info("S3 targets → edges: %s | dendrogram: %s", dataframe_key, dendrogram_key)
        
        # Save edges dataframe
        if nma.edges_df is not None:
            with timed("Upload edges_df.csv to S3"):
                EdgesDataframe(model_id, dataframe_key, nma.edges_df).save_dataframe()
        else:
            logger.warning("edges_df is None – nothing uploaded")
        
        # Build dendrogram with WordNet labels
        dendro = Dendrogram(dendrogram_key, nma.Z)
        
        with timed("Build tree hierarchy + upload dendrogram JSON"):
            
            if dataset == "imagenet":
                # Get the mapping dictionary
                dir_to_readable = _get_dataset_config(dataset)["directory_to_readable"]
                
                # Create readable labels by mapping the WordNet IDs
                readable_labels = []
                for label in dataset_obj.labels:
                    if label in dir_to_readable:
                        readable_labels.append(dir_to_readable[label])  # ← Convert to readable HERE
                    else:
                        readable_labels.append(label)
                
                # Use readable labels for building the tree hierarchy
            else: 
                readable_labels = dataset_obj.labels
                
            dendro._build_tree_hierarchy(nma.Z, readable_labels)
            
            # dendro._build_tree_hierarchy(nma.Z, labels) ### build with wordnet and not readable labels 
            
            dendro.save_dendrogram(nma.Z)
        
        if dataset == DatasetsEnum.IMAGENET.value:
            available_readable_labels = list(set(readable_labels[:100]))[:10]
            logger.info(f"Selected readable labels for JSON: {available_readable_labels}")
            
        else:
            available_readable_labels = list(set(readable_labels[:100]))[:10]
        
        init_json = dendro.get_sub_dendrogram_formatted(available_readable_labels)
        
        logger.debug("Initial sub-tree JSON length=%d", len(str(init_json)))
        
        # Update metadata
        with timed("Update current_model metadata"):
            update_current_model(
                user_id,
                model_id,
                graph_type,
                os.path.basename(model_key),
                dataset,
                min_confidence,
                top_k,
            )

        logger.info("🎉 create_nma completed in %.2fs", time.perf_counter() - t_global)
        return init_json

    except Exception:
        logger.exception("create_nma failed")
        raise


def convert_dendrogram_labels(dendrogram_data, label_mapping):
    """
    Recursively convert folder names to readable labels in the dendrogram data.
    
    Args:
        dendrogram_data: The dendrogram data structure (dict, list, or primitive)
        label_mapping: Dict mapping folder names to readable names
    
    Returns:
        The dendrogram data with converted labels
    """
    if isinstance(dendrogram_data, dict):
        # Create a new dict to avoid modifying the original
        new_dict = {}
        for key, value in dendrogram_data.items():
            if key == 'name' and isinstance(value, str) and value in label_mapping:
                # Convert the name field
                new_dict[key] = label_mapping[value]
            else:
                # Recursively process other fields
                new_dict[key] = convert_dendrogram_labels(value, label_mapping)
        return new_dict
    
    elif isinstance(dendrogram_data, list):
        # Recursively process all list items
        return [convert_dendrogram_labels(item, label_mapping) for item in dendrogram_data]
    
    elif isinstance(dendrogram_data, str):
        # If it's a string that needs conversion
        return label_mapping.get(dendrogram_data, dendrogram_data)
    
    return dendrogram_data
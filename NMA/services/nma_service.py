import os
import time
import sys
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Iterable

from NMA.utilss.enums.graph_types import GraphTypes
from NMA.utilss.enums.datasets_enum import DatasetsEnum
from NMA.classes.dendrogram import Dendrogram
from NMA.classes.edges_dataframe import EdgesDataframe
from NMA.classes.nma import NMA
from NMA.services.dataset_service import _get_dataset_config, _load_dataset
from NMA.services.model_service import _get_model_filename, _load_model
from NMA.utilss.files_utils import update_current_model
from NMA.utilss.s3_utils import get_datasets_s3_client, get_users_s3_client
from NMA.services.adversarial_files.adversarial_service import create_logistic_regression_detector
from NMA.services.ses_batch_service import send_email_notification
# from NMA.classes.user import User

sys.setrecursionlimit(10_000)  
# --------------------------------------------------------------------------- #
# 1) LOGGING CONFIGURATION 
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()           # export LOG_LEVEL=DEBUG for more noise

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    handlers=[
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


def _create_nma(
    user_id: str,
    model_id: str,
    graph_type: str,
    dataset: str,
    min_confidence: float,
    top_k: int,
    clean_images=None,
    adversarial_images=None,
    
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
        dataframe_key = f"{base_prefix}/{graph_type}/edges_df.csv"
        dendrogram_key = f"{base_prefix}/{graph_type}"
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
        
        # init_json = dendro.get_sub_dendrogram_formatted(available_readable_labels)
        
        create_logistic_regression_detector(model_id, graph_type, clean_images, adversarial_images, user_id)
        
        # logger.debug("Initial sub-tree JSON length=%d", len(str(init_json)))
        
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

        send_email_notification(user_id, model_key, graph_type)

        logger.info("🎉 create_nma completed in %.2fs", time.perf_counter() - t_global)


        # return init_json

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
import os
import time
import sys
import logging
from contextlib import contextmanager
from typing import Iterable
from ..utilss.enums.graph_types import GraphTypes
from ..utilss.enums.datasets_enum import DatasetsEnum
from ..classes.dendrogram import Dendrogram
from ..classes.edges_dataframe import EdgesDataframe
from ..classes.nma import NMA
from NMA.services.dataset_service import _get_dataset_config, _load_dataset
from NMA.services.model_service import _get_model_filename, _load_model
from NMA.services.adversarial_files.adversarial_service import create_logistic_regression_detector
from NMA.services.ses_batch_service import send_email_notification
from ..utilss.files_utils import update_current_model, update_job_status
from ..utilss.s3_utils import  get_users_s3_client
sys.setrecursionlimit(10_000)  
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  
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
S3_DATASETS_BUCKET_NAME = os.environ.get('S3_DATASETS_BUCKET_NAME', 'better-datasets')
S3_USERS_BUCKET_NAME    = os.environ.get('S3_USERS_BUCKET_NAME',    'better-xai-users')
s3_client = s3_client = get_users_s3_client


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
):
   
    t_global = time.perf_counter()
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
    
    update_job_status(user_id, model_id, "running")

    try:
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
        
        labels = dataset_obj.directory_labels if dataset == DatasetsEnum.IMAGENET.value else dataset_obj.labels
        logger.info("Dataset loaded – %d samples, %d labels", iterable_length(dataset_obj) or -1, len(labels))
        
        nma = NMA(
            loaded_model.model,
            dataset_obj,
            labels,  
            graph_type=graph_type,
            top_k=top_k,
            min_confidence=min_confidence,
        )
        
        logger.debug("Linkage matrix Z shape: %s", None if nma.Z is None else nma.Z.shape)
        base_prefix = f"{user_id}/{model_id}"
        dataframe_key = f"{base_prefix}/{graph_type}/edges_df.csv"
        dendrogram_key = f"{base_prefix}/{graph_type}/dendrogram"
        logger.info("S3 targets → edges: %s | dendrogram: %s", dataframe_key, dendrogram_key)
       
        if nma.edges_df is not None:
            with timed("Upload edges_df.csv to S3"):
                EdgesDataframe(model_id, dataframe_key, nma.edges_df).save_dataframe()
        else:
            logger.warning("edges_df is None – nothing uploaded")

        dendro = Dendrogram(dendrogram_key, nma.Z)
        
        with timed("Build tree hierarchy + upload dendrogram JSON"):
            if dataset == "imagenet":
                dir_to_readable = _get_dataset_config(dataset)["directory_to_readable"]
                readable_labels = []
                for label in dataset_obj.labels:
                    if label in dir_to_readable:
                        readable_labels.append(dir_to_readable[label])  # ← Convert to readable HERE
                    else:
                        readable_labels.append(label)
            else: 
                readable_labels = dataset_obj.labels
                
            dendro._build_tree_hierarchy(nma.Z, readable_labels)
            dendro.save_dendrogram(nma.Z)
        
        if dataset == DatasetsEnum.IMAGENET.value:
            available_readable_labels = list(set(readable_labels[:100]))[:10]
            logger.info(f"Selected readable labels for JSON: {available_readable_labels}")
            
        else:
            available_readable_labels = list(set(readable_labels[:100]))[:10]
        
        init_json = dendro.get_sub_dendrogram_formatted(available_readable_labels)
        logger.debug("Initial sub-tree JSON length=%d", len(str(init_json)))

        create_logistic_regression_detector(
            model_id=model_id,
            graph_type=graph_type,
            user_id=user_id
        )
        
        with timed("Update current_model metadata"):
            update_job_status(user_id, model_id, "succeeded")

        send_email_notification(user_id, model_id, graph_type)

        logger.info("🎉 create_nma completed in %.2fs", time.perf_counter() - t_global)
        return init_json

    except Exception:
        logger.exception("create_nma failed")
        update_job_status(user_id, model_id, "failed")

        raise


def convert_dendrogram_labels(dendrogram_data, label_mapping):
    if isinstance(dendrogram_data, dict):
        new_dict = {}
        for key, value in dendrogram_data.items():
            if key == 'name' and isinstance(value, str) and value in label_mapping:
                new_dict[key] = label_mapping[value]
            else:
                new_dict[key] = convert_dendrogram_labels(value, label_mapping)
        return new_dict
    
    elif isinstance(dendrogram_data, list):
        return [convert_dendrogram_labels(item, label_mapping) for item in dendrogram_data]
    
    elif isinstance(dendrogram_data, str):
        return label_mapping.get(dendrogram_data, dendrogram_data)

    return dendrogram_data
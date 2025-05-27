# from NMA.utilss.enums.graph_types import GraphTypes
# from NMA.utilss.enums.datasets import DatasetsEnum
# from NMA.classes.dendrogram import Dendrogram
# from NMA.classes.edges_dataframe import EdgesDataframe
# from NMA.classes.nma import NMA
# from NMA.dataset_service import _get_dataset_config, _load_dataset
# from NMA.model_service import _get_model_filename, _load_model



# def _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k):
#     if model_path is None:
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
#             detail="Model path is required"
#         )
        
#     if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
#             detail="Graph type must be either 'similarity', 'dissimilarity', or 'count'"
#         )
    
#     # Initialize S3 clients with correct credentials
#     users_s3_client = get_users_s3_client()
#     datasets_s3_client = get_datasets_s3_client()
    
#     s3_users_bucket = os.getenv("S3_USERS_BUCKET_NAME")
#     s3_datasets_bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
    
#     if not s3_users_bucket:
#         raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    
#     model_path = _get_model_filename(user_id, model_id, graph_type)
#     if model_path is None:
#         raise ValueError("Model path cannot be None")
    
#     print(f"Model path: {model_path}")
#     return model_path
    
    

# # # def _create_nma(model_file, graph_type, dataset_str, user, min_confidence, top_k, model_id_md):
# # def _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k):
# #     if user_id is None:
# #         raise ValueError("user_id cannot be None")
    
# #     if model_id is None:
# #         raise ValueError("model_id cannot be None")
        
# #     if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
# #         raise ValueError("Graph type must be either 'similarity', 'dissimilarity', or 'count'")

# #     model_path = _get_model_filename(user_id, model_id, graph_type)
# #     if model_path is None:
# #         raise ValueError("Model path cannot be None")
    
# #     print(f"Model path: {model_path}")
# #     return model_path
    
    
    
#     # dataset_config = _get_dataset_config(dataset_str)
#     # dataset = _load_dataset(dataset_config)
    
#     # try:
#     #     loaded_model = _load_model(dataset_str, model_file.name, dataset_config)
#     #     model_directory = _get_model_path(user.user_id, loaded_model.model_path)
#     #     if model_directory is None:
#     #         raise ValueError("Model directory cannot be None")
            
#     #     dataframe_filename = f'{model_directory}/{graph_type}/edges_df.csv'
#     #     dendrogram_filename = f'{model_directory}/{graph_type}/dendrogram'
        
#     #     labels = dataset.labels
#     #     if dataset_str == DatasetsEnum.IMAGENET.value:
#     #         labels = dataset.directory_labels

#     #     nma = NMA(
#     #         loaded_model.model,
#     #         dataset,
#     #         labels,
#     #         graph_type=graph_type,
#     #         top_k=top_k,
#     #         min_confidence=min_confidence           
#     #     )
        
#     #     edges_df_obj = EdgesDataframe(model_file.name, dataframe_filename, nma.edges_df)
#     #     edges_df_obj.save_dataframe()
#     #     print(nma.Z)            
        
#     #     dendrogram = Dendrogram(dendrogram_filename, nma.Z)
#     #     dendrogram._build_tree_hierarchy(nma.Z, dataset.labels)
        
#     #     dendrogram.save_dendrogram(nma.Z)
        
#     #     init_sub_z = dendrogram.get_sub_dendrogram_formatted(dataset_config["init_selected_labels"])
#     #     print(init_sub_z)

#     #     update_current_model(
#     #         user,
#     #         model_id_md,
#     #         graph_type,
#     #         model_file.filename,
#     #         dataset,
#     #         min_confidence,
#     #         top_k,
#     #     )
#     #     return init_sub_z
    
#     # except Exception as e:
#     #     raise RuntimeError(f"Error during NMA creation: {str(e)}")


from fastapi import HTTPException, status
import os
from .dataset_service import _get_dataset_config, _load_dataset
from .model_service import _load_model, _get_model_path, _get_model_filename
# from utilss.classes.nma import NMA
# from utilss.classes.edges_dataframe import EdgesDataframe
# from utilss.classes.dendrogram import Dendrogram
from utilss.enums.datasets_enum import DatasetsEnum
from utilss.enums.graph_types import GraphTypes
# from utilss.files_utils import update_current_model
import os
import boto3
from utilss.s3_utils import get_users_s3_client, get_datasets_s3_client
import json
import numpy as np
        
import logging
logger = logging.getLogger(__name__)

### original implemetation ###
# def _create_nma(model_file, graph_type, dataset_str, user, min_confidence, top_k, model_id_md):
#     if model_file is None:
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
#             detail="Model file is required"
#         )
        
#     if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
#             detail="Graph type must be either 'similarity' or 'dissimilarity'"
#         )
    
#     dataset_config = _get_dataset_config(dataset_str)
#     # dataset = _load_dataset(dataset_config)
#     dataset = _load_dataset(dataset_str)
    
#     try:
#         loaded_model = _load_model(dataset_str, model_file.name, dataset_config)
#         model_directory = _get_model_path(user.user_id, loaded_model.model_path)
#         if model_directory is None:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
#                 detail="Could not find model directory"
#             )
            
#         dataframe_filename = f'{model_directory}/{graph_type}/edges_df.csv'
#         dendrogram_filename = f'{model_directory}/{graph_type}/dendrogram'
        
#         labels = dataset.labels
#         if dataset_str == DatasetsEnum.IMAGENET.value:
#             labels = dataset.directory_labels

#         nma = NMA(
#             loaded_model.model,
#             dataset,
#             labels,
#             graph_type=graph_type,
#             top_k=top_k,
#             min_confidence=min_confidence           
#         )
        
#         edges_df_obj = EdgesDataframe(model_file.name, dataframe_filename, nma.edges_df)
#         edges_df_obj.save_dataframe()
#         print(nma.Z)            
        
#         dendrogram = Dendrogram(dendrogram_filename, nma.Z)
#         dendrogram._build_tree_hierarchy(nma.Z, dataset.labels)
        
#         dendrogram.save_dendrogram(nma.Z)
        
#         init_sub_z = dendrogram.get_sub_dendrogram_formatted(dataset_config["init_selected_labels"])
#         print(init_sub_z)

#         update_current_model(
#             user,
#             model_id_md,
#             graph_type,
#             model_file.filename,
#             dataset,
#             min_confidence,
#             top_k,
#         )
#         return init_sub_z
    
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


### S3 implementation ### 
# def _create_nma(model_path, graph_type, dataset_str, user, min_confidence, top_k, model_id_md):
#     if model_path is None:
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
#             detail="Model path is required"
#         )
        
#     if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
#             detail="Graph type must be either 'similarity', 'dissimilarity', or 'count'"
#         )
    
#     # Initialize S3 clients with correct credentials
#     users_s3_client = get_users_s3_client()
#     datasets_s3_client = get_datasets_s3_client()
    
#     s3_users_bucket = os.getenv("S3_USERS_BUCKET_NAME")
#     s3_datasets_bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
    
#     if not s3_users_bucket:
#         raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
#     dataset_config = _get_dataset_config(dataset_str)
    
#     # Pass datasets client to dataset loader
#     dataset = _load_dataset(dataset_str)
    
#     try:
#         # Use users client for model loading
#         loaded_model = _load_model(dataset_str, model_path, dataset_config)
#         model_directory = _get_model_path(user.user_id, model_id_md)
        
#         if model_directory is None:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
#                 detail="Could not find model directory"
#             )
            
#         dataset.model_directory = model_directory    
#         # S3 paths for files
#         dataframe_filename = f'{model_directory}/{graph_type}/edges_df.csv'
#         dendrogram_filename = f'{model_directory}/{graph_type}/dendrogram'
        
#         labels = dataset.labels
#         if dataset_str == DatasetsEnum.IMAGENET.value:
#             labels = dataset.directory_labels

#         # Pass both clients to NMA
#         nma = NMA(
#             loaded_model.model,
#             dataset,
#             labels,
#             graph_type=graph_type,
#             top_k=top_k,
#             min_confidence=min_confidence,
#         )
        
#         s3_params = {
#             "s3_client": users_s3_client,
#             "s3_bucket": s3_users_bucket
#         }
        
#         # Create and save edges dataframe using users client
#         edges_df_obj = EdgesDataframe(model_path, dataframe_filename, nma.edges_df)
#         edges_df_obj.save_dataframe()
#         print(nma.Z)            
        
#         # Create and save dendrogram using users client
#         dendrogram = Dendrogram(dendrogram_filename, nma.Z)
#         dendrogram._build_tree_hierarchy(nma.Z, dataset.labels)
        
#         dendrogram.save_dendrogram(nma.Z)
        
#         # Convert init_selected_labels for ImageNet
#         if dataset_str == DatasetsEnum.IMAGENET.value:
#             # Create a mapping from human readable names to synset IDs
#             readable_to_synset = {}
            
#             try:
#                 # Load the synset mapping from S3
#                 mapping_key = "imagenet/train/LOC_synset_mapping.txt"
#                 response = datasets_s3_client.get_object(
#                     Bucket=s3_datasets_bucket, 
#                     Key=mapping_key
#                 )
#                 content = response['Body'].read().decode('utf-8')
                
#                 # Parse the mapping file: "n01440764 tench, Tinca tinca"
#                 for line in content.strip().split('\n'):
#                     if ' ' in line:
#                         synset, readable_names = line.split(' ', 1)
#                         # Split multiple names and clean them
#                         names = [name.strip() for name in readable_names.split(',')]
#                         for name in names:
#                             readable_to_synset[name] = synset
                
#                 logger.info(f"Loaded {len(readable_to_synset)} readable name mappings")
                
#                 # Convert init_selected_labels from readable names to synset IDs
#                 original_labels = dataset_config["init_selected_labels"]
#                 converted_labels = []
                
#                 for label in original_labels:
#                     if label in readable_to_synset:
#                         converted_labels.append(readable_to_synset[label])
#                         logger.info(f"Converted '{label}' -> '{readable_to_synset[label]}'")
#                     else:
#                         logger.warning(f"Could not find synset for '{label}', skipping")
                
#                 logger.info(f"Converted {len(converted_labels)} labels for ImageNet dendrogram")
#                 init_selected_labels = converted_labels
                
#             except Exception as e:
#                 logger.warning(f"Could not load synset mapping: {str(e)}, using first 5 synsets")
#                 # Fallback: use first 5 synsets from directory_labels
#                 init_selected_labels = dataset.directory_labels[:5]
#         else:
#             # For CIFAR-100, use labels as-is
#             init_selected_labels = dataset_config["init_selected_labels"]

#         init_sub_z = dendrogram.get_sub_dendrogram_formatted(init_selected_labels)
#         print(init_sub_z)

#         # ✅ Convert JSON to the format expected by NMAResult
#         init_sub_z_dict = json.loads(init_sub_z)

#         # Convert the nested dict structure to nested lists (recursive function)
#         def dict_to_nested_lists(obj):
#             if isinstance(obj, dict):
#                 # Convert dict to list of [key, value] pairs
#                 return [[k, dict_to_nested_lists(v)] for k, v in obj.items()]
#             elif isinstance(obj, list):
#                 return [dict_to_nested_lists(item) for item in obj]
#             else:
#                 return obj

#         # Convert to nested list structure
#         nested_list_structure = dict_to_nested_lists(init_sub_z_dict)
#         init_z_array = np.array([nested_list_structure], dtype=object)


#         # Extract filename from model_path
#         model_filename = os.path.basename(model_path)
#         update_current_model(
#             user,
#             model_id_md,
#             graph_type,
#             model_filename,
#             dataset,
#             min_confidence,
#             top_k,
#         )
#         return init_z_array 
    
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k):
    if model_path is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model path is required"
        )
        
    if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Graph type must be either 'similarity', 'dissimilarity', or 'count'"
        )
    
    # Initialize S3 clients with correct credentials
    users_s3_client = get_users_s3_client()
    datasets_s3_client = get_datasets_s3_client()
    
    s3_users_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    s3_datasets_bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
    
    if not s3_users_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    
    model_path = _get_model_filename(user_id, model_id, graph_type)
    if model_path is None:
        raise ValueError("Model path cannot be None")
    
    print(f"Model path: {model_path}")
    return model_path
    
    
    
    ### the previous before integration with BATCH implementation
    
    
    # try:
    #     # Use users client for model loading
    #     loaded_model = _load_model(dataset_str, model_path, dataset_config)
    #     model_directory = _get_model_path(user.user_id, model_id_md)
        
    #     if model_directory is None:
    #         raise HTTPException(
    #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
    #             detail="Could not find model directory"
    #         )
            
    #     dataset.model_directory = model_directory    
    #     # S3 paths for files
    #     dataframe_filename = f'{model_directory}/{graph_type}/edges_df.csv'
    #     dendrogram_filename = f'{model_directory}/{graph_type}/dendrogram'
        
    #     labels = dataset.labels
    #     if dataset_str == DatasetsEnum.IMAGENET.value:
    #         labels = dataset.directory_labels

    #     # Pass both clients to NMA
    #     nma = NMA(
    #         loaded_model.model,
    #         dataset,
    #         labels,
    #         graph_type=graph_type,
    #         top_k=top_k,
    #         min_confidence=min_confidence,
    #     )
        
    #     s3_params = {
    #         "s3_client": users_s3_client,
    #         "s3_bucket": s3_users_bucket
    #     }
        
    #     # Create and save edges dataframe using users client
    #     edges_df_obj = EdgesDataframe(model_path, dataframe_filename, nma.edges_df)
    #     edges_df_obj.save_dataframe()
    #     print(nma.Z)            
        
    #     # Create and save dendrogram using users client
    #     dendrogram = Dendrogram(dendrogram_filename, nma.Z)
    #     dendrogram._build_tree_hierarchy(nma.Z, dataset.labels)
        
    #     dendrogram.save_dendrogram(nma.Z)
        
    #     # Convert init_selected_labels for ImageNet
    #     if dataset_str == DatasetsEnum.IMAGENET.value:
    #         # Create a mapping from human readable names to synset IDs
    #         readable_to_synset = {}
            
    #         try:
    #             # Load the synset mapping from S3
    #             mapping_key = "imagenet/train/LOC_synset_mapping.txt"
    #             response = datasets_s3_client.get_object(
    #                 Bucket=s3_datasets_bucket, 
    #                 Key=mapping_key
    #             )
    #             content = response['Body'].read().decode('utf-8')
                
    #             # Parse the mapping file: "n01440764 tench, Tinca tinca"
    #             for line in content.strip().split('\n'):
    #                 if ' ' in line:
    #                     synset, readable_names = line.split(' ', 1)
    #                     # Split multiple names and clean them
    #                     names = [name.strip() for name in readable_names.split(',')]
    #                     for name in names:
    #                         readable_to_synset[name] = synset
                
    #             logger.info(f"Loaded {len(readable_to_synset)} readable name mappings")
                
    #             # Convert init_selected_labels from readable names to synset IDs
    #             original_labels = dataset_config["init_selected_labels"]
    #             converted_labels = []
                
    #             for label in original_labels:
    #                 if label in readable_to_synset:
    #                     converted_labels.append(readable_to_synset[label])
    #                     logger.info(f"Converted '{label}' -> '{readable_to_synset[label]}'")
    #                 else:
    #                     logger.warning(f"Could not find synset for '{label}', skipping")
                
    #             logger.info(f"Converted {len(converted_labels)} labels for ImageNet dendrogram")
    #             init_selected_labels = converted_labels
                
    #         except Exception as e:
    #             logger.warning(f"Could not load synset mapping: {str(e)}, using first 5 synsets")
    #             # Fallback: use first 5 synsets from directory_labels
    #             init_selected_labels = dataset.directory_labels[:5]
    #     else:
    #         # For CIFAR-100, use labels as-is
    #         init_selected_labels = dataset_config["init_selected_labels"]

    #     init_sub_z = dendrogram.get_sub_dendrogram_formatted(init_selected_labels)
    #     print(init_sub_z)

    #     # ----- BEGIN FIXED CODE -----
    #     # Parse the JSON string into a Python dictionary
    #     init_sub_z_dict = json.loads(init_sub_z)
        
    #     # Convert hierarchical structure to flat list of lists of floats
    #     # This is the format expected by the NMAResult model
    #     flat_cluster_data = []
        
    #     def flatten_dendrogram(node, parent_id=-1.0, level=0.0):
    #         """
    #         Recursively flatten a hierarchical dendrogram structure into a list of float lists.
    #         Each list contains: [id, parent_id, level, value]
    #         """
    #         if not isinstance(node, dict):
    #             return
                
    #         # Extract node ID and value
    #         node_id = float(node.get('id', 0))
    #         node_value = float(node.get('value', 0.0))
            
    #         # Add this node as a row [id, parent_id, level, value]
    #         flat_cluster_data.append([node_id, parent_id, level, node_value])
            
    #         # Process children
    #         if 'children' in node and isinstance(node['children'], list):
    #             for child in node['children']:
    #                 flatten_dendrogram(child, node_id, level + 1.0)
        
    #     # Start the recursion from the root node
    #     flatten_dendrogram(init_sub_z_dict)
        
    #     # If no data was extracted, add a placeholder to avoid empty list error
    #     if not flat_cluster_data:
    #         # Add placeholder data if needed
    #         flat_cluster_data = [[0.0, 0.0, 0.0, 0.0]]
        
    #     # Save the original hierarchical structure to S3 for future reference
    #     hierarchy_path = f'{model_directory}/{graph_type}/hierarchy.json'
    #     users_s3_client.put_object(
    #         Body=init_sub_z,
    #         Bucket=s3_users_bucket,
    #         Key=hierarchy_path
    #     )
        
    #     logger.info(f"Flattened dendrogram to {len(flat_cluster_data)} data points")
    #     # ----- END FIXED CODE -----

    #     # Extract filename from model_path
    #     model_filename = os.path.basename(model_path)
    #     update_current_model(
    #         user,
    #         model_id_md,
    #         graph_type,
    #         model_filename,
    #         dataset,
    #         min_confidence,
    #         top_k,
    #     )
        
    #     # Return the flattened data structure
    #     return np.array(flat_cluster_data)
    
    # except Exception as e:
    #     logger.error(f"Error in _create_nma: {str(e)}", exc_info=True)
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
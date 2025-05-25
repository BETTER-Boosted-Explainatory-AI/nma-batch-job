from NMA.utilss.enums.graph_types import GraphTypes
from NMA.utilss.enums.datasets import DatasetsEnum
from NMA.classes.dendrogram import Dendrogram
from NMA.classes.edges_dataframe import EdgesDataframe
from NMA.classes.nma import NMA
from NMA.dataset_service import _get_dataset_config, _load_dataset
from NMA.model_service import _get_model_path, _load_model

def _create_nma(model_file, graph_type, dataset_str, user, min_confidence, top_k, model_id_md):
    if model_file is None:
        raise ValueError("Model file cannot be None")
        
    if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
        raise ValueError("Graph type must be either 'similarity', 'dissimilarity', or 'count'")

    
    dataset_config = _get_dataset_config(dataset_str)
    dataset = _load_dataset(dataset_config)
    
    try:
        loaded_model = _load_model(dataset_str, model_file.name, dataset_config)
        model_directory = _get_model_path(user.user_id, loaded_model.model_path)
        if model_directory is None:
            raise ValueError("Model directory cannot be None")
            
        dataframe_filename = f'{model_directory}/{graph_type}/edges_df.csv'
        dendrogram_filename = f'{model_directory}/{graph_type}/dendrogram'
        
        labels = dataset.labels
        if dataset_str == DatasetsEnum.IMAGENET.value:
            labels = dataset.directory_labels

        nma = NMA(
            loaded_model.model,
            dataset,
            labels,
            graph_type=graph_type,
            top_k=top_k,
            min_confidence=min_confidence           
        )
        
        edges_df_obj = EdgesDataframe(model_file.name, dataframe_filename, nma.edges_df)
        edges_df_obj.save_dataframe()
        print(nma.Z)            
        
        dendrogram = Dendrogram(dendrogram_filename, nma.Z)
        dendrogram._build_tree_hierarchy(nma.Z, dataset.labels)
        
        dendrogram.save_dendrogram(nma.Z)
        
        init_sub_z = dendrogram.get_sub_dendrogram_formatted(dataset_config["init_selected_labels"])
        print(init_sub_z)

        update_current_model(
            user,
            model_id_md,
            graph_type,
            model_file.filename,
            dataset,
            min_confidence,
            top_k,
        )
        return init_sub_z
    
    except Exception as e:
        raise RuntimeError(f"Error during NMA creation: {str(e)}")
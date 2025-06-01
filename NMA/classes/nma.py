# import tensorflow as tf
# import pandas as pd
# from igraph import Graph
# from enums.graph_types import GraphTypes
# from NMA.enums.heap_types import HeapType
# from classes.preprocessing.batch_predictor import BatchPredictor
# from classes.preprocessing.graph_builder import GraphBuilder
# from classes.preprocessing.heap_processor import HeapProcessor
# from classes.preprocessing.hierarchical_clustering_builder import HierarchicalClusteringBuilder
# from classes.preprocessing.z_builder import ZBuilder

# class NMA:
#     def __init__(
#         self,
#         model,
#         dataset_class,
#         labels,
#         graph_type=GraphTypes.SIMILARITY.value,
#         top_k=4,
#         min_confidence=0.8,
#         save_connections=True,
#         batch_size=32,
#     ):
#         """
#         X: images array,
#         y: labels,
#         model: keras model,
#         graph_type: similarity/dissimilarity
#         k: top predictions
#         t: top prediction thresholds to take into account
#         Preprossing the nma
#         """
#         if not isinstance(model, tf.keras.Model):
#             raise TypeError("model must be an instance of tf.keras.Model")

#         self.model = model
#         self.save_connections = save_connections
#         self.graph_type = graph_type
#         self.top_k = top_k
#         self.graph_threshold = dataset_class.threshold
#         self.infinity = dataset_class.infinity
#         self.min_confidence = min_confidence
#         self.labels = labels
#         self.edges_df = None
#         self.Z = None

#         if graph_type == GraphTypes.DISSIMILARITY.value:
#             self.heap_type = HeapType.MINIMUM.value
#         elif graph_type == GraphTypes.SIMILARITY.value or graph_type == GraphTypes.COUNT.value:
#             self.heap_type = HeapType.MAXIMUM.value

#         self.TBD_graph = None

#         self._preprocessing(dataset_class, batch_size)

#     def _get_image_probabilities_by_id(self, image_id):
#         probabilities_df = self.edges_df[self.edges_df["image_id"] == image_id]
#         return probabilities_df

#     def _get_dataframe_by_count(self):
#         return (
#             self.edges_df.groupby("source")["target"]
#             .value_counts()
#             .reset_index(name="count")
#         )

#     def _preprocessing(self, dataset_class, batch_size):
#         try:
#             X = dataset_class.x_train
#             y = dataset_class.y_train
            
#             graph = Graph(directed=False)
#             graph.add_vertices(self.labels)
    
#             edges_data = []
            
#             # Process all images in batches while preserving original indices
#             predictor = BatchPredictor(self.model, batch_size=batch_size)
#             builder = GraphBuilder(self.graph_type, self.infinity)
            
#             # Process images in batches
#             for i in range(0, len(X), batch_size):
#                 batch_end = min(i + batch_size, len(X))
#                 batch_images = X[i:batch_end]
#                 batch_labels = y[i:batch_end]
#                 original_indices = list(range(i, batch_end))  # Track the original indices
                
#                 top_predictions_batch = predictor.get_top_predictions(
#                     batch_images, self.labels, self.top_k, self.graph_threshold
#                 )
    
#                 for j, top_predictions in enumerate(top_predictions_batch):
#                     current_label = batch_labels[j]
#                     original_index = original_indices[j]  # Use the correct original index
    
#                     seen_labels_for_image = {current_label}
                    
#                     if len(top_predictions) == 0:
#                         print(f"No predictions for image at index {original_index}")
#                         continue
    
#                     if len(top_predictions[0]) < 2:
#                         continue
    
#                     if top_predictions[0][2] > self.min_confidence:
#                         filtered_predictions = top_predictions
    
#                         for _, pred_label, pred_prob in filtered_predictions:
#                             if pred_label not in self.labels:
#                                 raise ValueError(
#                                     f"Prediction label '{pred_label}' not in graph labels."
#                                 )
                                
#                             seen_labels_for_image.add(pred_label)
    
#                             if current_label != pred_label:
#                                 edge_data = builder.update_graph(
#                                     graph, current_label, pred_label, pred_prob, original_index, dataset_class
#                                 )
#                                 # Only append edge_data if it's not None (not a self-loop)
#                                 if edge_data is not None:
#                                     edges_data.append(edge_data)
                                    
#                     if self.graph_type == GraphTypes.DISSIMILARITY.value:
#                         for label in self.labels:
#                             # Only add infinity edges for labels not seen in this image's predictions
#                             if label not in seen_labels_for_image:
#                                 builder.add_infinity_edges(
#                                     graph, label, current_label
#                                 )
    
#             if self.save_connections:
#                 self.edges_df = pd.DataFrame(edges_data)
    
#             self.TBD_graph = graph
#             heap_processor = HeapProcessor(self.TBD_graph, self.graph_type, self.labels)
#             self.heap_processor = heap_processor
    
#             clustering = HierarchicalClusteringBuilder(heap_processor, self.labels)
#             self.Z = ZBuilder.create_z_matrix_from_tree(clustering, self.labels)
    
#         except Exception as e:
#             print(f"Error while preprocessing model: {str(e)}")

#     def get_neighbors_by_label_name(self, label_name):
#         neighbors = self.TBD_graph.neighbors(label_name)
#         neighbors_names = [self.TBD_graph.vs[n]["name"] for n in neighbors]

#         data = [
#             {
#                 "Neighbor": neighbor,
#                 "Weight": self.TBD_graph.es[
#                     self.TBD_graph.get_eid(label_name, neighbor)
#                 ]["weight"],
#             }
#             for neighbor in neighbors_names
#         ]

#         return pd.DataFrame(data).sort_values(by="Weight", ascending=False)


import tensorflow as tf
import pandas as pd
from igraph import Graph
from NMA.utilss.enums.heap_types import HeapType
from .preprocessing.batch_predictor import BatchPredictor
from .preprocessing.heap_processor import HeapProcessor
from .preprocessing.graph_builder import GraphBuilder
from .preprocessing.hierarchical_clustering_builder import HierarchicalClusteringBuilder
from .preprocessing.z_builder import ZBuilder
from NMA.utilss.enums.graph_types import GraphTypes
import logging
from NMA.utilss.enums.graph_types import GraphTypes
import boto3
import os
import numpy as np

logger = logging.getLogger(__name__)



class NMA:
    def __init__(
        self,
        model,
        dataset_class,
        labels,
        graph_type=GraphTypes.SIMILARITY.value,
        top_k=4,
        min_confidence=0.8,
        save_connections=True,
        batch_size=32,
    ):
        """
        X: images array,
        y: labels,
        model: keras model,
        graph_type: similarity/dissimilarity
        k: top predictions
        t: top prediction thresholds to take into account
        Preprossing the nma
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError("model must be an instance of tf.keras.Model")

        self.model = model
        self.save_connections = save_connections
        self.graph_type = graph_type
        self.top_k = top_k
        self.graph_threshold = dataset_class.threshold
        self.infinity = dataset_class.infinity
        self.min_confidence = min_confidence
        self.labels = labels
        self.edges_df = None
        self.Z = None
        
### S3 implementation ### 

        from NMA.utilss.s3_utils import get_users_s3_client, get_datasets_s3_client
        self.users_s3_client =  get_users_s3_client()
        self.datasets_s3_client =  get_datasets_s3_client()
        
        self.users_s3_bucket =  os.getenv("S3_USERS_BUCKET_NAME")
        self.datasets_s3_bucket =  os.getenv("S3_DATASETS_BUCKET_NAME")
        
        if self.users_s3_bucket is None:
            logger.warning("S3_USERS_BUCKET_NAME not set, will not save to users bucket")
        if self.datasets_s3_bucket is None:
            logger.warning("S3_DATASETS_NAME not set, will not load from datasets bucket")
        if graph_type == GraphTypes.DISSIMILARITY.value:
            self.heap_type = HeapType.MINIMUM.value
        elif graph_type == GraphTypes.SIMILARITY.value or graph_type == GraphTypes.COUNT.value:
            self.heap_type = HeapType.MAXIMUM.value

        self.TBD_graph = None
        
        self._preprocessing(dataset_class, batch_size)

    def _get_image_probabilities_by_id(self, image_id):
        probabilities_df = self.edges_df[self.edges_df["image_id"] == image_id]
        return probabilities_df

    def _get_dataframe_by_count(self):
        return (
            self.edges_df.groupby("source")["target"]
            .value_counts()
            .reset_index(name="count")
        )
        
### original implemetation ###
    # def _preprocessing(self, dataset_class, batch_size):
    #     try:
    #         X = dataset_class.x_train
    #         y = dataset_class.y_train
            
    #         graph = Graph(directed=False)
    #         graph.add_vertices(self.labels)
    
    #         edges_data = []
            
    #         # Process all images in batches while preserving original indices
    #         predictor = BatchPredictor(self.model, batch_size=batch_size)
    #         builder = GraphBuilder(self.graph_type, self.infinity)
            
    #         # Process images in batches
    #         for i in range(0, len(X), batch_size):
    #             batch_end = min(i + batch_size, len(X))
    #             batch_images = X[i:batch_end]
    #             batch_labels = y[i:batch_end]
    #             original_indices = list(range(i, batch_end))  # Track the original indices
                
    #             top_predictions_batch = predictor.get_top_predictions(
    #                 batch_images, self.labels, self.top_k, self.graph_threshold
    #             )
    
    #             for j, top_predictions in enumerate(top_predictions_batch):
    #                 current_label = batch_labels[j]
    #                 original_index = original_indices[j]  # Use the correct original index
    
    #                 seen_labels_for_image = {current_label}
                    
    #                 if len(top_predictions) == 0:
    #                     print(f"No predictions for image at index {original_index}")
    #                     continue
    
    #                 if len(top_predictions[0]) < 2:
    #                     continue
    
    #                 if top_predictions[0][2] > self.min_confidence:
    #                     filtered_predictions = top_predictions
    
    #                     for _, pred_label, pred_prob in filtered_predictions:
    #                         if pred_label not in self.labels:
    #                             raise ValueError(
    #                                 f"Prediction label '{pred_label}' not in graph labels."
    #                             )
                                
    #                         seen_labels_for_image.add(pred_label)
    
    #                         if current_label != pred_label:
    #                             edge_data = builder.update_graph(
    #                                 graph, current_label, pred_label, pred_prob, original_index, dataset_class
    #                             )
    #                             # Only append edge_data if it's not None (not a self-loop)
    #                             if edge_data is not None:
    #                                 edges_data.append(edge_data)
                                    
    #                 if self.graph_type == GraphTypes.DISSIMILARITY.value:
    #                     for label in self.labels:
    #                         # Only add infinity edges for labels not seen in this image's predictions
    #                         if label not in seen_labels_for_image:
    #                             builder.add_infinity_edges(
    #                                 graph, label, current_label
    #                             )
    
    #         if self.save_connections:
    #             self.edges_df = pd.DataFrame(edges_data)
    
    #         self.TBD_graph = graph
    #         heap_processor = HeapProcessor(self.TBD_graph, self.graph_type, self.labels)
    #         self.heap_processor = heap_processor
    
    #         clustering = HierarchicalClusteringBuilder(heap_processor, self.labels)
    #         self.Z = ZBuilder.create_z_matrix_from_tree(clustering, self.labels)
    
    #     except Exception as e:
    #         print(f"Error while preprocessing model: {str(e)}")
    
    ### S3 implementation ### 
    def _preprocessing(self, dataset_class, batch_size):
        """
        Preprocessing function that loads data directly from S3 using existing dataset infrastructure.
        Uses the dataset class's existing load_from_s3 method for data loading.
        """
        try:
            # Initialize graph and data structures
            graph = Graph(directed=False)
            graph.add_vertices(self.labels)
            edges_data = []
            
            # Initialize processors
            predictor = BatchPredictor(self.model, batch_size=batch_size)
            builder = GraphBuilder(self.graph_type, self.infinity)
            
            logger.info("Loading dataset from S3...")
            
            # Determine the dataset type from the class name
            dataset_class_name = dataset_class.__class__.__name__.lower()
            logger.info(f"Dataset class: {dataset_class_name}")
            
            # Map class names to dataset names and S3 prefixes
            if dataset_class_name == 'cifar100':
                dataset_name = 'cifar100'
                s3_prefix = 'cifar100/train'
            elif dataset_class_name == 'imagenet':
                dataset_name = 'imagenet' 
                s3_prefix = 'imagenet/train'
            else:
                # Fallback: try to get from dataset attribute if it exists
                dataset_name = getattr(dataset_class, 'dataset', dataset_class_name)
                s3_prefix = f"{dataset_name}/train"
            
            logger.info(f"Using dataset: {dataset_name}, S3 prefix: {s3_prefix}")
            
            # Load data using the existing load_from_s3 method
            logger.info(f"Calling load_from_s3 with bucket={self.datasets_s3_bucket}, prefix={s3_prefix}")
            
            try:
                result = dataset_class.load_from_s3(
                    s3_client=self.datasets_s3_client,
                    bucket=self.datasets_s3_bucket,
                    prefix=s3_prefix
                )
                logger.info(f"load_from_s3 returned: {type(result)}")
                
                if result is None:
                    raise ValueError(f"load_from_s3 returned None for {dataset_name}")
                
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError(f"load_from_s3 returned invalid format: {type(result)}, expected tuple of length 2")
                
                X, y = result
                logger.info(f"X type: {type(X)}, y type: {type(y)}")
                
                if X is None:
                    raise ValueError(f"X data is None for {dataset_name}")
                if y is None:
                    raise ValueError(f"y data is None for {dataset_name}")
                    
                logger.info(f"X shape: {getattr(X, 'shape', 'no shape attribute')}")
                logger.info(f"y length: {len(y) if hasattr(y, '__len__') else 'no length'}")
                
            except Exception as load_error:
                logger.error(f"Error in load_from_s3: {str(load_error)}")
                logger.info(f"Attempting fallback to regular load method...")
                
                # Fallback: try using the regular load method
                try:
                    if dataset_class_name == 'cifar100':
                        dataset_class.load('cifar100')
                    elif dataset_class_name == 'imagenet':
                        dataset_class.load('imagenet')
                    else:
                        dataset_class.load(dataset_name)
                        
                    X = dataset_class.x_train
                    y = dataset_class.y_train
                    logger.info(f"Fallback successful - X: {type(X)}, y: {type(y)}")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {str(fallback_error)}")
                    raise ValueError(f"Both S3 loading and fallback failed. S3 error: {str(load_error)}, Fallback error: {str(fallback_error)}")
            
            if X is None or not hasattr(X, '__len__'):
                raise ValueError(f"X data is invalid for {dataset_name}: {type(X)}")
            
            if len(X) == 0:
                raise ValueError(f"No training data loaded for {dataset_name}")
            
            total_images = len(X)
            logger.info(f"Processing {total_images} images from S3")
            
            # Process images in batches - same logic as original
            for i in range(0, total_images, batch_size):
                batch_end = min(i + batch_size, total_images)
                batch_images = X[i:batch_end]
                batch_labels = y[i:batch_end]
                original_indices = list(range(i, batch_end))
                
                # Get predictions for the batch
                top_predictions_batch = predictor.get_top_predictions(
                    batch_images, self.labels, self.top_k, self.graph_threshold
                )
                
                # Process each image in the batch
                for j, top_predictions in enumerate(top_predictions_batch):
                    if j >= len(batch_labels):  # Safety check
                        break
                        
                    current_label = batch_labels[j]
                    original_index = original_indices[j]
                    
                    seen_labels_for_image = {current_label}
                    
                    if len(top_predictions) == 0:
                        logger.debug(f"No predictions for image at index {original_index}")
                        continue
                    
                    if len(top_predictions[0]) < 2:
                        continue
                    
                    # Process predictions if confidence threshold is met
                    if top_predictions[0][2] > self.min_confidence:
                        filtered_predictions = top_predictions
                        
                        for _, pred_label, pred_prob in filtered_predictions:
                            if pred_label not in self.labels:
                                raise ValueError(
                                    f"Prediction label '{pred_label}' not in graph labels."
                                )
                            
                            seen_labels_for_image.add(pred_label)
                            
                            # Add edges for different labels
                            if current_label != pred_label:
                                edge_data = builder.update_graph(
                                    graph, current_label, pred_label, pred_prob, original_index, dataset_class
                                )
                                if edge_data is not None:
                                    edges_data.append(edge_data)
                    
                    # Add infinity edges for dissimilarity graphs
                    if self.graph_type == GraphTypes.DISSIMILARITY.value:
                        for label in self.labels:
                            if label not in seen_labels_for_image:
                                builder.add_infinity_edges(graph, label, current_label)
                
                # Log progress every 10 batches
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {batch_end}/{total_images} images")
            
            # Save edges dataframe if requested
            if self.save_connections:
                self.edges_df = pd.DataFrame(edges_data)
            
            # Build final structures
            self.TBD_graph = graph
            heap_processor = HeapProcessor(self.TBD_graph, self.graph_type, self.labels)
            self.heap_processor = heap_processor
            
            clustering = HierarchicalClusteringBuilder(heap_processor, self.labels)
            self.Z = ZBuilder.create_z_matrix_from_tree(clustering, self.labels)
            
            print("self.Z", self.Z)
            
            logger.info("Preprocessing completed successfully")
            
        except Exception as e:
            logger.error(f"Error while preprocessing model: {str(e)}")
            raise

    def get_neighbors_by_label_name(self, label_name):
        neighbors = self.TBD_graph.neighbors(label_name)
        neighbors_names = [self.TBD_graph.vs[n]["name"] for n in neighbors]
        data = [
            {
                "Neighbor": neighbor,
                "Weight": self.TBD_graph.es[
                    self.TBD_graph.get_eid(label_name, neighbor)
                ]["weight"],
            }
            for neighbor in neighbors_names
        ]

        return pd.DataFrame(data).sort_values(by="Weight", ascending=False)


    def _dataframe_to_csv_buffer(self, df):
        """Convert DataFrame to CSV buffer for streaming to S3"""
        import io
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    def _numpy_to_buffer(self, array):
        """Convert numpy array to buffer for streaming to S3"""
        import io
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        return buffer
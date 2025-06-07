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
        batch_size=16,
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
        self.uf = None
        
        # S3 implementation
        from NMA.utilss.s3_utils import get_users_s3_client, get_datasets_s3_client
        self.users_s3_client = get_users_s3_client()
        self.datasets_s3_client = get_datasets_s3_client()
        
        self.users_s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
        self.datasets_s3_bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
        
        if self.users_s3_bucket is None:
            logger.warning("S3_USERS_BUCKET_NAME not set, will not save to users bucket")
        if self.datasets_s3_bucket is None:
            logger.warning("S3_DATASETS_BUCKET_NAME not set, will not load from datasets bucket")
        
        if graph_type == GraphTypes.DISSIMILARITY.value:
            self.heap_type = HeapType.MINIMUM.value
        elif graph_type == GraphTypes.SIMILARITY.value or graph_type == GraphTypes.COUNT.value:
            self.heap_type = HeapType.MAXIMUM.value
    
        self.TBD_graph = None
    
        self._preprocessing(dataset_class, batch_size)
        
    def _get_image_probabilities_by_id(self, image_id):
        probabilities_df = self.edges_df[self.edges_df['image_id'] == image_id]
        return probabilities_df

    def _get_dataframe_by_count(self):
        return self.edges_df.groupby('source')['target'].value_counts().reset_index(name='count')
    
    def _preprocessing(self, dataset_class, batch_size):
        try:
            # S3 loading logic
            logger.info("Loading dataset from S3...")
            
            dataset_class_name = dataset_class.__class__.__name__.lower()
            logger.info(f"Dataset class: {dataset_class_name}")
            
            if dataset_class_name == 'cifar100':
                dataset_name = 'cifar100'
                s3_prefix = 'cifar100/train'
            elif dataset_class_name == 'imagenet':
                dataset_name = 'imagenet' 
                s3_prefix = 'imagenet/train'
            else:
                dataset_name = getattr(dataset_class, 'dataset', dataset_class_name)
                s3_prefix = f"{dataset_name}/train"
            
            logger.info(f"Using dataset: {dataset_name}, S3 prefix: {s3_prefix}")
            
            # Load data from S3
            try:
                result = dataset_class.load_from_s3(
                    s3_client=self.datasets_s3_client,
                    bucket=self.datasets_s3_bucket,
                    prefix=s3_prefix
                )
                X, y = result
                logger.info(f"Successfully loaded {len(X)} images from S3")
            except Exception as e:
                logger.error(f"Error loading from S3: {str(e)}")
                # Fallback to local if needed
                dataset_class.load(dataset_name)
                X = dataset_class.x_train
                y = dataset_class.y_train
            
            graph = Graph(directed=False)
            graph.add_vertices(self.labels)
            
            edges_data = []
            batch_images = []
            true_labels = []
            original_dataset_positions = [] 
            
            predictor = BatchPredictor(self.model)
            builder = GraphBuilder(self.graph_type, self.infinity)
            
            for i, image in enumerate(X):
                source_label = y[i]
                
                batch_images.append(image)
                true_labels.append(source_label)
                original_dataset_positions.append(i)
                
                if len(batch_images) == predictor.batch_size or i == len(X) - 1:
                    top_predictions_batch = predictor.get_top_predictions(
                        batch_images, self.labels, self.top_k, self.graph_threshold
                    )
                    
                    added_labels = []
                    for j, top_predictions in enumerate(top_predictions_batch):
                        current_label = true_labels[j]
                        original_index = original_dataset_positions[j]
                        seen_labels_for_image = {current_label}
                        
                        if len(top_predictions) == 0:
                            print("Empty predictions for image", original_index)
                            continue
                        
                        if len(top_predictions[0]) < 2:
                            print("Malformed predictions for image", original_index)
                            continue
                        
                        if top_predictions[0][2] > self.min_confidence:
                            filtered_predictions = top_predictions
                                                        
                            for _, pred_label, pred_prob in filtered_predictions:
                                if pred_label not in self.labels:
                                    print(f"Prediction label '{pred_label}' not in graph labels.")
                                    continue
                                
                                seen_labels_for_image.add(pred_label)
    
                                if current_label != pred_label:
                                    edge_data = builder.update_graph(
                                        # graph, current_label, pred_label, pred_prob, i, dataset_class
                                        graph, current_label, pred_label, pred_prob, original_index, dataset_class
                                    )
                                    if edge_data is not None:
                                        edges_data.append(edge_data)
                                        added_labels.append(pred_label)
                                        
                        # Using the working version's logic for dissimilarity
                        if self.graph_type == "dissimilarity":
                            for label in self.labels:
                                # if label != current_label:
                                if label not in seen_labels_for_image:                                
                                    builder.add_infinity_edges(
                                        graph, added_labels, label, current_label
                                    )
                
                    batch_images = []
                    true_labels = []
                    original_dataset_positions = []
                    
                # Log progress every 10 batches
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(X)} images")
    
        except Exception as e:
            print(f'Error while preprocessing model: {str(e)}')
            raise
            
        try:
            if self.save_connections:
                self.edges_df = pd.DataFrame(edges_data)

            self.TBD_graph = graph
            heap_processor = HeapProcessor(self.TBD_graph, self.graph_type, self.labels)
            self.heap_processor = heap_processor
            
            # Use the working clustering builder
            clustering = HierarchicalClusteringBuilder(heap_processor, self.labels)
            self.Z = ZBuilder.create_z_matrix_from_tree(clustering, self.labels)
            
            logger.info("Preprocessing completed successfully")
                  
        except Exception as e:
            print(f'Error while preprocessing model 2: {str(e)}')
            raise

    def get_neighbors_by_label_name(self, label_name):
        neighbors = self.TBD_graph.neighbors(label_name)  
        neighbors_names = [self.TBD_graph.vs[n]["name"] for n in neighbors]
        
        data = [
            {"Neighbor": neighbor, "Weight": self.TBD_graph.es[self.TBD_graph.get_eid(label_name, neighbor)]["weight"]}
            for neighbor in neighbors_names
        ]
        
        return pd.DataFrame(data).sort_values(by="Weight", ascending=False)
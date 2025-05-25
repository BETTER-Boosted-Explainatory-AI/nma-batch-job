import tensorflow as tf
import pandas as pd
from igraph import Graph
from enums.graph_types import GraphTypes
from NMA.enums.heap_types import HeapType
from classes.preprocessing.batch_predictor import BatchPredictor
from classes.preprocessing.graph_builder import GraphBuilder
from classes.preprocessing.heap_processor import HeapProcessor
from classes.preprocessing.hierarchical_clustering_builder import HierarchicalClusteringBuilder
from classes.preprocessing.z_builder import ZBuilder

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

    def _preprocessing(self, dataset_class, batch_size):
        try:
            X = dataset_class.x_train
            y = dataset_class.y_train
            
            graph = Graph(directed=False)
            graph.add_vertices(self.labels)
    
            edges_data = []
            
            # Process all images in batches while preserving original indices
            predictor = BatchPredictor(self.model, batch_size=batch_size)
            builder = GraphBuilder(self.graph_type, self.infinity)
            
            # Process images in batches
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                batch_images = X[i:batch_end]
                batch_labels = y[i:batch_end]
                original_indices = list(range(i, batch_end))  # Track the original indices
                
                top_predictions_batch = predictor.get_top_predictions(
                    batch_images, self.labels, self.top_k, self.graph_threshold
                )
    
                for j, top_predictions in enumerate(top_predictions_batch):
                    current_label = batch_labels[j]
                    original_index = original_indices[j]  # Use the correct original index
    
                    seen_labels_for_image = {current_label}
                    
                    if len(top_predictions) == 0:
                        print(f"No predictions for image at index {original_index}")
                        continue
    
                    if len(top_predictions[0]) < 2:
                        continue
    
                    if top_predictions[0][2] > self.min_confidence:
                        filtered_predictions = top_predictions
    
                        for _, pred_label, pred_prob in filtered_predictions:
                            if pred_label not in self.labels:
                                raise ValueError(
                                    f"Prediction label '{pred_label}' not in graph labels."
                                )
                                
                            seen_labels_for_image.add(pred_label)
    
                            if current_label != pred_label:
                                edge_data = builder.update_graph(
                                    graph, current_label, pred_label, pred_prob, original_index, dataset_class
                                )
                                # Only append edge_data if it's not None (not a self-loop)
                                if edge_data is not None:
                                    edges_data.append(edge_data)
                                    
                    if self.graph_type == GraphTypes.DISSIMILARITY.value:
                        for label in self.labels:
                            # Only add infinity edges for labels not seen in this image's predictions
                            if label not in seen_labels_for_image:
                                builder.add_infinity_edges(
                                    graph, label, current_label
                                )
    
            if self.save_connections:
                self.edges_df = pd.DataFrame(edges_data)
    
            self.TBD_graph = graph
            heap_processor = HeapProcessor(self.TBD_graph, self.graph_type, self.labels)
            self.heap_processor = heap_processor
    
            clustering = HierarchicalClusteringBuilder(heap_processor, self.labels)
            self.Z = ZBuilder.create_z_matrix_from_tree(clustering, self.labels)
    
        except Exception as e:
            print(f"Error while preprocessing model: {str(e)}")

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
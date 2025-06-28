class GraphBuilder:
    def __init__(self, graph_type, infinity):
        self.graph_type = graph_type
        self.infinity = infinity
    
    def get_infinity(self):
        return self.infinity

    def get_threshold(self):
        return self.threshold
    
    def create_edge_weight(self, pred_prob):
        if self.graph_type == "dissimilarity":
            return 1 - pred_prob
        elif self.graph_type == "count":
            return 1
        return pred_prob
    
    def update_graph(self, graph, source_label, target_label, probability, image_id, dataset_class):
        if source_label == target_label:
            return None
 
        weight = self.create_edge_weight(probability)

        if graph.are_adjacent(source_label, target_label):
            edge_id = graph.get_eid(source_label, target_label)
            graph.es[edge_id]["weight"] += weight 
        else:
            graph.add_edge(source_label, target_label, weight=weight)
        
        edge_data = {
            "image_id": image_id,
            "source": dataset_class.get_label_readable_name(source_label),
            "target": dataset_class.get_label_readable_name(target_label),
            "target_probability": probability,
        }
        
        return edge_data
    
    
    def add_infinity_edges(self, graph, source_label, target_label):
        if target_label == source_label:
            return
        
        if graph.are_adjacent(source_label, target_label):
            edge_id = graph.get_eid(source_label, target_label)
            graph.es[edge_id]["weight"] += self.infinity
        else:
            graph.add_edge(source_label, target_label, weight=self.infinity) 

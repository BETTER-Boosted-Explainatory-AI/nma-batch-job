import numpy as np
import json
import os
from .tree_node import TreeNode
from NMA.utilss.enums.heap_types import HeapType

class HierarchicalClusteringBuilder:
    def __init__(self, heap_processor, labels):
        self.nodes = {}
        self.next_cluster_id = len(labels)
        self.max_weight = max(abs(weight) for weight,_, _ in heap_processor.get_heap())
        self.forest = []
        self._build(heap_processor, labels)
        
    def _initialize_nodes(self, labels):
        for i, label in enumerate(labels):
            self.nodes[label] = TreeNode(node_id=i, label=label)
    
    def _get_root(self, node):
        while node.parent is not None:
            node = node.parent
        return node
    
    def _build(self, heap_processor, labels):
        self._initialize_nodes(labels)
        n = len(labels)
        distances = np.full((n, n), np.inf)
        np.fill_diagonal(distances, 0)  # Self-distance is 0
        label_to_idx = {label: i for i, label in enumerate(labels)}
        for weight, source, target in heap_processor.get_heap():
            if heap_processor.heap_type == HeapType.MAXIMUM.value:
                distance = self.max_weight - (-1*weight)
            else:
                distance = weight
                
            src_idx = label_to_idx[source]
            tgt_idx = label_to_idx[target]
            
            distances[src_idx, tgt_idx] = distance
            distances[tgt_idx, src_idx] = distance  # Make symmetric
        
        max_dist = np.max(distances[~np.isinf(distances)])
        distances[np.isinf(distances)] = max_dist * 2
        
        active_clusters = {i: [i] for i in range(n)}
        next_id = n
        while len(active_clusters) > 1:
            min_dist = float('inf')
            closest_pair = None
            
            for i in active_clusters:
                for j in active_clusters:
                    if i < j:  # Avoid duplicate pairs
                        dist_sum = 0
                        count = 0
                        for idx1 in active_clusters[i]:
                            for idx2 in active_clusters[j]:
                                dist_sum += distances[idx1, idx2]
                                count += 1
                        
                        avg_dist = dist_sum / count
                        
                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            closest_pair = (i, j)
            
            i, j = closest_pair
            
            if len(active_clusters[i]) == 1:
                node_i = self.nodes[labels[active_clusters[i][0]]]
            else:
                node_i = self.nodes[f"cluster_{i}"]
                
            if len(active_clusters[j]) == 1:
                node_j = self.nodes[labels[active_clusters[j][0]]]
            else:
                node_j = self.nodes[f"cluster_{j}"]
            
            cluster_name = f"cluster_{next_id}"
            new_node = TreeNode(
                node_id=next_id,
                label=cluster_name,
                children=[node_i, node_j],
                weight=min_dist
            )
            
            self.nodes[cluster_name] = new_node
            node_i.parent = new_node
            node_j.parent = new_node
            
            merged_cluster = active_clusters[i] + active_clusters[j]
            active_clusters[next_id] = merged_cluster
            del active_clusters[i]
            del active_clusters[j]
            
            next_id += 1
        
        final_id = list(active_clusters.keys())[0]
        if final_id < n:
            self.final_tree = self.nodes[labels[final_id]]
        else:
            self.final_tree = self.nodes[f"cluster_{final_id}"]
        
        self.forest = [self.final_tree]  # Only one tree in the forest
        
        print(f"Built a forest with {len(self.forest)} tree{'s' if len(self.forest) > 1 else ''}")
    

    def get_forest(self):
        """Return all trees in the forest"""
        return self.forest
    
    def get_tree(self, index=0):
        """Get a specific tree from the forest by index"""
        if 0 <= index < len(self.forest):
            return self.forest[index]
        return None
        
    def forest_size(self):
        """Return the number of trees in the forest"""
        return len(self.forest)




    

import numpy as np
import pickle
import boto3
import os
from NMA.utilss.s3_utils import get_users_s3_client 

class ScoreCalculator: 
    def __init__(self, Z_filename, class_names):
        self.s3_client = get_users_s3_client()
        try:
            if Z_filename.startswith('s3://'):
                parts = Z_filename.replace('s3://', '').split('/', 1)
                bucket = parts[0]
                key = parts[1]
            else:
                bucket = os.getenv("S3_USERS_BUCKET_NAME")
                if not bucket:
                    raise ValueError("S3_USERS_BUCKET_NAME environment variable is required when not using full s3:// path")
                key = Z_filename
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            stream = response['Body']
            self.Z_full = pickle.load(stream)
            
        except Exception as e:
            print(f"Error loading Z file from S3: {e}")
            self.Z_full = None
            
        self.class_names = class_names


    def count_ancestors_to_lca(self, label1, label2):
        if isinstance(label1, str):
            idx1 = self.class_names.index(label1)
        else:
            idx1 = label1
            
        if isinstance(label2, str):
            idx2 = self.class_names.index(label2)
        else:
            idx2 = label2
        
        n_samples = len(self.class_names)
        n_nodes = 2 * n_samples - 1
        parent = np.zeros(n_nodes, dtype=np.int64) - 1  # -1 means no parent
        
        for i, (left, right, height, _) in enumerate(self.Z_full):
            left = int(left)
            right = int(right)
            node_id = n_samples + i
            parent[left] = node_id
            parent[right] = node_id
        
        path1 = []
        current = idx1
        while parent[current] != -1:
            path1.append(parent[current])
            current = parent[current]
        path2 = []
        current = idx2
        lca = None
        
        while parent[current] != -1:
            current_parent = parent[current]
            path2.append(current_parent)
            if current_parent in path1:
                lca = current_parent
                break
            current = current_parent
        
        if lca is None:
            return n_nodes
        steps1 = path1.index(lca) + 1
        steps2 = path2.index(lca) + 1
        total_count = steps1 + steps2
        return total_count
    
    def calculate_adversarial_score(self, predictions, top_k=5):
        if len(predictions.shape) > 1:
            predictions = predictions[0]  # For batch predictions, take the first item
            
            
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_probs = [predictions[i] for i in top_indices]
        top_labels = [self.class_names[i] for i in top_indices]
        pairwise_distances = []
        distance_prob_products = []
        all_pairs = []
        
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                idx1, idx2 = top_indices[i], top_indices[j]
                label1, label2 = top_labels[i], top_labels[j]
                prob1, prob2 = top_probs[i], top_probs[j]
                
                rank_count = self.count_ancestors_to_lca(idx1, idx2)
                print(f"Rank Count for {label1} and {label2}: {rank_count}")
                
                prob_product = prob1 * prob2
                rank_prob = rank_count * prob_product
                
                pair_info = {
                    'label1': label1,
                    'label2': label2,
                    'probability1': float(prob1),
                    'probability2': float(prob2),
                    'ancestor_distance': rank_count,
                    'prob_product': prob_product,
                    'weighted_distance': rank_prob
                }
                
                pairwise_distances.append(rank_count)
                distance_prob_products.append(rank_prob)
                all_pairs.append(pair_info)

        if pairwise_distances:
            sum_distance = sum(pairwise_distances)
            score = sum_distance
        else:
            score = 0
        return score
        

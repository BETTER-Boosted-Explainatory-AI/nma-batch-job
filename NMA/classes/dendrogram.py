from scipy.cluster.hierarchy import to_tree
from ..utilss.wordnet_utils import process_hierarchy
import json
import os
import numpy as np
import pickle
import boto3
import io
from botocore.exceptions import ClientError
import logging
from ..utilss.s3_utils import get_users_s3_client
logger = logging.getLogger(__name__)
S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
if not S3_BUCKET:
    raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")


class Dendrogram:
    def __init__(self, dendrogram_filename, Z=None):
        self.Z = Z
        self.Z_tree_format = None
        self.dendrogram_filename = dendrogram_filename
        self.s3_pickle_key = f"{dendrogram_filename}.pkl"
        self.s3_json_key   = f"{dendrogram_filename}.json"
        
    def _build_tree_format(self, node, labels):
        if node.is_leaf():
            return {
                "id": node.id,
                "name": labels[node.id],
                }
        else:
            return {
                "id": node.id,
                "name": f"Cluster {node.id}",
                "children": [self._build_tree_format(node.get_left(), labels), self._build_tree_format(node.get_right(), labels)],
                "value": node.dist
            }

    def _build_tree_hierarchy(self, linkage_matrix, labels):
        tree, nodes = to_tree(linkage_matrix, rd=True)
        self.Z_tree_format = self._build_tree_format(tree, labels)
        self.Z_tree_format = process_hierarchy(self.Z_tree_format)   
        return self.Z_tree_format  
    
    def filter_dendrogram_by_labels(self, full_data, target_labels):
        def contains_target_label(node):
            if 'children' not in node:
                return node.get('name') in target_labels
            
            for child in node.get('children', []):
                if contains_target_label(child):
                    return True
            return False
        
        def filter_tree(node):
            if not contains_target_label(node):
                return None
            
            new_node = {
                'id': node.get('id'),
                'name': node.get('name')
            }
            
            if 'value' in node:
                new_node['value'] = node.get('value')
            
            if 'children' not in node:
                return new_node
            
            filtered_children = []
            for child in node.get('children', []):
                filtered_child = filter_tree(child)
                if filtered_child:
                    filtered_children.append(filtered_child)
                    
            if filtered_children:
                new_node['children'] = filtered_children
            
            return new_node
        
        return filter_tree(full_data)

    def merge_clusters(self, node):
        if node is None:
            return {"id": 0, "name": "Empty Node"}
        
        if "children" not in node:
            return node
        
        merged_children = []
        for child in node["children"]:
            merged_child = self.merge_clusters(child)
            if merged_child:
                merged_children.append(merged_child)
        
        if all(c.get("value", 0) == 100 for c in merged_children):
            node["children"] = [grandchild for child in merged_children for grandchild in child.get("children", [])]
        else:
            node["children"] = merged_children
        
        if len(node["children"]) == 1:
            return node["children"][0]
        
        return node
    
    def get_sub_dendrogram_formatted(self, selected_labels):
        filtered_tree = self.filter_dendrogram_by_labels(self.Z_tree_format, selected_labels)
        filtered_tree = self.merge_clusters(filtered_tree)
        filtered_tree_json = json.dumps(filtered_tree, indent=2)
        return filtered_tree_json
    
    
    def save_dendrogram(self, linkage_matrix=None):
        try:
            if linkage_matrix is not None:
                upload_pickle_to_s3(linkage_matrix, S3_BUCKET, self.s3_pickle_key)
                print(f"Linkage matrix saved to {S3_BUCKET}/{self.s3_pickle_key}")
            
            if self.Z_tree_format is not None:
                upload_json_to_s3(self.Z_tree_format, S3_BUCKET, self.s3_json_key)
                print(f"Tree format saved to s3://{S3_BUCKET}/{self.s3_json_key}")
            
            return f"{S3_BUCKET}/{self.s3_pickle_key}", f"{S3_BUCKET}/{self.s3_json_key}"
            
        except Exception as e:
            logger.error(f"Error saving dendrogram to S3: {e}")
            raise
        
    def load_dendrogram(self):
        try:
            if s3_file_exists(S3_BUCKET, self.s3_pickle_key):
                self.Z = download_pickle_from_s3(S3_BUCKET, self.s3_pickle_key)
                print(f"Linkage matrix loaded from s3://{S3_BUCKET}/{self.s3_pickle_key}")
            else:
                print(f"Pickle file not found: s3://{S3_BUCKET}/{self.s3_pickle_key}")
                self.Z = None
            
            if s3_file_exists(S3_BUCKET, self.s3_json_key):
                self.Z_tree_format = download_json_from_s3(S3_BUCKET, self.s3_json_key)
                print(f"Tree format loaded from s3://{S3_BUCKET}/{self.s3_json_key}")
            else:
                print(f"JSON file not found: s3://{S3_BUCKET}/{self.s3_json_key}")
                self.Z_tree_format = None
            return self
        except Exception as e:
            logger.error(f"Error loading dendrogram from S3: {e}")
            raise
    
    
    
    def find_name_hierarchy(self, node, target_name):
        if node.get('name') == target_name:
            return [target_name]
        
        if 'children' in node:
            for child in node['children']:
                result = self.find_name_hierarchy(child, target_name)
                
                if result is not None:
                    if node.get('name'):
                        result.append(node['name'])
                    return result
        return None
        
    def rename_cluster(self, cluster_id, new_name):
        print(f"Renaming cluster {cluster_id} to {new_name}")

        def collect_names(node, names):
            names.add(node.get('name'))
            for child in node.get('children', []):
                collect_names(child, names)
        existing_names = set()
        collect_names(self.Z_tree_format, existing_names)

        if new_name in existing_names:
            print(f"Name '{new_name}' already exists. Rename aborted.")
            return None

        def find_node(node):
            if node.get('id') == cluster_id:
                return node
            for child in node.get('children', []):
                result = find_node(child)
                if result:
                    return result
            return None

        target_node = find_node(self.Z_tree_format)
        if target_node is None:
            print(f"Cluster ID {cluster_id} not found.")
            return None

        if 'children' not in target_node or not target_node['children']:
            print(f"Cluster ID {cluster_id} is a leaf. Rename aborted.")
            return None

        target_node['name'] = new_name
        return self.Z_tree_format
    
    
def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
    s3_client = get_users_s3_client() 
    print(f"Checking if file exists in S3: {bucket_name}/{s3_key}")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File found: {bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"File not found: {bucket_name}/{s3_key}, Error: {str(e)}")
        return False


def upload_json_to_s3(data: dict, bucket_name: str, s3_key: str):
    """Upload JSON data directly to S3"""
    s3_client = get_users_s3_client() 
    try:
        json_string = json.dumps(data, indent=2)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json_string.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"JSON data uploaded to s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        logger.error(f"Error uploading JSON to S3: {e}")
        raise

def download_json_from_s3(bucket_name: str, s3_key: str) -> dict:
    """Download and parse JSON data from S3"""
    s3_client = get_users_s3_client() 
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        logger.error(f"Error downloading JSON from S3: {e}")
        raise

def upload_pickle_to_s3(data, bucket_name: str, s3_key: str):
    """Upload pickle data directly to S3"""
    s3_client = get_users_s3_client() 
    try:
        pickle_bytes = pickle.dumps(data)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=pickle_bytes,
            ContentType='application/octet-stream'
        )
        logger.info(f"Pickle data uploaded to s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        logger.error(f"Error uploading pickle to S3: {e}")
        raise

def download_pickle_from_s3(bucket_name: str, s3_key: str):
    """Download and deserialize pickle data from S3"""
    s3_client = get_users_s3_client() 
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        pickle_bytes = response['Body'].read()
        return pickle.loads(pickle_bytes)
    except ClientError as e:
        logger.error(f"Error downloading pickle from S3: {e}")
        raise
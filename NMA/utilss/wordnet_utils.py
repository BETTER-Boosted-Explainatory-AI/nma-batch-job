from nltk.corpus import wordnet as wn
import os
import boto3
import re
from utilss.s3_utils import get_datasets_s3_client
 
# def convert_folder_names_to_readable_labels(dataset_path):
#     if not os.path.exists(dataset_path):
#         raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
#     # Get all class names (subdirectories)
#     class_names = sorted([
#         d for d in os.listdir(dataset_path)
#         if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('n') and d[1:].isdigit()
#     ])
    
#     if not class_names:
#         raise ValueError(f"No subdirectories found in {dataset_path}")
    
#     # Define special case mapping for disambiguation
#     special_case_mapping = {
#         "n02012849": "crane_bird",       # Crane bird
#         "n03126707": "crane_machine",    # Crane machine
#         "n03710637": "maillot",          # Maillot (swimsuit)
#         "n03710721": "tank_suit"         # Tank suit (different type of swimsuit)
#     }
    
#     # Create mapping from folder names to readable labels
#     folder_to_label = {}
#     for folder_name in class_names:
#         if folder_name in special_case_mapping:
#             # Use special case mapping for known ambiguous folders
#             readable_label = special_case_mapping[folder_name]
#         elif folder_name.startswith('n'):
#             # Use WordNet utility for standard cases
#             readable_label = WordnetUtils.convert_folder_name_to_label(folder_name)
#         else:
#             readable_label = folder_name
            
#         folder_to_label[folder_name] = readable_label
    
#     # Convert original folder names to readable labels while preserving order
#     readable_labels = [folder_to_label[folder_name] for folder_name in class_names]
    
#     print(f"Sample conversions:")
#     for i in range(min(5, len(class_names))):
#         print(f"{class_names[i]} → {readable_labels[i]}")
    
#     # Print specific indices for debugging
#     if len(readable_labels) > 638:
#         print(f"Label at index 638: {readable_labels[638]}")
#     if len(readable_labels) > 639:
#         print(f"Label at index 639: {readable_labels[639]}")
        
#     return readable_labels, folder_to_label


def convert_folder_names_to_readable_labels(dataset_path):
    """
    Convert folder names in S3 to readable labels using WordNet.
    
    Parameters:
    - dataset_path: S3 path in the format 's3://bucket/prefix' or just 'prefix'
                   (will use S3_USERS_BUCKET_NAME env var)
    
    Returns:
    - readable_labels: List of readable class labels
    - folder_to_label: Dictionary mapping folder names to readable labels
    """
    # Initialize S3 client
    s3_client = get_datasets_s3_client()
    
    # Parse the S3 path
    if dataset_path.startswith('s3://'):
        parts = dataset_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
    else:
        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            raise ValueError("S3_BUCKET_NAME environment variable is required when not using full s3:// path")
        prefix = dataset_path
    
    # Ensure prefix ends with '/'
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'
    
    # Check if the prefix exists in the bucket
    try:
        s3_client.head_object(Bucket=bucket, Key=prefix)
    except:
        # Try listing objects to see if prefix exists as a directory
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/', MaxKeys=1)
        if 'CommonPrefixes' not in response and 'Contents' not in response:
            raise ValueError(f"Dataset path does not exist in S3: {bucket}/{prefix}")
    
    # Get all class names (subdirectories)
    class_names = []
    
    # Use delimiter to list "directories" in S3
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    for page in pages:
        if 'CommonPrefixes' in page:
            for common_prefix in page['CommonPrefixes']:
                # Extract the directory name from the prefix
                dir_name = common_prefix['Prefix'].rstrip('/')
                dir_name = dir_name.split('/')[-1]  # Get the last part of the path
                
                # Check if it follows the pattern 'n' followed by digits
                if dir_name.startswith('n') and dir_name[1:].isdigit():
                    class_names.append(dir_name)
    
    class_names.sort()
    
    if not class_names:
        raise ValueError(f"No subdirectories found in S3 path {bucket}/{prefix}")
    
    # Define special case mapping for disambiguation
    special_case_mapping = {
        "n02012849": "crane_bird",       # Crane bird
        "n03126707": "crane_machine",    # Crane machine
        "n03710637": "maillot",          # Maillot (swimsuit)
        "n03710721": "tank_suit"         # Tank suit (different type of swimsuit)
    }
    
    # Create mapping from folder names to readable labels
    folder_to_label = {}
    for folder_name in class_names:
        if folder_name in special_case_mapping:
            # Use special case mapping for known ambiguous folders
            readable_label = special_case_mapping[folder_name]
        elif folder_name.startswith('n'):
            # Use WordNet utility for standard cases
            readable_label = WordnetUtils.convert_folder_name_to_label(folder_name)
        else:
            readable_label = folder_name
            
        folder_to_label[folder_name] = readable_label
    
    # Convert original folder names to readable labels while preserving order
    readable_labels = [folder_to_label[folder_name] for folder_name in class_names]
    
    print(f"Sample conversions:")
    for i in range(min(5, len(class_names))):
        print(f"{class_names[i]} → {readable_labels[i]}")
    
    # Print specific indices for debugging
    if len(readable_labels) > 638:
        print(f"Label at index 638: {readable_labels[638]}")
    if len(readable_labels) > 639:
        print(f"Label at index 639: {readable_labels[639]}")
        
    return readable_labels, folder_to_label
    
    
    
def folder_name_to_number(folder_name):
    synsets = wn.synsets(folder_name)
    if synsets:
        offset = synsets[0].offset()        
        folder_number = 'n{:08d}'.format(offset)
        return folder_number
    
def common_group(groups):
    common_hypernyms = []
    hierarchy = {}
    
    for group in groups:
        hierarchy[group] = []
        synsets = wn.synsets(group)
        if synsets:
            hypernyms = synsets[0].hypernym_paths()
            for path in hypernyms:
                hierarchy[group].extend([node.name().split('.')[0] for node in path])
                
    if len(hierarchy) == 1:
        common_hypernyms = list(hierarchy.values())[0]
    else:
        for hypernym in hierarchy[groups.pop()]:
            if all(hypernym in hypernyms for hypernyms in hierarchy.values()):
                common_hypernyms.append(hypernym)
    
    return common_hypernyms[::-1]

def get_all_leaf_names(node):
    """Extract all leaf node names from a cluster hierarchy."""
    if "children" not in node:
        # Only return actual object names, not cluster names
        if "Cluster" not in node["name"]:
            return [node["name"]]
        return []
    
    names = []
    for child in node["children"]:
        names.extend(get_all_leaf_names(child))
    return names

def find_common_hypernyms(words, abstraction_level=0):
    """Find common hypernyms for a list of words, allowing abstraction level selection, and avoid leaf duplicates."""
    from nltk.corpus import wordnet as wn
    
    clean_words = [word.replace('_', ' ').lower() for word in words if "Cluster" not in word]
    if len(clean_words) < 2:
        return None

    synsets_list = []
    for word in clean_words:
        synsets = wn.synsets(word, pos=wn.NOUN)
        if synsets:
            synsets_list.append(synsets[0])
    if len(synsets_list) >= 2:
        all_hypernyms = []
        for i in range(len(synsets_list) - 1):
            for j in range(i + 1, len(synsets_list)):
                lchs = synsets_list[i].lowest_common_hypernyms(synsets_list[j])
                for lch in lchs:
                    name = lch.name().split('.')[0].replace('_', ' ')
                    depth = lch.min_depth()
                    # Don't use the leaf name or entity as cluster name
                    if name.lower() not in clean_words and name.lower() != 'entity':
                        all_hypernyms.append((name, depth))

        if all_hypernyms:
            # Sort by depth descending (more specific first)
            all_hypernyms.sort(key=lambda x: x[1], reverse=True)
            # Use abstraction_level to pick a more general name if needed
            idx = min(abstraction_level, len(all_hypernyms) - 1)
            return all_hypernyms[idx][0]

    return None

def improved_rename_clusters(node, depth=0, used_names=None, all_leaf_names=None):
    """
    Rename clusters based on WordNet hypernyms while ensuring they differ from leaf nodes.
    Uses more abstract hypernyms when a name conflicts with leaf nodes.
    """
    if used_names is None:
        used_names = set()
    
    if all_leaf_names is None:
        # On first call, collect all leaf names in the entire tree
        all_leaves = get_all_leaf_names(node)
        all_leaf_names = {name.lower() for name in all_leaves}
    
    # Recursively rename children first
    if "children" in node:
        for i, child in enumerate(node["children"]):
            node["children"][i] = improved_rename_clusters(child, depth + 1, used_names, all_leaf_names)

    # Rename cluster if applicable
    if "Cluster" in node["name"]:
        leaf_names = get_all_leaf_names(node)
        if not leaf_names and "children" in node:
            child_names = [child["name"] for child in node["children"] if "Cluster" not in child["name"]]
            if child_names:
                leaf_names = child_names

        # Try different abstraction levels to find a suitable hypernym that's not a leaf name
        abstraction_level = 0
        max_attempts = 5  # Try more levels of abstraction
        
        while abstraction_level < max_attempts:
            new_name = find_common_hypernyms(leaf_names, abstraction_level)
            
            if new_name:
                # If the proposed name is too similar to a leaf, try a more abstract hypernym
                if new_name.lower() in all_leaf_names:
                    abstraction_level += 1
                    continue
                
                # Make the name unique if it's already used
                base_name = new_name.capitalize()
                unique_name = base_name
                counter = 1
                while unique_name.lower() in {name.lower() for name in used_names}:
                    counter += 1
                    unique_name = f"{base_name} {counter}"
                
                node["name"] = unique_name
                used_names.add(unique_name)
                break
            else:
                # No more hypernyms at this level, try next level
                abstraction_level += 1
        
        # If no suitable hypernym is found after all attempts,
        # keep the original "Cluster {id}" name
    return node

def process_hierarchy(hierarchy_data):
    """Process the entire hierarchy, renaming clusters while preserving structure."""
    return improved_rename_clusters(hierarchy_data)

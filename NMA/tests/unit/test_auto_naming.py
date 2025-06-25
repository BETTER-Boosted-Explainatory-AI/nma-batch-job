#!/usr/bin/env python3
"""
Completely objective test of WordNet functions on real S3 dendrogram data.
No bias, no hints, no pre-selected categories.
Just raw performance metrics.
"""

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import os
import json
import copy
from botocore.exceptions import ClientError

def download_dendrogram_from_s3():
    """Download the dendrogram from S3"""
    try:
        from NMA.utilss.s3_utils import get_users_s3_client
        
        s3_client = get_users_s3_client()
        bucket = "better-xai-users"
        # key = "4225e444-c031-7077-1026-cb470f0a8a98/33b8e5d1-c7d2-4c35-8a29-5764e8dd86fd/dissimilarity/dendrogram.json" # cifar100
        key = "4225e444-c031-7077-1026-cb470f0a8a98/496105c6-8406-47eb-b2ff-00f0fd532d38/dissimilarity/dendrogram_test.json" # imagenet

        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
        
    except Exception as e:
        print(f"❌ Error downloading dendrogram: {e}")
        return None

def collect_all_cluster_nodes(data):
    """Collect all cluster nodes that could potentially be renamed"""
    clusters = []
    
    def traverse(node):
        # Collect any node that has children (potential cluster)
        if "children" in node and len(node["children"]) > 0:
            clusters.append({
                "id": node.get("id"),
                "original_name": node.get("name"),
                "children_count": len(node["children"]),
                "node_ref": node  # Keep reference for testing
            })
        
        # Recurse to children
        if "children" in node:
            for child in node["children"]:
                traverse(child)
    
    traverse(data)
    return clusters

def get_leaf_names_for_node(node):
    """Get all leaf names under a node"""
    leaf_names = []
    
    def collect_leaves(n):
        if "children" not in n or len(n["children"]) == 0:
            # This is a leaf
            leaf_names.append(n.get("name", ""))
        else:
            # This has children, recurse
            for child in n["children"]:
                collect_leaves(child)
    
    collect_leaves(node)
    return [name for name in leaf_names if name]  # Filter out empty names

def test_wordnet_function_on_cluster(cluster_info):
    """Test WordNet function on a single cluster objectively"""
    try:
        from NMA.utilss.wordnet_utils import find_common_hypernyms_improved
        
        # Get leaf names for this cluster
        leaf_names = get_leaf_names_for_node(cluster_info["node_ref"])
        
        if len(leaf_names) < 2:
            return {
                "success": False,
                "reason": "insufficient_leaves",
                "leaf_count": len(leaf_names),
                "suggested_name": None,
                "leaf_names": leaf_names
            }
        
        # Test the function with no hints or bias
        suggested_name = find_common_hypernyms_improved(leaf_names, 0, debug=False)
        
        return {
            "success": suggested_name is not None,
            "reason": "function_result" if suggested_name else "no_hypernym_found",
            "leaf_count": len(leaf_names),
            "suggested_name": suggested_name,
            "leaf_names": leaf_names[:10]  # Limit for display
        }
        
    except Exception as e:
        return {
            "success": False,
            "reason": f"function_error: {str(e)}",
            "leaf_count": 0,
            "suggested_name": None,
            "leaf_names": []
        }

def run_objective_test(data):
    """Run completely objective test on all clusters"""
    print("🔬 OBJECTIVE WORDNET FUNCTION TEST")
    print("=" * 50)
    print("Testing WordNet functions on ALL clusters without bias...")
    
    # Collect all potential clusters
    all_clusters = collect_all_cluster_nodes(data)
    print(f"📊 Found {len(all_clusters)} total cluster nodes")
    
    # Test statistics
    stats = {
        "total_tested": 0,
        "function_succeeded": 0,
        "function_failed": 0,
        "insufficient_leaves": 0,
        "function_errors": 0,
        "results": []
    }
    
    # Test each cluster objectively
    for i, cluster in enumerate(all_clusters):
        if i >= 100:  # Limit to first 100 for performance
            break
            
        result = test_wordnet_function_on_cluster(cluster)
        result["original_name"] = cluster["original_name"]
        result["cluster_id"] = cluster["id"]
        
        stats["total_tested"] += 1
        stats["results"].append(result)
        
        if result["success"]:
            stats["function_succeeded"] += 1
        elif result["reason"] == "insufficient_leaves":
            stats["insufficient_leaves"] += 1
        elif "function_error" in result["reason"]:
            stats["function_errors"] += 1
        else:
            stats["function_failed"] += 1
    
    return stats

def analyze_results_objectively(stats):
    """Analyze results with pure objective metrics"""
    print(f"\n📈 OBJECTIVE RESULTS ANALYSIS")
    print("=" * 50)
    
    total = stats["total_tested"]
    succeeded = stats["function_succeeded"]
    failed = stats["function_failed"]
    insufficient = stats["insufficient_leaves"]
    errors = stats["function_errors"]
    
    # Basic success rate
    if total > 0:
        success_rate = (succeeded / total) * 100
        print(f"📊 Overall Statistics:")
        print(f"   Total clusters tested: {total}")
        print(f"   Function returned result: {succeeded} ({success_rate:.1f}%)")
        print(f"   Function returned None: {failed} ({(failed/total)*100:.1f}%)")
        print(f"   Insufficient leaves (< 2): {insufficient} ({(insufficient/total)*100:.1f}%)")
        print(f"   Function errors: {errors} ({(errors/total)*100:.1f}%)")
    
    # Analyze successful results
    successful_results = [r for r in stats["results"] if r["success"]]
    if successful_results:
        print(f"\n✅ SUCCESSFUL NAMING ATTEMPTS:")
        print(f"   Sample of generated names:")
        
        # Show a random sample without bias
        import random
        sample_size = min(10, len(successful_results))
        sample = random.sample(successful_results, sample_size) if len(successful_results) > sample_size else successful_results
        
        for result in sample:
            print(f"      • '{result['suggested_name']}' (from {result['leaf_count']} leaves)")
    
    # Analyze failed results  
    failed_results = [r for r in stats["results"] if not r["success"] and r["reason"] == "no_hypernym_found"]
    if failed_results:
        print(f"\n❌ FAILED NAMING ATTEMPTS:")
        print(f"   {len(failed_results)} clusters where function returned None")
        print(f"   Sample leaf combinations that failed:")
        
        import random
        sample_size = min(5, len(failed_results))
        sample = random.sample(failed_results, sample_size) if len(failed_results) > sample_size else failed_results
        
        for result in sample:
            print(f"      • {result['leaf_count']} leaves: {result['leaf_names'][:3]}...")
    
    # Function error analysis
    error_results = [r for r in stats["results"] if "function_error" in r["reason"]]
    if error_results:
        print(f"\n⚠️  FUNCTION ERRORS:")
        error_types = {}
        for result in error_results:
            error_type = result["reason"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            print(f"   • {error_type}: {count} occurrences")

def test_full_hierarchy_processing(original_data):
    """Test the full process_hierarchy function objectively"""
    print(f"\n🔬 TESTING FULL HIERARCHY PROCESSING")
    print("=" * 50)
    
    # Count original cluster names
    def count_generic_names(node):
        count = 0
        if "Cluster" in node.get("name", ""):
            count += 1
        if "children" in node:
            for child in node["children"]:
                count += count_generic_names(child)
        return count
    
    original_generic_count = count_generic_names(original_data)
    print(f"📊 Original generic 'Cluster' names: {original_generic_count}")
    
    # Apply processing
    print("🔄 Applying process_hierarchy function...")
    try:
        from NMA.utilss.wordnet_utils import process_hierarchy
        test_data = copy.deepcopy(original_data)
        processed_data = process_hierarchy(test_data, debug=False)
        
        # Count after processing
        processed_generic_count = count_generic_names(processed_data)
        improved_count = original_generic_count - processed_generic_count
        
        print(f"📊 After processing:")
        print(f"   Remaining generic names: {processed_generic_count}")
        print(f"   Successfully renamed: {improved_count}")
        
        if original_generic_count > 0:
            improvement_rate = (improved_count / original_generic_count) * 100
            print(f"   Improvement rate: {improvement_rate:.1f}%")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Error in process_hierarchy: {e}")
        return None

def main():
    """Run completely objective test"""
    print("🧪 COMPLETELY OBJECTIVE WORDNET TEST")
    print("=" * 60)
    print("No bias, no hints, no pre-selected categories.")
    print("Pure function performance on real data.")
    print("=" * 60)
    
    # Download data
    data = download_dendrogram_from_s3()
    if not data:
        print("❌ Could not download test data")
        return
    
    # Test individual function objectively
    stats = run_objective_test(data)
    
    # Analyze results objectively
    analyze_results_objectively(stats)
    
    # Test full hierarchy processing
    processed_data = test_full_hierarchy_processing(data)
    
    # Save results for inspection
    if processed_data:
        with open("original_dendrogram.json", "w") as f:
            json.dump(data, f, indent=2)
        with open("improved_dendrogram.json", "w") as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"\n💾 Raw results saved:")
        print(f"   original_dendrogram.json")  
        print(f"   improved_dendrogram.json")
    
    print(f"\n" + "=" * 60)
    print("✅ OBJECTIVE TEST COMPLETED")
    print("=" * 60)
    print("Results are based purely on function performance.")
    print("No bias or hints were provided to the functions.")

if __name__ == "__main__":
    main()
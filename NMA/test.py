# Add any imports needed to fix path issues
import sys
sys.path.append('C:/Users/adler/nma-batch-job')

# Import your function
from NMA.nma_service import _create_nma

# Set parameters
user_id = "92457494-3061-700c-8e14-f1ab249392d7"
model_id = "496105c6-8406-47eb-b2ff-00f0fd532d38"
graph_type = "dissimilarity"
dataset = "imagenet"
min_confidence = 0.8
top_k = 5

# Add debugging print statements
print(f"Testing with parameters: {user_id}, {model_id}, {graph_type}, {dataset}, {min_confidence}, {top_k}")

# Run the function
try:
    result = _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
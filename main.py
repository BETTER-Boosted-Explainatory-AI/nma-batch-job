import os
from NMA.services.nma_service import _create_nma
import sys
sys.path.append('/app')
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    user_id = os.getenv("user_id")
    model_id = os.getenv("model_id")
    graph_type = os.getenv("graph_type")
    dataset = os.getenv("dataset")
    min_confidence = float(os.getenv("min_confidence", 0.5))
    top_k = int(os.getenv("top_k", 4))
    result = _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k)

if __name__ == "__main__":
    main()
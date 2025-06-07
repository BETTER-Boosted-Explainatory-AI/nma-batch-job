import os
from NMA.services.nma_service import _create_nma
from dotenv import load_dotenv
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

sys.stdout.flush()
sys.stderr.flush()

def main():
    load_dotenv()
    
    user_id = os.getenv("user_id")
    model_id = os.getenv("model_id")
    graph_type = os.getenv("graph_type")
    dataset = os.getenv("dataset")
    min_confidence_raw = os.getenv("min_confidence", "0.5")
    min_confidence = float(min_confidence_raw)
    top_k = int(os.getenv("top_k", 4))
    _create_nma(user_id, model_id, graph_type, dataset, min_confidence, top_k)

if __name__ == "__main__":
    main()
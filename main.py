import os
from NMA.nma_service import _create_nma

def main():
    model_file = os.getenv("model_file")
    dataset = os.getenv("dataset")
    graph_type = os.getenv("graph_type")
    model_id = os.getenv("model_id")
    min_confidence = float(os.getenv("min_confidence", 0.5))
    top_k = int(os.getenv("top_k", 5))
    result = _create_nma(model_file, graph_type, dataset, user=None, min_confidence=min_confidence, top_k=top_k, model_id_md=model_id)
    print("NMA result:", result)

if __name__ == "__main__":
    main()
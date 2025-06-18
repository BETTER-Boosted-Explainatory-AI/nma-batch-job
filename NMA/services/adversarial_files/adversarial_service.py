from services.adversarial_files.adversarial_dataset import AdversarialDataset
from services.adversarial_files.adversarial_detector import AdversarialDetector
from services.model_service import get_user_models_info, get_model_files
import logging
logger = logging.getLogger(__name__)


def _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_filename, dataset) -> AdversarialDataset:
    logger.info("Creating adversarial dataset")
    adversarial_dataset = AdversarialDataset(Z_file, clean_images, adversarial_images, model_filename, dataset)
    X_train, y_train, X_test, y_test = adversarial_dataset.create_logistic_regression_dataset()
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}



def create_logistic_regression_detector(model_id, graph_type, clean_images, adversarial_images, user_id):
    model_info = get_user_models_info(user_id, model_id)
    
    print(f"DEBUG: model_info = {model_info}")

    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user_id, model_info, graph_type)
        model_graph_folder = model_files["model_graph_folder"]
        model_file = model_files["model_file"]
        print("model_files", model_files)
        print("model_files[\"Z_file\"]\"", model_files["Z_file"])
        Z_file = model_files["Z_file"]
        
    adversarial_detector = AdversarialDetector(model_graph_folder)
    adversarial_dataset = _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_file, model_info["dataset"])
    adversarial_detector.train_adversarial_detector(adversarial_dataset)

    return adversarial_detector
    
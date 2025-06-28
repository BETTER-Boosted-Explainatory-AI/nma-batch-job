import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

# Force eager execution and configure memory growth
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Configure GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth configuration failed: {e}")

# Set memory limit to prevent silent failures
try:
    tf.config.set_logical_device_configuration(
        gpus[0] if gpus else None,
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
    )
except:
    pass  # Ignore if no GPU available

import os
from NMA.services.nma_service import _create_nma
from dotenv import load_dotenv
import sys

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
    
    
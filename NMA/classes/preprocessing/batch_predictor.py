import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import UpSampling2D
import logging

logger = logging.getLogger(__name__)

class BatchPredictor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.buffer_images = []  # To store images
        self.buffer_labels = []  # To store corresponding labels
        self.buffer_results = []  # To store batch results

        self.has_upsampling = self._check_for_upsampling()
        
        # Ensure model is compiled for eager execution
        if not self.model.built:
            logger.warning("Model not built, this might cause issues")
        
        # Force eager execution for this model
        self.model.run_eagerly = True


    def _check_for_upsampling(self):
        """Check if the model contains UpSampling2D layers"""
        for layer in self.model.layers:
            if isinstance(layer, UpSampling2D):
                return True
            # Check nested Sequential models
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, UpSampling2D):
                        return True
        return False

    def get_top_predictions(self, X, labels, top_k, graph_threshold):
        try:
            if self.has_upsampling:
                X = tf.image.resize(X, (32, 32))
            else:
                X = tf.image.resize(X, (224, 224))
                
            X = preprocess_input(X)           

            # Convert to numpy and ensure proper shape
            X_np = np.array(X)
            logger.debug(f"Input shape: {X_np.shape}")
            
            # Use smaller batch size if memory is an issue
            if len(X_np) > 16:  # Reduce batch size for large inputs
                batch_preds = []
                for i in range(0, len(X_np), 16):
                    batch = X_np[i:i+16]
                    pred = self.model.predict(batch, verbose=0)
                    batch_preds.extend(pred)
                batch_preds = np.array(batch_preds)
            else:
                batch_preds = self.model.predict(X_np, verbose=0)
            
            logger.debug(f"Predictions shape: {batch_preds.shape}")
            
            batch_results = []
            for pred in batch_preds:
                try:
                    top_indices = pred.argsort()[-top_k:][::-1]

                    valid_indices = [i for i in top_indices if i < len(labels)]

                    top_predictions = [
                        (i, labels[i], pred[i])
                        for i in valid_indices
                        if pred[i] >= graph_threshold
                    ]

                    batch_results.append(top_predictions)
                except Exception as e:
                    logger.error(f"Error processing prediction: {e}")
                    batch_results.append([])

            return batch_results
            
        except Exception as e:
            logger.error(f"Error in get_top_predictions: {e}")
            # Return empty results instead of failing silently
            return [[] for _ in range(len(X))]


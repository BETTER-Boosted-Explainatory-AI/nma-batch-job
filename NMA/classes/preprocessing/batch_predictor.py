import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import UpSampling2D


class BatchPredictor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.buffer_images = []  # To store images
        self.buffer_labels = []  # To store corresponding labels
        self.buffer_results = []  # To store batch results

        self.has_upsampling = self._check_for_upsampling()


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
        # the only change: 
        
        if self.has_upsampling:
            # For models with 3x UpSampling2D (8x total): use 32x32
            # This gives us 32 * 8 = 256, perfect for ResNet50
            X = tf.image.resize(X, (32, 32))
        else:
            # For standard models: use 224x224 (ImageNet standard)
            X = tf.image.resize(X, (224, 224))
            
        #################
        X = preprocess_input(X)           

        
        batch_preds = self.model.predict(np.array(X), verbose=0)
        batch_results = []
        for pred in batch_preds:
            # Get the top k indices
            top_indices = pred.argsort()[-top_k:][::-1]

            # Filter indices that are within the valid range of your labels
            valid_indices = [i for i in top_indices if i < len(labels)]

            top_predictions = [
                (i, labels[i], pred[i])
                for i in valid_indices
                if pred[i] >= graph_threshold
            ]

            batch_results.append(top_predictions)

        return batch_results


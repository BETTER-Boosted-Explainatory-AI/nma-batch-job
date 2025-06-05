# import numpy as np
# import tensorflow as tf


# class BatchPredictor:
#     def __init__(self, model, batch_size=32):
#         self.model = model
#         self.batch_size = batch_size
#         self.buffer_images = []  # To store images
#         self.buffer_labels = []  # To store corresponding labels
#         self.buffer_results = []  # To store batch results

#     def get_top_predictions(self, X, labels, top_k, graph_threshold):
#         batch_preds = self.model.predict(np.array(X), verbose=0)
#         batch_results = []
#         for pred in batch_preds:
#             # Get the top k indices
#             top_indices = pred.argsort()[-top_k:][::-1]

#             # Filter indices that are within the valid range of your labels
#             valid_indices = [i for i in top_indices if i < len(labels)]

#             top_predictions = [
#                 (i, labels[i], pred[i])
#                 for i in valid_indices
#                 if pred[i] >= graph_threshold
#             ]

#             batch_results.append(top_predictions)

#         return batch_results



import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input



class BatchPredictor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.buffer_images = []  # To store images
        self.buffer_labels = []  # To store corresponding labels
        self.buffer_results = []  # To store batch results

    def get_top_predictions(self, X, labels, top_k, graph_threshold):
        
        X = tf.image.resize(X, (224, 224))     # <- use the same name
        X = preprocess_input(X)                # cast + scale
        batch_preds = self.model.predict(X, verbose=0)
        
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
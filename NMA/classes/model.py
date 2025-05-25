import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics import f1_score
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from .datasets.TBD_imagenet_batch_predictor import ImageNetBatchPredictor
# from .datasets.TBD_cifar100_batch_predictor import Cifar100BatchPredictor
import time

class Model:
    def __init__(self, model, top_k, min_confidence, model_filename, dataset):
        # MODELS_PATH = os.getenv("MODELS_PATH")
        # model_file_path = f'{MODELS_PATH}/{model_filename}.keras'
        
        self.model = model
        # self.model_filename = model_file_path
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.size = (256, 256)
        self.accuracy = -1
        self.loss = -1
        self.f1score = -1
        self.dataset = dataset

## Need to change the load and save models to aws s3 bucket in the future
    def load_model(self):
        if os.path.exists(self.model_filename):
            self.model = tf.keras.models.load_model(self.model_filename)
            print(f'Model {self.model_filename} has been loaded')
        else:
            print(f'Model {self.model_filename} does not exist')

    def save_model(self):
        self.model.save(self.model_filename)
        print(f'Model has been saved: {self.model_filename}') 
    
    def model_evaluation(self, x_test, y_test):
        if self.accuracy != -1 & self.loss != -1:
            return self.accuracy
        batch_size = 64
        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.loss, self.accuracy = self.model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
        print(f'Model accuracy: {self.accuracy:.4f}')
        print(f'Model loss: {self.loss:.4f}')
        
        return self.accuracy
    
    def model_evaluate_imagenet(self, test_data_dir):
        ## Not working yet
        img_height, img_width = 224, 224
        batch_size = 32

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Evaluate the model
        start_time = time.time()

        self.loss, self.accuracy = self.model.evaluate(
            test_generator,
            steps=test_generator.samples // batch_size,
            verbose=1
        )

        elapsed_time = time.time() - start_time

        # Print results
        print(f"Loss: {self.loss:.4f}")
        print(f"Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"Evaluation completed in {elapsed_time:.2f} seconds")

    
    def model_accuracy_selected(self, dataset_instance, selected_labels):       
        # works for cifar100 dataset 
        x_test = dataset_instance.x_test
        y_test = dataset_instance.y_test

        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)

        # validation for specific labels accuracy using mask
        selected_indices = [selected_labels.index(label) for label in selected_labels]

        # Create masks for train and test sets
        test_mask = np.isin(np.argmax(y_test, axis=1), selected_indices)

        # Filter x_test, and y_test
        x_test_filtered = x_test[test_mask]
        y_test_filtered = y_test[test_mask]

        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        batch_size = 64  # Adjust the batch size as needed
        test_fitlered_loss, test_filtered_accuracy = self.model.evaluate(x_test_filtered, y_test_filtered, batch_size=batch_size,verbose=0)
        print(f'selected Labels accuracy: {test_filtered_accuracy:.4f}')
        print(f'selected Labels loss: {test_fitlered_loss:.4f}')

        return test_filtered_accuracy, test_fitlered_loss


    def model_f1score(self, x_test, y_test):
        # works for cifar100 dataset
        if self.f1score != -1:
            return self.f1score
        
        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)

        y_test_pred = self.model.predict(x_test)

        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_test_true_classes = np.argmax(y_test, axis=1)

        self.f1score = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        print(f'F1 Score (full dataset): {self.f1score:.4f}')
    
    def model_f1score_selected(self, dataset_instance, selected_labels): 
        # works for cifar100 dataset
        x_test = dataset_instance.x_test
        y_test = dataset_instance.y_test

        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)

        # validation for specific labels accuracy using mask
        selected_indices = [selected_labels.index(label) for label in selected_labels]

        # Create masks for train and test sets
        test_mask = np.isin(np.argmax(y_test, axis=1), selected_indices)

        # Filter x_test, and y_test
        x_test_filtered = x_test[test_mask]
        y_test_filtered = y_test[test_mask]

        y_test_filtered_pred = self.model.predict(x_test_filtered)
        y_test_filtered_pred_classes = np.argmax(y_test_filtered_pred, axis=1)
        y_test_filtered_true_classes = np.argmax(y_test_filtered, axis=1)

        f1_filtered = f1_score(y_test_filtered_true_classes, y_test_filtered_pred_classes, average='weighted')
        print(f'F1 Score (selected labels): {f1_filtered:.4f}')
        return f1_filtered

    def predict(self, image_input):
        if "cifar100" in self.dataset:
            return self.predict_cifar100(image_input)
        elif "imagenet" in self.dataset:
            return self.predict_imagenet(image_input)
        else:
            print("Model not recognized")


    def predict_cifar100(self, image_input):
        expected_size = (32, 32)
        # Assume it's a path and load the image
        img = tf.keras.preprocessing.image.load_img(image_input, target_size=expected_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = self.model.predict(img_array)

        # Get the indices of the top k predictions
        top_indices = np.argsort(predictions[0])[-self.top_k:][::-1]
        
        # Get the probabilities for those indices
        top_probabilities = predictions[0][top_indices]
        
        # Combine indices and probabilities
        results = [(idx, prob) for idx, prob in zip(top_indices, top_probabilities)]
        
        return results

    
    def predict_imagenet(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=self.top_k)[0]

        return decoded_predictions

    # def predict_batches(self, image_paths): 
    #     if "cifar100" in self.dataset:
    #         return self.predict_batches_cifar100(image_paths)
    #     elif "imagenet" in self.dataset:
    #         return self.predict_batches_imagenet(image_paths)
    #     else:
    #         print("Model not recognized")

    # def predict_batches_imagenet(self, image_paths):
    #     batch_predictor = ImageNetBatchPredictor(self.model, batch_size=32, num_workers=2)
    #     return batch_predictor.get_top_predictions(image_paths, self.top_k)

    # def predict_batches_cifar100(self, image_paths): 
    #     batched_predictor = Cifar100BatchPredictor(self.model, batch_size=32)
    #     return batched_predictor.get_top_predictions(image_paths, self.top_k)

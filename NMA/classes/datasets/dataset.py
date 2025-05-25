from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, name, threshold, infinity, labels):
        self.name = name
        self.threshold = threshold
        self.infinity = infinity
        self.labels = labels

    @abstractmethod
    def load(self, name):
        pass

    @abstractmethod
    def get_train_image_by_id(self, image_id):
        pass

    @abstractmethod
    def get_test_image_by_id(self, image_id):
        pass

    @abstractmethod
    def get_label_readable_name(self, label):
        pass
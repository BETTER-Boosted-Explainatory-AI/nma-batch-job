from .cifar100 import Cifar100
from .imagenet import ImageNet


class DatasetFactory:
    _datasets = {"cifar100": Cifar100, "imagenet": ImageNet}

    @classmethod
    def register_dataset(cls, name, dataset_class):
        cls._datasets[name] = dataset_class

    @classmethod
    def create_dataset(cls, dataset_type):
        if dataset_type not in cls._datasets:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. Available types: {list(cls._datasets.keys())}"
            )

        dataset = cls._datasets[dataset_type]()
        return dataset

    @classmethod
    def get_available_datasets(cls):
        return list(cls._datasets.keys())

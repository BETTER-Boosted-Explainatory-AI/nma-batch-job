# Import the config function
from NMA.services.dataset_service import _get_dataset_config
from NMA.classes.datasets.cifar100 import Cifar100
from NMA.classes.datasets.imagenet import ImageNet

print("="*60)
print("DATASET CONFIGURATION INFORMATION")
print("="*60)

# Test CIFAR-100
print("\n### CIFAR-100 ###")
print("-"*40)

# Get the full config
cifar_config = _get_dataset_config("cifar100")
print("\nFull CIFAR-100 Config from _get_dataset_config():")
for key, value in cifar_config.items():
    if isinstance(value, list) and len(value) > 5:
        print(f"  {key}: [list with {len(value)} items, first 5: {value[:5]}...]")
    else:
        print(f"  {key}: {value}")

# Create dataset instance and check attributes
print("\nCIFAR-100 Dataset Instance Attributes:")
cifar_dataset = Cifar100()
dataset_attrs = ['threshold', 'infinity', 'dataset', 'labels']
for attr in dataset_attrs:
    if hasattr(cifar_dataset, attr):
        val = getattr(cifar_dataset, attr)
        if isinstance(val, list) and len(val) > 5:
            print(f"  {attr}: [list with {len(val)} items, first 5: {val[:5]}...]")
        else:
            print(f"  {attr}: {val}")

# Test ImageNet
print("\n\n### ImageNet ###")
print("-"*40)

# Get the full config
imagenet_config = _get_dataset_config("imagenet")
print("\nFull ImageNet Config from _get_dataset_config():")
for key, value in imagenet_config.items():
    if isinstance(value, (list, dict)) and len(value) > 5:
        print(f"  {key}: [{type(value).__name__} with {len(value)} items]")
        if isinstance(value, list):
            print(f"    First 5 items: {value[:5]}...")
        elif isinstance(value, dict):
            print(f"    First 5 keys: {list(value.keys())[:5]}...")
    else:
        print(f"  {key}: {value}")

# Create dataset instance and check attributes
print("\nImageNet Dataset Instance Attributes:")
imagenet_dataset = ImageNet()
dataset_attrs = ['threshold', 'infinity', 'dataset', 'directory_labels']
for attr in dataset_attrs:
    if hasattr(imagenet_dataset, attr):
        val = getattr(imagenet_dataset, attr)
        if isinstance(val, (list, dict)) and len(val) > 5:
            print(f"  {attr}: [{type(val).__name__} with {len(val)} items]")
            if isinstance(val, list):
                print(f"    First 5 items: {val[:5]}...")
        else:
            print(f"  {attr}: {val}")

# Additional debug info
print("\n\n### Additional Debug Info ###")
print("-"*40)

# Check if there's a directory_to_readable mapping for ImageNet
if 'directory_to_readable' in imagenet_config:
    print("\nImageNet has directory_to_readable mapping")
    print(f"  Number of mappings: {len(imagenet_config['directory_to_readable'])}")
    # Show a few examples
    items = list(imagenet_config['directory_to_readable'].items())[:3]
    print("  Examples:")
    for k, v in items:
        print(f"    {k} -> {v}")

# Show all available keys in configs
print("\n\nAll Config Keys:")
print(f"  CIFAR-100 config keys: {list(cifar_config.keys())}")
print(f"  ImageNet config keys: {list(imagenet_config.keys())}")
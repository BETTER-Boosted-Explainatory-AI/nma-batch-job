import dotenv
dotenv.load_dotenv()

import os
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from NMA.utilss.s3_utils import get_datasets_s3_client, get_users_s3_client

import pickle
import io
import sys
from types import ModuleType

class S3Handler:
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, bucket_name=None):
        self.aws_access_key_id = aws_access_key_id or os.environ.get('AWS_DATASETS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_DATASETS_SECRET_ACCESS_KEY')
        self.bucket_name = bucket_name or os.environ.get('S3_DATASETS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name not specified")
        try:
            if self.aws_access_key_id and self.aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key
                )
            else:
                self.s3_client = boto3.client('s3')
        except Exception as e:
            print(f"Error creating S3 client with boto3: {str(e)}")
            try:
                self.s3_client = get_datasets_s3_client()
            except Exception as e2:
                print(f"Error creating S3 client with get_datasets_s3_client: {str(e2)}")
                raise ValueError(f"Failed to create S3 client: {str(e2)}")
             
    def list_objects(self, prefix=''):
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend([obj['Key'] for obj in page['Contents']])
            
            return objects
        except (NoCredentialsError, ClientError) as e:
            print(f"Error listing S3 objects: {str(e)}")
            return []
    list_images = list_objects
    
    def get_folder_contents(self, dataset_name, folder_type):
        prefix = f"{dataset_name}/{folder_type}/"
        return self.list_objects(prefix)
    
    def get_object_data(self, key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except (NoCredentialsError, ClientError) as e:
            print(f"Error getting object from S3: {str(e)}")
            return None

    get_single_image = get_object_data
    
    def get_object_stream(self, key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body']
        except (NoCredentialsError, ClientError) as e:
            print(f"Error getting object stream from S3: {str(e)}")
            return None
    
    get_image_stream = get_object_stream
    
    def get_cifar100_as_numpy(self, folder_type):
        prefix = f"cifar100/{folder_type}/"
        s3_files = self.list_objects(prefix)
        bin_files = [f for f in s3_files if f.endswith('.bin') or f.endswith('.pkl')]
        
        if bin_files:
            images_list = []
            labels_list = []
            
            for s3_key in bin_files:
                try:
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                    file_content = response['Body'].read()
                    data = pickle.loads(file_content, encoding='bytes')
                    
                    if b'data' in data:
                        batch_images = data[b'data']
                        batch_labels = data[b'fine_labels'] if b'fine_labels' in data else data[b'labels']
                        
                        images_list.append(batch_images)
                        labels_list.append(batch_labels)
                except Exception as e:
                    print(f"Error processing CIFAR file {s3_key}: {str(e)}")
                    continue
            
            if images_list:
                images = np.vstack(images_list)
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                labels = np.concatenate(labels_list)
                return images, labels
        
        npy_files = [f for f in s3_files if f.endswith('.npy')]
        
        if npy_files:
            try:
                npy_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                images = []
                
                for s3_key in npy_files:
                    try:
                        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                        file_content = response['Body'].read()
                        array_data = np.load(io.BytesIO(file_content), allow_pickle=True)
                        images.append(array_data)
                    except Exception as e:
                        print(f"Error loading numpy file {s3_key}: {str(e)}")
                        continue
                
                if images:
                    images_array = np.stack(images)
                    labels = np.arange(len(images_array))
                    return images_array, labels
                
            except Exception as e:
                print(f"Error processing .npy files: {str(e)}")
        return np.array([]), np.array([])
    
    def get_cifar100_meta(self):
        meta_key = "cifar100/meta"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=meta_key)
            file_content = response['Body'].read()
            meta_data = pickle.loads(file_content, encoding='bytes')
            return meta_data
        except Exception as e:
            print(f"Error getting CIFAR-100 meta: {str(e)}")
            return None
    
    def get_imagenet_class_data(self, class_dir):
        if not class_dir.startswith('n') or '.txt' in class_dir:
            print(f"Skipping non-class directory: {class_dir}")
            return []
        prefix = f"imagenet/train/{class_dir}/"
        return self.list_objects(prefix)
    
    def get_imagenet_classes(self):
        prefix = "imagenet/train/"
        all_files = self.list_objects(prefix)
        class_dirs = set()
        for file_path in all_files:
            parts = file_path.replace(prefix, '').split('/')
            if parts and parts[0]:
                if parts[0].startswith('n') and len(parts[0]) > 1 and not parts[0].endswith('.txt'):
                    class_dirs.add(parts[0])
        return class_dirs
    
    def load_python_module_from_s3(self, s3_key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            module_content = response['Body'].read()
            module_name = os.path.splitext(os.path.basename(s3_key))[0]
            module = ModuleType(module_name)
            module.__file__ = s3_key
            exec(module_content, module.__dict__)
            return module
        
        except Exception as e:
            print(f"Error loading Python module from S3: {str(e)}")
            raise
    
    def get_numpy_data(self, dataset_name, folder_type):
        if folder_type not in ['clean', 'adversarial']:
            raise ValueError(f"Invalid folder type for numpy data: {folder_type}")
        prefix = f"{dataset_name}/{folder_type}/"
        s3_files = self.list_objects(prefix)
        npy_files = [f for f in s3_files if f.endswith('.npy')]
        if not npy_files:
            print(f"Warning: No .npy files found in {prefix}")
            return {}
        numpy_data = {}
        for s3_key in npy_files:
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                file_content = response['Body'].read()
                array_data = np.load(io.BytesIO(file_content), allow_pickle=True)
                file_name = os.path.basename(s3_key)
                numpy_data[file_name] = array_data
                
            except Exception as e:
                print(f"Error getting numpy data from S3 for {s3_key}: {str(e)}")
                continue
        return numpy_data
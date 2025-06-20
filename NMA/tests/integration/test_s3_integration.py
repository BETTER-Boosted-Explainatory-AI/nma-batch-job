import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import os
import pytest
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
import tempfile
import shutil
import numpy as np
from datetime import datetime
import time
import uuid
import tensorflow as tf
from pathlib import Path

# Import your actual modules
from NMA.services.nma_service import _create_nma
from NMA.services.model_service import (
    _get_model_path, _get_model_filename, _load_model,
    s3_file_exists, read_json_from_s3, load_model_from_s3,
    get_user_models_info, get_model_files
)
from NMA.services.dataset_service import (
    _get_dataset_config, _load_dataset, get_dataset_labels,
    _load_dataset_folder, load_single_image
)
from NMA.utilss.s3_utils import get_datasets_s3_client, get_users_s3_client
from NMA.s3_connector.s3_dataset_loader import S3DatasetLoader
from NMA.utilss.files_utils import update_current_model
from NMA.classes.datasets.cifar100 import Cifar100
from NMA.classes.datasets.imagenet import ImageNet

# Test configuration - should be in .env.integration or similar
TEST_USER_ID = os.getenv('TEST_USER_ID', f'integration_test_{uuid.uuid4().hex[:8]}')
TEST_MODEL_ID = os.getenv('TEST_MODEL_ID', f'test_model_{uuid.uuid4().hex[:8]}')
TEST_CLEANUP = os.getenv('TEST_CLEANUP', 'true').lower() == 'true'

@pytest.mark.parametrize("dataset_object, expected_name", [
    (Cifar100(), "cifar100"),
    (ImageNet(), "imagenet")
])

def test_model_metadata_dataset_names(ensure_real_credentials, test_data_cleanup, dataset_object, expected_name):
    user_id = TEST_USER_ID
    model_id = f"{TEST_MODEL_ID}_{expected_name}"
    filename = f"{expected_name}_test_model.keras"

    update_current_model(
        user_id=user_id,
        model_id=model_id,
        graph_type="similarity",
        model_filename=filename,
        dataset=dataset_object,
        min_confidence=0.9,
        top_k=5
    )

    key = f"{user_id}/models.json"
    test_data_cleanup.append(key)

    s3 = ensure_real_credentials['users_client']
    bucket = ensure_real_credentials['users_bucket']

    assert s3_file_exists(bucket, key)
    models_data = read_json_from_s3(bucket, key)

    test_model = next((m for m in models_data if m['model_id'] == model_id), None)
    assert test_model is not None
    assert test_model['dataset'] == expected_name

@pytest.fixture(scope='session')
def ensure_real_credentials():
    """Ensure we have real AWS credentials before running integration tests"""
    try:
        # Try to create clients
        datasets_client = get_datasets_s3_client()
        users_client = get_users_s3_client()
        
        # Verify credentials by listing buckets
        datasets_client.list_buckets()
        users_client.list_buckets()
        
        # Verify bucket access
        datasets_bucket = os.getenv('S3_DATASETS_BUCKET_NAME')
        users_bucket = os.getenv('S3_USERS_BUCKET_NAME')
        
        datasets_client.head_bucket(Bucket=datasets_bucket)
        users_client.head_bucket(Bucket=users_bucket)
        
        return {
            'datasets_client': datasets_client,
            'users_client': users_client,
            'datasets_bucket': datasets_bucket,
            'users_bucket': users_bucket
        }
    except (NoCredentialsError, ClientError) as e:
        pytest.skip(f"AWS credentials not configured properly: {e}")


@pytest.fixture(scope='session')
def test_data_cleanup(ensure_real_credentials):
    """Fixture to clean up test data after tests"""
    created_keys = []
    
    yield created_keys
    
    # Cleanup after all tests
    if TEST_CLEANUP:
        users_client = ensure_real_credentials['users_client']
        users_bucket = ensure_real_credentials['users_bucket']
        
        for key in created_keys:
            try:
                users_client.delete_object(Bucket=users_bucket, Key=key)
                print(f"Cleaned up: {key}")
            except ClientError:
                pass


class TestS3BucketConnectivity:
    """Test actual connectivity to S3 buckets"""
    
    def test_datasets_bucket_exists(self, ensure_real_credentials):
        """Test that datasets bucket exists and is accessible"""
        client = ensure_real_credentials['datasets_client']
        bucket = ensure_real_credentials['datasets_bucket']
        
        response = client.head_bucket(Bucket=bucket)
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
    
    def test_users_bucket_exists(self, ensure_real_credentials):
        """Test that users bucket exists and is accessible"""
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        response = client.head_bucket(Bucket=bucket)
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
    
    def test_list_datasets_bucket_contents(self, ensure_real_credentials):
        """Test listing contents of datasets bucket"""
        client = ensure_real_credentials['datasets_client']
        bucket = ensure_real_credentials['datasets_bucket']
        
        response = client.list_objects_v2(Bucket=bucket, MaxKeys=10)
        
        # Should have some contents
        assert 'Contents' in response or response.get('KeyCount', 0) == 0
        
        if 'Contents' in response:
            for obj in response['Contents']:
                assert 'Key' in obj
                assert 'Size' in obj
                assert 'LastModified' in obj
    
    def test_datasets_bucket_permissions(self, ensure_real_credentials):
        """Test read permissions on datasets bucket"""
        client = ensure_real_credentials['datasets_client']
        bucket = ensure_real_credentials['datasets_bucket']
        
        # Try to list specific dataset folders
        for prefix in ['cifar100/', 'imagenet/']:
            response = client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=5
            )
            # Just verify no errors - might be empty
            assert 'ResponseMetadata' in response


class TestS3FileOperations:
    """Test real S3 file operations"""
    
    def test_write_read_delete_cycle(self, ensure_real_credentials, test_data_cleanup):
        """Test complete file lifecycle in S3"""
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        # Create test data
        test_key = f"{TEST_USER_ID}/test_file_{uuid.uuid4().hex}.txt"
        test_content = f"Integration test content - {datetime.now()}"
        
        # Write to S3
        client.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=test_content.encode('utf-8')
        )
        test_data_cleanup.append(test_key)
        
        # Verify file exists
        assert s3_file_exists(bucket, test_key) is True
        
        # Read from S3
        response = client.get_object(Bucket=bucket, Key=test_key)
        read_content = response['Body'].read().decode('utf-8')
        assert read_content == test_content
        
        # Delete from S3
        client.delete_object(Bucket=bucket, Key=test_key)
        
        # Verify file no longer exists
        assert s3_file_exists(bucket, test_key) is False
    
    def test_json_operations(self, ensure_real_credentials, test_data_cleanup):
        """Test JSON read/write operations"""
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        test_key = f"{TEST_USER_ID}/test_data_{uuid.uuid4().hex}.json"
        test_data = {
            'test_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'values': [1, 2, 3, 4, 5],
            'nested': {
                'key1': 'value1',
                'key2': 42
            }
        }
        
        # Write JSON
        client.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=json.dumps(test_data).encode('utf-8'),
            ContentType='application/json'
        )
        test_data_cleanup.append(test_key)
        
        # Read JSON using your function
        read_data = read_json_from_s3(bucket, test_key)
        
        assert read_data == test_data
        assert read_data['test_id'] == test_data['test_id']
        assert read_data['values'] == test_data['values']


class TestModelOperations:
    """Test real model operations with S3"""
    
    def test_model_path_operations(self, ensure_real_credentials, test_data_cleanup):
        """Test model path creation and retrieval"""
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        # Create a dummy model file
        model_key = f"{TEST_USER_ID}/{TEST_MODEL_ID}/test_model.keras"
        client.put_object(
            Bucket=bucket,
            Key=model_key,
            Body=b"dummy model content"
        )
        test_data_cleanup.append(model_key)
        
        # Test get_model_path
        model_path = _get_model_path(TEST_USER_ID, TEST_MODEL_ID)
        assert model_path == f"{TEST_USER_ID}/{TEST_MODEL_ID}"
        
        # Test get_model_filename
        filename = _get_model_filename(TEST_USER_ID, TEST_MODEL_ID, 'similarity')
        assert filename == model_key
    
    def test_model_metadata_operations(self, ensure_real_credentials, test_data_cleanup):
        """Test model metadata save and retrieve"""
        # Test update_current_model
        update_current_model(
            user_id=TEST_USER_ID,
            model_id=TEST_MODEL_ID,
            graph_type='similarity',
            model_filename='test_model.keras',
            dataset='cifar100',
            min_confidence=0.8,
            top_k=5
        )
        
        models_json_key = f"{TEST_USER_ID}/models.json"
        test_data_cleanup.append(models_json_key)
        
        # Verify models.json was created
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        assert s3_file_exists(bucket, models_json_key)
        
        # Read and verify content
        models_data = read_json_from_s3(bucket, models_json_key)
        assert isinstance(models_data, list)
        assert len(models_data) > 0
        
        # Find our test model
        test_model = next((m for m in models_data if m['model_id'] == TEST_MODEL_ID), None)
        assert test_model is not None
        assert test_model['dataset'] == 'cifar100'
        assert test_model['graph_type'] == 'similarity'
        assert test_model['min_confidence'] == 0.8
        assert test_model['top_k'] == 5


class TestDatasetOperations:
    """Test real dataset operations"""
    
    def test_dataset_loader_initialization(self, ensure_real_credentials):
        """Test S3DatasetLoader with real S3"""
        loader = S3DatasetLoader()
        
        assert loader.bucket_name == os.getenv('S3_DATASETS_BUCKET_NAME')
        assert loader.s3_handler is not None
    
    def test_get_dataset_config_real(self, ensure_real_credentials):
        """Test getting real dataset configuration from S3"""
        # Test CIFAR100 config
        cifar_config = _get_dataset_config('cifar100')
        
        assert cifar_config is not None
        assert cifar_config['dataset'] == 'cifar100'
        assert 'labels' in cifar_config
        assert 'top_k' in cifar_config
        assert 'min_confidence' in cifar_config
        
        # Verify labels structure
        assert isinstance(cifar_config['labels'], list)
        assert len(cifar_config['labels']) == 100  # CIFAR100 has 100 classes
    
    @pytest.mark.slow
    def test_load_dataset_folder_structure(self, ensure_real_credentials):
        """Test loading actual dataset folder structure"""
        # Test CIFAR100 test folder
        files = _load_dataset_folder('cifar100', 'test')
        
        assert files is not None
        assert isinstance(files, list)
        
        # Should have some test files
        if len(files) > 0:
            # Check file format
            assert any('.pkl' in f or '.npy' in f for f in files[:10])
    
    def test_dataset_labels_retrieval(self, ensure_real_credentials):
        """Test getting dataset labels"""
        # Test CIFAR100 labels
        cifar_labels = get_dataset_labels('cifar100')
        
        assert cifar_labels is not None
        assert len(cifar_labels) == 100
        assert isinstance(cifar_labels[0], str)
        
        # Check some known CIFAR100 labels
        expected_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver']
        for label in expected_labels:
            assert label in cifar_labels


class TestEndToEndWorkflow:
    """Test complete workflows using real S3"""
    
    @pytest.mark.slow
    def test_model_upload_and_retrieve_workflow(self, ensure_real_credentials, test_data_cleanup):
        """Test complete model upload and retrieve workflow"""
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        # Create a simple keras model
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, input_shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            model_path = os.path.join(tmpdir, 'test_model.keras')
            model.save(model_path)
            
            # Upload to S3
            model_key = f"{TEST_USER_ID}/{TEST_MODEL_ID}/integration_test_model.keras"
            with open(model_path, 'rb') as f:
                client.put_object(
                    Bucket=bucket,
                    Key=model_key,
                    Body=f.read()
                )
            test_data_cleanup.append(model_key)
            
            # Update metadata
            update_current_model(
                user_id=TEST_USER_ID,
                model_id=TEST_MODEL_ID,
                graph_type='similarity',
                model_filename='test_model.keras',
                dataset=Cifar100(),  # ← OBJECT with .dataset = "cifar100"
                min_confidence=0.8,
                top_k=5
            )
            test_data_cleanup.append(f"{TEST_USER_ID}/models.json")
            
            # Retrieve and verify
            retrieved_path = _get_model_path(TEST_USER_ID, TEST_MODEL_ID)
            assert retrieved_path is not None
            
            # Load model from S3
            loaded_model = load_model_from_s3(bucket, model_key)
            assert loaded_model is not None
            assert len(loaded_model.layers) == 3
    
    @pytest.mark.slow
    def test_s3_performance_metrics(self, ensure_real_credentials, test_data_cleanup):
        """Test S3 operation performance with real calls"""
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        # Test write performance
        write_times = []
        test_data = b"x" * 1024  # 1KB of data
        
        for i in range(10):
            key = f"{TEST_USER_ID}/perf_test_{i}.dat"
            start = time.time()
            client.put_object(Bucket=bucket, Key=key, Body=test_data)
            write_times.append(time.time() - start)
            test_data_cleanup.append(key)
        
        avg_write_time = sum(write_times) / len(write_times)
        print(f"\nAverage write time: {avg_write_time:.3f}s")
        
        # Test read performance
        read_times = []
        for i in range(10):
            key = f"{TEST_USER_ID}/perf_test_{i}.dat"
            start = time.time()
            client.get_object(Bucket=bucket, Key=key)
            read_times.append(time.time() - start)
        
        avg_read_time = sum(read_times) / len(read_times)
        print(f"Average read time: {avg_read_time:.3f}s")
        
        # Performance assertions
        assert avg_write_time < 2.0, f"Write operations too slow: {avg_write_time}s"
        assert avg_read_time < 1.0, f"Read operations too slow: {avg_read_time}s"


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling with real S3 errors"""
    
    def test_nonexistent_file_handling(self, ensure_real_credentials):
        """Test handling of nonexistent files"""
        bucket = ensure_real_credentials['users_bucket']
        
        # Test file_exists with nonexistent file
        assert s3_file_exists(bucket, 'nonexistent/file.txt') is False
        
        # Test read_json_from_s3 with nonexistent file
        with pytest.raises(ClientError) as exc_info:
            read_json_from_s3(bucket, 'nonexistent/file.json')
        assert exc_info.value.response['Error']['Code'] == 'NoSuchKey'
    
    def test_permission_errors(self, ensure_real_credentials):
        """Test handling of permission errors"""
        # Try to access a bucket we don't have permissions for
        client = ensure_real_credentials['users_client']
        
        with pytest.raises(ClientError) as exc_info:
            client.head_bucket(Bucket='some-random-bucket-we-dont-own-12345')
        
        error_code = exc_info.value.response['Error']['Code']
        assert error_code in ['403', 'AccessDenied', 'NoSuchBucket']


# Helper function to run integration tests
def run_integration_tests():
    """Run only integration tests"""
    return pytest.main([
        __file__,
        '-v',
        '-m', 'integration or slow',
        '--tb=short'
    ])


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING S3 INTEGRATION TESTS")
    print("These tests connect to real S3 buckets")
    print("="*60 + "\n")
    
    run_integration_tests()
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
import json
import tempfile
from datetime import datetime
import time
import uuid
import tensorflow as tf
import boto3
from dotenv import load_dotenv
from pathlib import Path
test_env_path = Path(__file__).resolve().parent / ".env.test"
load_dotenv(test_env_path, override=True)
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


TEST_USER_ID = os.getenv('user_TEST_id')
TEST_MODEL_ID = os.getenv('model_TEST_id')
TEST_GRAPH_TYPE = os.getenv('graph_TEST_type')
TEST_DATASET = os.getenv('dataset_TEST_')
TEST_MIN_CONFIDENCE = float(os.getenv('min_confidence_TEST_'))
TEST_TOP_K = int(os.getenv('top_k_TEST_'))
TEST_MODEL_FILE = os.getenv('model_file_TEST_')
TEST_CLEANUP = os.getenv('TEST_CLEANUP').lower() == 'true'


@pytest.fixture(scope='session')
def ensure_real_credentials():
    original_env = {}
    test_mappings = {
        'S3_DATASETS_BUCKET_NAME': os.getenv('S3_TEST_DATASETS_BUCKET_NAME'),
        'S3_USERS_BUCKET_NAME': os.getenv('S3_TEST_USERS_BUCKET_NAME'),
        'AWS_DATASETS_ACCESS_KEY_ID': os.getenv('AWS_TEST_DATASETS_ACCESS_KEY_ID'),
        'AWS_DATASETS_SECRET_ACCESS_KEY': os.getenv('AWS_TEST_DATASETS_SECRET_ACCESS_KEY'),
        'AWS_USERS_ACCESS_KEY_ID': os.getenv('AWS_TEST_USERS_ACCESS_KEY_ID'),
        'AWS_USERS_SECRET_ACCESS_KEY': os.getenv('AWS_TEST_USERS_SECRET_ACCESS_KEY'),
    }
    
    for key, test_value in test_mappings.items():
        if test_value:
            original_env[key] = os.getenv(key)
            os.environ[key] = test_value
    
    try:
        datasets_client = get_datasets_s3_client()
        users_client = get_users_s3_client()
        datasets_client.list_buckets()
        users_client.list_buckets()
        datasets_bucket = os.getenv('S3_TEST_DATASETS_BUCKET_NAME')
        users_bucket = os.getenv('S3_TEST_USERS_BUCKET_NAME')
        datasets_client.head_bucket(Bucket=datasets_bucket)
        users_client.head_bucket(Bucket=users_bucket)
        
        yield {
            'datasets_client': datasets_client,
            'users_client': users_client,
            'datasets_bucket': datasets_bucket,
            'users_bucket': users_bucket,
            'original_env': original_env
        }
        
    except (NoCredentialsError, ClientError) as e:
        import traceback, sys
        print("\n=== DATASET-CLIENT EXCEPTION ===", file=sys.stderr)
        traceback.print_exc()
        print("================================\n", file=sys.stderr)
        pytest.skip(f"AWS test credentials not configured properly: {e}")
    finally:
        for key, original_value in original_env.items():
            if original_value is not None:
                os.environ[key] = original_value
            else:
                os.environ.pop(key, None)


@pytest.fixture(scope='session')
def test_data_cleanup(ensure_real_credentials):
    created_keys = []
    yield created_keys
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
    def test_datasets_bucket_exists(self, ensure_real_credentials):
        client = ensure_real_credentials['datasets_client']
        bucket = ensure_real_credentials['datasets_bucket']
        response = client.head_bucket(Bucket=bucket)
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        print(f"✓ Successfully connected to test datasets bucket: {bucket}")
    
    def test_users_bucket_exists(self, ensure_real_credentials):
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        response = client.head_bucket(Bucket=bucket)
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        print(f"✓ Successfully connected to test users bucket: {bucket}")
    
    def test_list_datasets_bucket_contents(self, ensure_real_credentials):
        client = ensure_real_credentials['datasets_client']
        bucket = ensure_real_credentials['datasets_bucket']
        response = client.list_objects_v2(Bucket=bucket, MaxKeys=10)
        
        assert 'Contents' in response or response.get('KeyCount', 0) == 0
        
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects in test datasets bucket")
            for obj in response['Contents']:
                assert 'Key' in obj
                assert 'Size' in obj
                assert 'LastModified' in obj
    
    def test_datasets_bucket_permissions(self, ensure_real_credentials):
        client = ensure_real_credentials['datasets_client']
        bucket = ensure_real_credentials['datasets_bucket']
        
        for prefix in ['cifar100/', 'imagenet/']:
            response = client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=5
            )
            assert 'ResponseMetadata' in response


class TestS3FileOperations:
    def test_write_read_delete_cycle(self, ensure_real_credentials, test_data_cleanup):
        """Test complete file lifecycle in test S3 bucket"""
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        test_key = f"{TEST_USER_ID}/test_file_{uuid.uuid4().hex}.txt"
        test_content = f"Integration test content - {datetime.now()}"
        
        client.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=test_content.encode('utf-8')
        )
        
        test_data_cleanup.append(test_key)
        assert s3_file_exists(bucket, test_key) is True
        response = client.get_object(Bucket=bucket, Key=test_key)
        read_content = response['Body'].read().decode('utf-8')
        assert read_content == test_content
        client.delete_object(Bucket=bucket, Key=test_key)
        assert s3_file_exists(bucket, test_key) is False
    
    def test_json_operations(self, ensure_real_credentials, test_data_cleanup):
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
        
        client.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=json.dumps(test_data).encode('utf-8'),
            ContentType='application/json'
        )
        test_data_cleanup.append(test_key)
        read_data = read_json_from_s3(bucket, test_key)
        
        assert read_data == test_data
        assert read_data['test_id'] == test_data['test_id']
        assert read_data['values'] == test_data['values']


class TestModelOperations:
    def test_model_path_operations(self, ensure_real_credentials, test_data_cleanup):
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        # Create a dummy model file
        model_key = f"{TEST_USER_ID}/{TEST_MODEL_ID}/{TEST_MODEL_FILE}"
        client.put_object(
            Bucket=bucket,
            Key=model_key,
            Body=b"dummy model content for testing"
        )
        test_data_cleanup.append(model_key)
        
        # Test get_model_path
        model_path = _get_model_path(TEST_USER_ID, TEST_MODEL_ID)
        assert model_path == f"{TEST_USER_ID}/{TEST_MODEL_ID}"
        
        # Test get_model_filename
        filename = _get_model_filename(TEST_USER_ID, TEST_MODEL_ID, TEST_GRAPH_TYPE)
        assert filename == model_key
    
    def test_model_metadata_operations(self, ensure_real_credentials, test_data_cleanup):
        dataset_obj = Cifar100() if TEST_DATASET == "cifar100" else ImageNet()
        update_current_model(
            user_id=TEST_USER_ID,
            model_id=TEST_MODEL_ID,
            graph_type=TEST_GRAPH_TYPE,
            model_filename=TEST_MODEL_FILE,
            dataset=dataset_obj, 
            min_confidence=TEST_MIN_CONFIDENCE,
            top_k=TEST_TOP_K
        )
        
        models_json_key = f"{TEST_USER_ID}/models.json"
        test_data_cleanup.append(models_json_key)
        
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        assert s3_file_exists(bucket, models_json_key)
        models_data = read_json_from_s3(bucket, models_json_key)
        assert isinstance(models_data, list)
        assert len(models_data) > 0
        
        test_model = next((m for m in models_data if m['model_id'] == TEST_MODEL_ID), None)
        assert test_model is not None
        assert test_model["dataset"] == TEST_DATASET
        assert test_model['graph_type'] == TEST_GRAPH_TYPE
        assert test_model['min_confidence'] == TEST_MIN_CONFIDENCE
        assert test_model['top_k'] == TEST_TOP_K


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


class TestDatasetOperations:
    def test_dataset_loader_initialization(self, ensure_real_credentials):
        loader = S3DatasetLoader()
        
        assert loader.bucket_name == os.getenv('S3_DATASETS_BUCKET_NAME')  # This is now the test bucket
        assert loader.s3_handler is not None
    
    def test_get_dataset_config_real(self, ensure_real_credentials):
        cifar_config = _get_dataset_config(TEST_DATASET)
        
        assert cifar_config is not None
        assert cifar_config['dataset'] == TEST_DATASET
        assert 'labels' in cifar_config
        assert 'top_k' in cifar_config
        assert 'min_confidence' in cifar_config
        
        assert isinstance(cifar_config['labels'], list)
        if TEST_DATASET == 'cifar100':
            assert len(cifar_config['labels']) == 100  # CIFAR100 has 100 classes
    
    @pytest.mark.slow
    def test_load_dataset_folder_structure(self, ensure_real_credentials):
        """Test loading actual dataset folder structure from test bucket"""
        files = _load_dataset_folder(TEST_DATASET, 'test')
        
        assert files is not None
        assert isinstance(files, list)
        
        if len(files) > 0:
            assert any('.pkl' in f or '.npy' in f for f in files[:10])
    
    def test_dataset_labels_retrieval(self, ensure_real_credentials):
        labels = get_dataset_labels(TEST_DATASET)
        
        assert labels is not None
        if TEST_DATASET == 'cifar100':
            assert len(labels) == 100
            assert isinstance(labels[0], str)
            
            expected_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver']
            for label in expected_labels:
                assert label in labels


class TestEndToEndWorkflow:
    @pytest.mark.slow
    def test_model_upload_and_retrieve_workflow(self, ensure_real_credentials, test_data_cleanup):
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, input_shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            model_path = os.path.join(tmpdir, TEST_MODEL_FILE)
            model.save(model_path)
            
            model_key = f"{TEST_USER_ID}/{TEST_MODEL_ID}/{TEST_MODEL_FILE}"
            with open(model_path, 'rb') as f:
                client.put_object(
                    Bucket=bucket,
                    Key=model_key,
                    Body=f.read()
                )
            test_data_cleanup.append(model_key)
            
            dataset_obj = Cifar100() if TEST_DATASET == 'cifar100' else ImageNet()
            update_current_model(
                user_id=TEST_USER_ID,
                model_id=TEST_MODEL_ID,
                graph_type=TEST_GRAPH_TYPE,
                model_filename=TEST_MODEL_FILE,
                dataset=dataset_obj,
                min_confidence=TEST_MIN_CONFIDENCE,
                top_k=TEST_TOP_K
            )
            test_data_cleanup.append(f"{TEST_USER_ID}/models.json")
            
            retrieved_path = _get_model_path(TEST_USER_ID, TEST_MODEL_ID)
            assert retrieved_path is not None
            
            loaded_model = load_model_from_s3(bucket, model_key)
            assert loaded_model is not None
            assert len(loaded_model.layers) == 3
    
    @pytest.mark.slow
    def test_s3_performance_metrics(self, ensure_real_credentials, test_data_cleanup):
        client = ensure_real_credentials['users_client']
        bucket = ensure_real_credentials['users_bucket']
        
        write_times = []
        test_data = b"x" * 1024
        for i in range(10):
            key = f"{TEST_USER_ID}/perf_test_{i}.dat"
            start = time.time()
            client.put_object(Bucket=bucket, Key=key, Body=test_data)
            write_times.append(time.time() - start)
            test_data_cleanup.append(key)
        
        avg_write_time = sum(write_times) / len(write_times)
        print(f"\nAverage write time to test bucket: {avg_write_time:.3f}s")
        
        read_times = []
        for i in range(10):
            key = f"{TEST_USER_ID}/perf_test_{i}.dat"
            start = time.time()
            client.get_object(Bucket=bucket, Key=key)
            read_times.append(time.time() - start)
        
        avg_read_time = sum(read_times) / len(read_times)
        print(f"Average read time from test bucket: {avg_read_time:.3f}s")
        
        assert avg_write_time < 2.0, f"Write operations too slow: {avg_write_time}s"
        assert avg_read_time < 1.0, f"Read operations too slow: {avg_read_time}s"


@pytest.mark.integration
class TestErrorHandling:
    def test_nonexistent_file_handling(self, ensure_real_credentials):
        bucket = ensure_real_credentials['users_bucket']
        assert s3_file_exists(bucket, 'nonexistent/file.txt') is False
        
        with pytest.raises(ClientError) as exc_info:
            read_json_from_s3(bucket, 'nonexistent/file.json')
        assert exc_info.value.response['Error']['Code'] == 'NoSuchKey'
    
    def test_permission_errors(self, ensure_real_credentials):
        client = ensure_real_credentials['users_client']
        
        with pytest.raises(ClientError) as exc_info:
            client.head_bucket(Bucket='some-random-bucket-we-dont-own-12345')
        
        error_code = exc_info.value.response['Error']['Code']
        assert error_code in ['403', '404', 'AccessDenied', 'NoSuchBucket']


def run_integration_tests():
    return pytest.main([
        __file__,
        '-v',
        '-m', 'integration or slow',
        '--tb=short'
    ])


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING S3 INTEGRATION TESTS")
    print(f"Test User ID: {TEST_USER_ID}")
    print(f"Test Model ID: {TEST_MODEL_ID}")
    print(f"Test Dataset: {TEST_DATASET}")
    print(f"Test Buckets: ")
    print(f"  - Datasets: {os.getenv('S3_TEST_DATASETS_BUCKET_NAME')}")
    print(f"  - Users: {os.getenv('S3_TEST_USERS_BUCKET_NAME')}")
    print("="*60 + "\n")
    
    run_integration_tests()
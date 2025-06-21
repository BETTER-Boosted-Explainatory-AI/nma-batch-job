import os
import pytest
from unittest.mock import patch, MagicMock, Mock
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
import io
import numpy as np
import tempfile
from datetime import datetime

# Import your modules
from NMA.services.nma_service import _create_nma
from NMA.services.model_service import (
    _get_model_path, _get_model_filename, _load_model,
    s3_file_exists, read_json_from_s3, load_model_from_s3
)
from NMA.services.dataset_service import _get_dataset_config, _load_dataset
from NMA.utilss.s3_utils import get_datasets_s3_client, get_users_s3_client
from NMA.s3_connector.s3_dataset_loader import S3DatasetLoader


# Fixtures
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables"""
    env_vars = {
        'S3_DATASETS_BUCKET_NAME': 'test-datasets',
        'S3_USERS_BUCKET_NAME': 'test-users',
        'AWS_DATASETS_ACCESS_KEY_ID': 'test-key',
        'AWS_DATASETS_SECRET_ACCESS_KEY': 'test-secret',
        'AWS_USERS_ACCESS_KEY_ID': 'test-key',
        'AWS_USERS_SECRET_ACCESS_KEY': 'test-secret',
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client"""
    return MagicMock()


@pytest.fixture
def mock_dataset():
    """Create a mock dataset"""
    dataset = MagicMock()
    dataset.labels = ['label1', 'label2', 'label3']
    dataset.directory_labels = ['n01', 'n02', 'n03']
    return dataset


# Test S3 Client Creation
class TestS3ClientCreation:
    """Test S3 client creation for both buckets"""
    
    def test_datasets_client_creation(self, mock_env_vars):
        """Test datasets S3 client creation"""
        with patch('boto3.client') as mock_boto:
            mock_boto.return_value = MagicMock()
            client = get_datasets_s3_client()
            
            mock_boto.assert_called_with(
                's3',
                aws_access_key_id='test-key',
                aws_secret_access_key='test-secret'
            )
            assert client is not None
    
    def test_users_client_creation(self, mock_env_vars):
        """Test users S3 client creation"""
        with patch('boto3.client') as mock_boto:
            mock_boto.return_value = MagicMock()
            client = get_users_s3_client()
            
            mock_boto.assert_called_with(
                's3',
                aws_access_key_id='test-key',
                aws_secret_access_key='test-secret'
            )
            assert client is not None
    
    def test_client_error_handling(self):
        """Test client creation with missing credentials"""
        with patch('boto3.client') as mock_boto:
            mock_boto.side_effect = NoCredentialsError()
            
            with pytest.raises(NoCredentialsError):
                get_datasets_s3_client()


# Test S3 File Operations
class TestS3FileOperations:
    """Test S3 file operations"""
    
    @patch('NMA.utilss.s3_utils.get_users_s3_client')
    def test_file_exists_true(self, mock_get_client, mock_s3_client):
        """Test s3_file_exists when file exists"""
        mock_get_client.return_value = mock_s3_client
        mock_s3_client.head_object.return_value = {'ContentLength': 1024}
        
        result = s3_file_exists('test-bucket', 'test-key')
        assert result is True
        mock_s3_client.head_object.assert_called_with(
            Bucket='test-bucket',
            Key='test-key'
        )
    
    @patch('NMA.utilss.s3_utils.get_users_s3_client')
    def test_file_exists_false(self, mock_get_client, mock_s3_client):
        """Test s3_file_exists when file doesn't exist"""
        mock_get_client.return_value = mock_s3_client
        mock_s3_client.head_object.side_effect = ClientError(
            {'Error': {'Code': '404'}}, 'head_object'
        )
        
        result = s3_file_exists('test-bucket', 'nonexistent-key')
        assert result is False
    
    @patch('NMA.utilss.s3_utils.get_users_s3_client')
    def test_read_json_from_s3(self, mock_get_client, mock_s3_client):
        """Test reading JSON from S3"""
        mock_get_client.return_value = mock_s3_client
        
        test_data = {'key': 'value', 'items': [1, 2, 3]}
        mock_response = {
            'Body': io.BytesIO(json.dumps(test_data).encode('utf-8'))
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_json_from_s3('test-bucket', 'test.json')
        assert result == test_data
        mock_s3_client.get_object.assert_called_with(
            Bucket='test-bucket',
            Key='test.json'
        )


# Test Model Operations
class TestModelOperations:
    """Test model-related S3 operations"""
    
    @patch('NMA.utilss.s3_utils.get_users_s3_client')
    def test_get_model_path_success(self, mock_get_client, mock_s3_client):
        """Test successful model path retrieval"""
        mock_get_client.return_value = mock_s3_client
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [{'Key': 'user123/model456/model.keras'}]
        }
        
        path = _get_model_path('user123', 'model456')
        assert path == 'user123/model456'
    
    @patch('NMA.utilss.s3_utils.get_users_s3_client')
    def test_get_model_path_not_found(self, mock_get_client, mock_s3_client):
        """Test model path when not found"""
        mock_get_client.return_value = mock_s3_client
        mock_s3_client.list_objects_v2.return_value = {}
        
        path = _get_model_path('user123', 'nonexistent')
        assert path is None
    
    @patch('NMA.utilss.s3_utils.get_users_s3_client')
    @patch('tensorflow.keras.models.load_model')
    def test_load_model_from_s3(self, mock_keras_load, mock_get_client, mock_s3_client):
        """Test loading Keras model from S3"""
        mock_get_client.return_value = mock_s3_client
        mock_model = MagicMock()
        mock_keras_load.return_value = mock_model
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            tmp_path = tmp.name
        
        def mock_download(bucket, key, path):
            # Create empty file
            open(path, 'w').close()
        
        mock_s3_client.download_file.side_effect = mock_download
        
        try:
            model = load_model_from_s3('test-bucket', 'model.keras')
            assert model is not None
            mock_keras_load.assert_called()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# Test Dataset Operations
class TestDatasetOperations:
    """Test dataset-related S3 operations"""
    
    @patch('NMA.s3_connector.s3_handler.S3Handler')
    def test_dataset_loader_init(self, mock_handler_class, mock_env_vars):
        """Test S3DatasetLoader initialization"""
        loader = S3DatasetLoader(bucket_name='test-bucket')
        
        assert loader.bucket_name == 'test-bucket'
        assert hasattr(loader, 'cifar_loader')
        assert hasattr(loader, 'imagenet_loader')
    
    @patch('NMA.s3_connector.s3_handler.S3Handler')
    def test_get_dataset_info_cifar(self, mock_handler_class):
        """Test getting CIFAR100 dataset info"""
        mock_handler = MagicMock()
        mock_module = MagicMock()
        mock_module.CIFAR100_INFO = {
            'dataset': 'cifar100',
            'labels': ['airplane', 'automobile'],
            'top_k': 5,
            'min_confidence': 0.5
        }
        mock_handler.load_python_module_from_s3.return_value = mock_module
        
        loader = S3DatasetLoader(s3_handler=mock_handler, bucket_name='test')
        info = loader.get_dataset_info('cifar100')
        
        assert info['dataset'] == 'cifar100'
        assert 'labels' in info
        assert len(info['labels']) == 2
    
    @patch('NMA.s3_connector.s3_handler.S3Handler')
    def test_load_folder_imagenet(self, mock_handler_class):
        """Test loading ImageNet folder"""
        mock_handler = MagicMock()
        mock_handler.get_folder_contents.return_value = ['img1.JPEG', 'img2.JPEG']
        
        loader = S3DatasetLoader(s3_handler=mock_handler, bucket_name='test')
        loader.imagenet_loader.load_imagenet_folder = MagicMock(
            return_value=['img1.JPEG', 'img2.JPEG']
        )
        
        files = loader.load_folder('imagenet', 'test')
        assert len(files) == 2
        assert files[0] == 'img1.JPEG'


# Test NMA Service
class TestNMAService:
    """Test NMA service operations"""
    
    @patch('NMA.utilss.files_utils.update_current_model')
    @patch('NMA.classes.dendrogram.Dendrogram')
    @patch('NMA.classes.edges_dataframe.EdgesDataframe')
    @patch('NMA.classes.nma.NMA')
    @patch('NMA.services.dataset_service._load_dataset')
    @patch('NMA.services.dataset_service._get_dataset_config')
    @patch('NMA.services.model_service._load_model')
    @patch('NMA.services.model_service._get_model_filename')
    def test_create_nma_success(self, mock_get_filename, mock_load_model,
                               mock_get_config, mock_load_dataset, mock_nma,
                               mock_edges, mock_dendro, mock_update):
        """Test successful NMA creation"""
        # Setup mocks
        mock_get_filename.return_value = 'test_model.keras'
        mock_model_obj = MagicMock()
        mock_load_model.return_value = MagicMock(model=mock_model_obj)
        
        mock_dataset = MagicMock()
        mock_dataset.labels = ['cat', 'dog']
        mock_dataset.directory_labels = ['n01', 'n02']
        mock_load_dataset.return_value = mock_dataset
        
        mock_get_config.return_value = {
            'dataset': 'imagenet',
            'directory_to_readable': {'n01': 'Cat', 'n02': 'Dog'}
        }
        
        mock_nma_instance = MagicMock()
        mock_nma_instance.Z = np.array([[1, 2, 0.5, 2]])
        mock_nma_instance.edges_df = MagicMock()
        mock_nma.return_value = mock_nma_instance
        
        mock_dendro_instance = MagicMock()
        mock_dendro_instance.get_sub_dendrogram_formatted.return_value = {
            'name': 'root',
            'children': []
        }
        mock_dendro.return_value = mock_dendro_instance
        
        # Call function
        result = _create_nma(
            user_id='test_user',
            model_id='test_model',
            graph_type='similarity',
            dataset='imagenet',
            min_confidence=0.5,
            top_k=5
        )
        
        # Assertions
        assert result is not None
        assert 'name' in result
        mock_update.assert_called_once()
        mock_edges.assert_called_once()
        mock_dendro.assert_called_once()


# Integration tests (skip if no credentials)
@pytest.mark.integration
class TestS3Integration:
    """Integration tests that connect to real S3"""
    
    @pytest.fixture(autouse=True)
    def check_credentials(self):
        """Check if we have valid AWS credentials"""
        try:
            client = get_datasets_s3_client()
            client.list_buckets()
        except (NoCredentialsError, ClientError):
            pytest.skip("No valid AWS credentials for integration tests")
    
    def test_real_bucket_connectivity(self):
        """Test real S3 bucket connectivity"""
        datasets_bucket = os.getenv('S3_DATASETS_BUCKET_NAME')
        users_bucket = os.getenv('S3_USERS_BUCKET_NAME')
        
        datasets_client = get_datasets_s3_client()
        users_client = get_users_s3_client()
        
        # Test datasets bucket
        try:
            response = datasets_client.head_bucket(Bucket=datasets_bucket)
            assert 'ResponseMetadata' in response
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                pytest.skip(f"Bucket {datasets_bucket} not found")
            raise
        
        # Test users bucket
        try:
            response = users_client.head_bucket(Bucket=users_bucket)
            assert 'ResponseMetadata' in response
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                pytest.skip(f"Bucket {users_bucket} not found")
            raise


@pytest.mark.parametrize("dataset_name,expected_info", [
    ("cifar100", {"dataset": "cifar100", "num_classes": 100}),
    ("imagenet", {"dataset": "imagenet", "num_classes": 1000}),
])
def test_dataset_configs(dataset_name, expected_info):
    """Test dataset configurations"""
    with patch('NMA.s3_connector.s3_handler.S3Handler') as mock_handler:
        mock_module = MagicMock()
        setattr(mock_module, f"{dataset_name.upper()}_INFO", expected_info)
        
        mock_handler_instance = MagicMock()
        mock_handler_instance.load_python_module_from_s3.return_value = mock_module
        mock_handler.return_value = mock_handler_instance
        
        loader = S3DatasetLoader(bucket_name='test')
        loader.s3_handler = mock_handler_instance
        
        info = loader.get_dataset_info(dataset_name)
        assert info['dataset'] == dataset_name


# Performance tests
@pytest.mark.performance
def test_s3_operation_performance():
    """Test S3 operation performance"""
    import time
    
    with patch('NMA.utilss.s3_utils.get_users_s3_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.head_object.return_value = {'ContentLength': 1024}
        
        start_time = time.time()
        
        # Test 100 file existence checks
        for i in range(100):
            s3_file_exists('test-bucket', f'test-key-{i}')
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in under 1 second for mocked calls
        assert elapsed < 1.0, f"Performance test took {elapsed:.2f} seconds"


if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([__file__, '-v', '--cov=NMA', '--cov-report=html'])
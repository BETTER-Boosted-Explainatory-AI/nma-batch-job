
import pandas as pd
import os
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)
import io 
from ..utilss.s3_utils import get_users_s3_client

class EdgesDataframe:
    def __init__(self, model_filename, df_filename, edges_df=None):

        logger.info(f"Dataframe file path: {df_filename}")
        
        self.model_filename = model_filename
        self.df_filename = df_filename
        self.edges_df = pd.DataFrame() if edges_df is None else edges_df
        self.s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
        self.using_s3 = self.s3_bucket is not None

    def get_dataframe(self):
        return self.edges_df

    
    ### S3 implementation ### 
    def save_dataframe(self):
        """Save the DataFrame to S3 or local filesystem"""
        try:
            if self.using_s3:
                # Save to S3
                
                ###  not skipped for debugging 
                
                # if self._s3_file_exists(self.s3_bucket, self.df_filename):
                #     logger.info(f'File already exists in S3, skipping save: s3://{self.s3_bucket}/{self.df_filename}')
                #     return
                
                # Convert DataFrame to CSV in memory
                csv_buffer = io.StringIO()
                self.edges_df.to_csv(csv_buffer, index=False)
                
                # Upload to S3
                s3_client = get_users_s3_client() 
                s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=self.df_filename,
                    Body=csv_buffer.getvalue()
                )
                logger.info(f'Edges dataframe has been saved to S3: s3://{self.s3_bucket}/{self.df_filename}')
            else:
                # Save locally (original implementation)
                directory = os.path.dirname(self.df_filename)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f'Ensured directory exists: {directory}')
                
                # if os.path.exists(self.df_filename):
                #     logger.info(f'File already exists, skipping save: {self.df_filename}')
                #     return
                
                self.edges_df.to_csv(self.df_filename, index=False)
                logger.info(f'Edges dataframe has been saved: {self.df_filename}')
        
        except Exception as e:
            logger.error(f'Error saving dataframe: {str(e)}')

    ### S3 implementation ### 
    def load_dataframe(self):
        """Load the DataFrame from S3 or local filesystem"""
        logger.info(f"Attempting to load DataFrame from: {self.df_filename}")
        try:
            if self.using_s3:
                # Load from S3
                if self._s3_file_exists(self.s3_bucket, self.df_filename):
                    # Load directly into memory
                    s3_client = get_users_s3_client() 
                    response = s3_client.get_object(Bucket=self.s3_bucket, Key=self.df_filename)
                    
                    # Read CSV directly from the response body
                    csv_content = response['Body'].read().decode('utf-8')
                    self.edges_df = pd.read_csv(io.StringIO(csv_content))
                    
                    logger.info(f"DataFrame loaded from S3 with shape: {self.edges_df.shape}")
                    logger.info(f"DataFrame columns: {self.edges_df.columns.tolist()}")
                    logger.info(f"First few rows: {self.edges_df.head().to_string()}")
                else:
                    logger.info(f'File not found in S3: s3://{self.s3_bucket}/{self.df_filename}')
            else:
                # Load locally (original implementation)
                if os.path.exists(self.df_filename):
                    logger.info(f"File exists and has size: {os.path.getsize(self.df_filename)} bytes")
                    self.edges_df = pd.read_csv(self.df_filename)
                    logger.info(f"DataFrame loaded with shape: {self.edges_df.shape}")
                    logger.info(f"DataFrame columns: {self.edges_df.columns.tolist()}")
                    logger.info("First few rows:")
                    logger.info(self.edges_df.head())
                else:
                    logger.info(f'File not found: {self.df_filename}')
        except Exception as e:
            logger.error(f"Error loading DataFrame: {str(e)}")
                
    def get_image_probabilities_by_id(self, image_id):
        probabilities_df = self.edges_df[self.edges_df['image_id'] == image_id]
        return probabilities_df

    def get_dataframe_by_count(self):
        return self.edges_df.groupby('source')['target'].value_counts().reset_index(name='count')

    def _s3_file_exists(self, bucket_name, s3_key):
        """Check if a file exists in S3"""
        s3_client = get_users_s3_client()
        logger.info(f"Checking if file exists in S3: {bucket_name}/{s3_key}")
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            logger.info(f"File found: {bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.info(f"File not found: {bucket_name}/{s3_key}, Error: {str(e)}")
            return False
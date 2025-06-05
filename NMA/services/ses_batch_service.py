import boto3
import os
import uuid

def send_email_notification(user_id, model_name, graph_type):
    batch = boto3.client('batch', region_name=os.getenv("AWS_REGION"),
                        aws_access_key_id = os.getenv("AWS_JOBS_ACCESS_KEY_ID"),
                        aws_secret_access_key = os.getenv("AWS_JOBS_SECRET_ACCESS_KEY")) 
    
    job_name = f"ses-job-{uuid.uuid4()}"
    job_queue = os.getenv("JOB_QUEUE_NAME")
    job_definition = os.getenv("JOB_DEFINITION_NAME")

    environment = [
        {'name': 'user_id', 'value': str(user_id)},
        {'name': 'model_name', 'value': str(model_name)},
        {'name': 'graph_type', 'value': str(graph_type)},
    ]

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            'environment': environment
        }
    )

    if response.get("jobId"):
        print(f"Batch job submitted successfully: {response['jobId']}")
    else:
        raise Exception(f"Failed to submit batch job: {response}")
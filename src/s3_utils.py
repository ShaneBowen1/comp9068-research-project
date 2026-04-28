# Object oriented wrapper for S3

import pickle
import json
import os
import numpy as np
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

class S3Client:

    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name):
        print(f"Initializing S3Client...")
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=Config(signature_version='s3v4')
        )

    def upload_file(self, file_path, bucket_name, object_name):
        """Uploads a file to the specified S3 bucket."""
        self.s3.upload_file(file_path, bucket_name, object_name)

    def download_file(self, bucket_name, object_name, file_path):
        """Downloads a file from the specified S3 bucket."""
        self.s3.download_file(bucket_name, object_name, file_path)

    def delete_file(self, bucket_name, object_name):
        """Deletes a file from the specified S3 bucket."""
        self.s3.delete_object(Bucket=bucket_name, Key=object_name)

    def list_folders(self, bucket_name):
        """Lists all folders in the specified S3 bucket."""
        response = self.s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
        return [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]

    def file_exists(self, bucket_name, object_name):
        """Checks if a file exists in the specified S3 bucket."""
        try:
            self.s3.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except self.s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise

    def copy_file(self, source_bucket_name, source_object_name, dest_bucket_name, dest_object_name):
        """Copies a file from one S3 bucket to another."""
        copy_source = {
            'Bucket': source_bucket_name,
            'Key': source_object_name
        }
        self.s3.copy(copy_source, dest_bucket_name, dest_object_name)

    def move_file(self, source_bucket_name, source_object_name, dest_bucket_name, dest_object_name):
        """Moves a file from one S3 bucket to another."""
        self.copy_file(source_bucket_name, source_object_name, dest_bucket_name, dest_object_name)
        self.delete_file(source_bucket_name, source_object_name)

    def list_buckets(self):
        """Lists all S3 buckets in the account."""
        response = self.s3.list_buckets()
        return [bucket['Name'] for bucket in response.get('Buckets', [])]

    def create_bucket(self, bucket_name):
        """Creates a new S3 bucket with the specified name."""
        self.s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': self.s3.meta.region_name}
        )

    def bucket_exists(self, bucket_name):
        """Checks if a bucket exists in the account."""
        response = self.s3.list_buckets()
        return any(bucket['Name'] == bucket_name for bucket in response.get('Buckets', []))

    def list_files_in_folder(self, bucket_name, folder_name):
        """Lists all files in a specific folder within the specified S3 bucket."""
        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name + '/')
        return [obj['Key'] for obj in response.get('Contents', [])]

    def save_json(self, config, bucket_name, object_name):
        """Saves a configuration dictionary to the specified S3 bucket as a JSON file."""
        json_data = json.dumps(config)
        self.s3.put_object(Bucket=bucket_name, Key=object_name, Body=json_data)

    def load_json(self, bucket_name, object_name):
        """Loads a configuration dictionary from the specified S3 bucket."""
        response = self.s3.get_object(Bucket=bucket_name, Key=object_name)
        json_data = response['Body'].read().decode('utf-8')
        return json.loads(json_data)

    def save_object(self, data, bucket_name, object_name):
        """Saves a Python object to the specified S3 bucket as a pickle file."""
        pickle_data = pickle.dumps(data)
        self.s3.put_object(Bucket=bucket_name, Key=object_name, Body=pickle_data)

    def load_object(self, bucket_name, object_name):
        """Loads a Python object from the specified S3 bucket."""
        response = self.s3.get_object(Bucket=bucket_name, Key=object_name)
        pickle_data = response['Body'].read()
        return pickle.loads(pickle_data)

    def list_paginated_files_in_folder(self, bucket_name, folder_name):
        """Lists all files in a specific folder within the specified S3 bucket using pagination."""
        paginator = self.s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder_name + '/')
  
        files = []
        for page in page_iterator:
            files.extend([obj['Key'] for obj in page.get('Contents', [])])
        
        return files

    def _parse_s3_uri(self, s3_uri):
        """Parses an S3 URI and returns the bucket name and folder name."""
        if not s3_uri.startswith("s3://"):
            raise ValueError("Invalid S3 URI. It should start with 's3://'.")

        path_parts = s3_uri[5:].split('/', 1)
        bucket_name = path_parts[0]
        folder_name = path_parts[1].rstrip('/') if len(path_parts) > 1 else ''

        return bucket_name, folder_name

    def load_spectrograms_from_s3(self, folder_uri, num_samples=None):
        """Loads spectrograms and their corresponding file paths from a specified S3 folder URI."""
        bucket_name, folder_name = self._parse_s3_uri(folder_uri)
        file_keys = self.list_paginated_files_in_folder(bucket_name, folder_name)[:num_samples]

        spectrograms = []
        file_paths = []
        for key in file_keys:
            response = self.s3.get_object(Bucket=bucket_name, Key=key)
            spectrogram_data = response['Body'].read()
            spectrogram = pickle.loads(spectrogram_data)
            spectrograms.append(spectrogram)
            file_paths.append(key)

        # Convert list to numpy array
        spectrograms = np.array(spectrograms)
        return spectrograms[..., np.newaxis], file_paths

if __name__ == "__main__":
    # Create an instance of the S3Client
    s3_client = S3Client(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', None),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', None),
        region_name=os.getenv('AWS_REGION', 'eu-west-1')
    )

    # list_of_files = s3_client.list_paginated_files_in_folder(
    #     bucket_name='comp9068-research-project-bucket',
    #     folder_name='data_source/lj_speech/libopus/audio/16k/spectrograms/mel'
    # )
    # print("length: ", len(list_of_files))

    # s3_uri = "s3://comp9068-research-project-bucket/data_source/lj_speech/libopus/audio/16k/spectrograms/mel/"
    # specs, file_paths = s3_client.load_spectrograms_from_s3(s3_uri, num_samples=5)

    # SAVED_MODEL_PATH = "output/tensorflow-training-job-20260426194141"
    # parameters = [(80, 64, 1), (512, 256, 128, 64, 32), (3, 3, 3, 3, 3), (2, 2, 2, 2, 1), 32, 1000.0]

    # s3_client.save_object(
    #     data=parameters,
    #     bucket_name=os.getenv('S3_BUCKET_NAME'),
    #     object_name=os.path.join(SAVED_MODEL_PATH, "parameters.pkl").replace("\\", "/")
    # )
import os
import boto3
from dotenv import load_dotenv
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    SourceCode,
)
from sagemaker.train import ModelTrainer

load_dotenv()

class AWSTrainingJob:
    """
    AWSTrainingJob is responsible for configuring and running a SageMaker training job. It initializes the SageMaker session, retrieves the appropriate TensorFlow image URI for training, and defines the training job configuration, including the compute resources, input data configuration, and hyperparameters. The class also includes error handling to manage exceptions that may occur during the training process.
    """
    def __init__(self, framework, version, py_version, instance_type):
        # Initialize S3 client and SageMaker session
        boto_sess = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
        self.sagemaker_session = Session(boto_session=boto_sess)
        self.region = self.sagemaker_session.boto_region_name
        print(f"SageMaker session region: {self.region}")

        self.framework = framework
        self.version = version
        self.py_version = py_version
        self.instance_type = instance_type

        # Retrieve the appropriate TensorFlow image URI for training
        self.tf_model_image_uri = image_uris.retrieve(
            framework=self.framework,
            region=self.region,
            version=self.version,
            py_version=self.py_version,
            instance_type=self.instance_type,
            image_scope="training",
        )
        print(f"TensorFlow model image URI: {self.tf_model_image_uri}")

    def run_training_job(self, epochs, batch_size, learning_rate, instance_count, train_data_path):
        # Configure the SageMaker training job
        trainer = ModelTrainer(
            sagemaker_session=self.sagemaker_session,
            role=os.getenv("SAGEMAKER_ROLE_ARN", None),
            training_image=self.tf_model_image_uri,
            source_code=SourceCode(
                source_dir="./src",
                entry_script="train.py"
            ),
            compute=Compute(
                instance_type=self.instance_type,
                instance_count=instance_count
            ),
            output_data_config=OutputDataConfig(
                s3_output_path=f"s3://{os.getenv('S3_BUCKET_NAME')}/output"
            ),
            hyperparameters={
                "epochs": str(epochs),
                "batch_size": str(batch_size),
                "learning_rate": str(learning_rate),
            },
        )

        # Define the input data configuration for training
        train_channel = InputData(
            channel_name="train",
            data_source=train_data_path
        )

        # Start the training job
        try:
            trainer.train(input_data_config=[train_channel])
            print("Training job completed successfully.")
        except Exception as e:
            print(f"Error occurred while training: {e}")

if __name__ == "__main__":

    train_data_path = f"s3://{os.getenv('S3_BUCKET_NAME')}/data_source/lj_speech/libopus/audio/16k/spectrograms/"

    training_job = AWSTrainingJob(
        framework="tensorflow",
        version="2.19",
        py_version="py312",
        instance_type="ml.m5.large"
    )
    training_job.run_training_job(
        epochs=10,
        batch_size=8,
        learning_rate=0.0005,
        instance_count=1,
        train_data_path=train_data_path
    )

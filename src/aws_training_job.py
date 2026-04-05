import argparse
import os
import boto3
from dotenv import load_dotenv
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import (
    Compute,
    CheckpointConfig,
    InputData,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
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
            base_job_name="tensorflow-training-job",
            source_code=SourceCode(
                source_dir="./src",
                requirements="requirements.txt",
                entry_script="train.py"
            ),
            compute=Compute(
                instance_type=self.instance_type,
                instance_count=instance_count,
                enable_managed_spot_training=True,  # Enable spot instances to reduce costs
            ),
            output_data_config=OutputDataConfig(
                s3_output_path=f"s3://{os.getenv('S3_BUCKET_NAME')}/output"
            ),
            hyperparameters={
                "epochs": str(epochs),
                "batch_size": str(batch_size),
                "learning_rate": str(learning_rate),
            },
            checkpoint_config=CheckpointConfig(
                s3_uri=f"s3://{os.getenv('S3_BUCKET_NAME')}/checkpoints",
                local_path="/opt/ml/checkpoints"
            ),
            stopping_condition=StoppingCondition(
                max_runtime_in_seconds=86400,   # 24 hours
                max_wait_time_in_seconds=108000 # 30 hours
            ),
            environment={
                "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME")
            }
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
    
    parser = argparse.ArgumentParser(description="Run AWS training job.")

    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for training.")
    parser.add_argument("--instance_count", type=int, default=1, help="Number of instances for training.")
    parser.add_argument("--train_path_uri", type=str, required=True, help="S3 URI for training data.")
    parser.add_argument("--is_gpu", action="store_true", default=False, help="Flag to indicate whether to use GPU instance.")
    args = parser.parse_args()

    instanct_type = "ml.g5.2xlarge" if args.is_gpu else "ml.m5.large"

    training_job = AWSTrainingJob(
        framework="tensorflow",
        version="2.19",
        py_version="py312",
        instance_type=instanct_type
    )
    training_job.run_training_job(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        instance_count=args.instance_count,
        train_data_path=args.train_path_uri
    )

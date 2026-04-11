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
    TensorBoardOutputConfig
)
from sagemaker.train import ModelTrainer
from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.core.parameter import ContinuousParameter, IntegerParameter, CategoricalParameter

load_dotenv()

class AWSTrainingJob:
    """
    AWSTrainingJob is responsible for configuring and running a SageMaker training job. It initializes the SageMaker session, retrieves the appropriate TensorFlow image URI for training, and defines the training job configuration.
    """
    def __init__(self, framework, version, py_version, instance_type, instance_count, base_job_name):
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
        self.instance_count = instance_count
        self.base_job_name = base_job_name

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
    
    def _build_model_trainer(self, hyperparameters=None):
        """
        Build the ModelTrainer configuration for the SageMaker training job
        """

        return ModelTrainer(
            sagemaker_session=self.sagemaker_session,
            role=os.getenv("SAGEMAKER_ROLE_ARN", None),
            training_image=self.tf_model_image_uri,
            base_job_name=self.base_job_name,
            source_code=SourceCode(
                source_dir="./src",
                requirements="requirements.txt",
                entry_script="train.py"
            ),
            compute=Compute(
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                enable_managed_spot_training=True,  # Enable spot instances to reduce costs
            ),
            output_data_config=OutputDataConfig(
                s3_output_path=f"s3://{os.getenv('S3_BUCKET_NAME')}/output"
            ),
            checkpoint_config=CheckpointConfig(),
            stopping_condition=StoppingCondition(
                max_runtime_in_seconds=86400,   # 24 hours
                max_wait_time_in_seconds=108000 # 30 hours
            ),
            environment={
                "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME")
            },
            hyperparameters=hyperparameters
        ).with_tensorboard_output_config(TensorBoardOutputConfig())   # Enable TensorBoard output for monitoring training progress

    def run_training_job(
            self,
            epochs,
            batch_size,
            learning_rate,
            reconstruction_loss_weight,
            noisy_data_path,
            clean_data_path
        ):
        """
        Configure and start the SageMaker training job with the specified hyperparameters and input data
        """
        hyperparameters = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "reconstruction_loss_weight": reconstruction_loss_weight
        }
        trainer = self._build_model_trainer(hyperparameters)
    
        # Define the input data configuration for training
        noisy_channel = InputData(
            channel_name="noisy",
            data_source=noisy_data_path
        )
        clean_channel = InputData(
            channel_name="clean",
            data_source=clean_data_path
        )

        # Start the training job
        try:
            trainer.train(input_data_config=[noisy_channel, clean_channel])
            print("Training job completed successfully.")
        except Exception as e:
            print(f"Error occurred while training: {e}")

    def run_hyperparameter_tuning_job(
            self,
            hyperparameter_ranges,
            objective_metric_name,
            objective_type,
            max_jobs,
            max_parallel_jobs,
            noisy_data_path,
            clean_data_path,
            job_name
        ):
        """
        Configure the HyperparameterTuner for SageMaker hyperparameter tuning
        """

        tuner = HyperparameterTuner(
            model_trainer=self._build_model_trainer(),
            objective_metric_name=objective_metric_name,
            objective_type=objective_type,
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[
                {
                    "Name": objective_metric_name,
                    "Regex": f"{objective_metric_name}: ([0-9\\.]+)"
                }
            ],
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            strategy="Random",
            early_stopping_type="Auto"
        )

        # Start the hyperparameter tuning job
        try:
            tuner.tune(
                inputs={
                    "noisy": noisy_data_path,
                    "clean": clean_data_path
                },
                job_name=job_name
            )
            best_job_name = tuner.best_training_job()
            print(f"Hyperparameter tuning job completed. Best training job: {best_job_name}")
        except Exception as e:
            print(f"Error occurred while starting hyperparameter tuning job: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run AWS training job.")

    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for training.")
    parser.add_argument("--reconstruction_loss_weight", type=float, default=1000.0, help="Weight for the reconstruction loss in the combined loss function.")
    parser.add_argument("--instance_count", type=int, default=1, help="Number of instances for training.")
    parser.add_argument("--noisy_path_uri", type=str, required=True, help="S3 URI for noisy training data.")
    parser.add_argument("--clean_path_uri", type=str, required=True, help="S3 URI for clean training data.")
    parser.add_argument("--is_gpu", action="store_true", default=False, help="Flag to indicate whether to use GPU instance.")
    args = parser.parse_args()

    instanct_type = "ml.g5.2xlarge" if args.is_gpu else "ml.m5.large"

    training_job = AWSTrainingJob(
        framework="tensorflow",
        version="2.19",
        py_version="py312",
        instance_type=instanct_type,
        instance_count=args.instance_count,
        base_job_name="tensorflow-training-job"
    )
    # training_job.run_training_job(
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     learning_rate=args.learning_rate,
    #     reconstruction_loss_weight=args.reconstruction_loss_weight,
    #     noisy_data_path=args.noisy_path_uri,
    #     clean_data_path=args.clean_path_uri
    # )

    # Hyperparameter tuning configuration
    hyperparameter_ranges = {
        "learning_rate": ContinuousParameter(0.0001, 0.001),
        "batch_size": CategoricalParameter([16, 32, 64]),
        "reconstruction_loss_weight":CategoricalParameter([1000.0, 10000.0, 100000.0]),
        "epochs": CategoricalParameter([10, 15, 20])
    }

    training_job.run_hyperparameter_tuning_job(
        hyperparameter_ranges=hyperparameter_ranges,
        objective_metric_name="val_loss",
        objective_type="Minimize",
        max_jobs=1,
        max_parallel_jobs=1,
        noisy_data_path=args.noisy_path_uri,
        clean_data_path=args.clean_path_uri,
        job_name="hyperparameter-tuning-job"
    )

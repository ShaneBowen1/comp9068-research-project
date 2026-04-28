import argparse
import os
import boto3
import numpy as np
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
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.parameter import ContinuousParameter, CategoricalParameter
import cloudpickle
import s3_utils
import vae as _vae
import vae_inference_spec
from vae_inference_spec import VAEInferenceSpec


for _mod in (vae_inference_spec, s3_utils, _vae):
    cloudpickle.register_pickle_by_value(_mod)
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
            base_filters,
            n_layers,
            latent_space_dim,
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
            "reconstruction_loss_weight": reconstruction_loss_weight,
            "base_filters": base_filters,
            "n_layers": n_layers,
            "latent_space_dim": latent_space_dim
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
            job_name = trainer._latest_training_job.training_job_name
            print(f"Training job completed successfully: {job_name}")
            return trainer, job_name
        except Exception as e:
            print(f"Error occurred while training: {e}")
            return None, None

    def deploy_model(self, job_name, endpoint_instance_type, initial_instance_count, endpoint_name=None):
        """
        Deploy the artifact from the last completed ``training_job``.
        """
        VAE_IO_SHAPE = (1, 80, 64, 1)
        sample_input = np.zeros(VAE_IO_SHAPE, dtype=np.float32)
        sample_output = np.zeros(VAE_IO_SHAPE, dtype=np.float32)

        bucket = os.environ["S3_BUCKET_NAME"]
        artifact_prefix_uri = f"s3://{bucket}/output/{job_name}"
        print(f"VAE artifact prefix: {artifact_prefix_uri}")

        # inference_image_uri = image_uris.retrieve(
        #     framework=self.framework,
        #     region=self.region,
        #     version=self.version,
        #     py_version=self.py_version,
        #     instance_type=endpoint_instance_type,
        #     image_scope="inference",
        # )
        # print(f"Inference image URI: {inference_image_uri}")

        # Retrieve the appropriate TensorFlow image URI for inference
        inference_image_uri = image_uris.retrieve(
            framework="pytorch",
            region=self.region,
            version="2.3",
            py_version="py311",
            instance_type=endpoint_instance_type,
            image_scope="inference",
        )
        print(f"Inference image URI: {inference_image_uri}")

        inference_requirements = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "inference_requirements.txt"
        )
        model_builder = ModelBuilder(
            inference_spec=VAEInferenceSpec(s3_artifact_prefix_uri=artifact_prefix_uri),
            schema_builder=SchemaBuilder(sample_input, sample_output),
            role_arn=os.getenv("SAGEMAKER_ROLE_ARN"),
            instance_type=endpoint_instance_type,
            image_uri=inference_image_uri,
            model_server=ModelServer.TORCHSERVE,
            dependencies={"auto": False, "requirements": inference_requirements},
        )
        model_builder.build()  # Creates Model Resource
        endpoint = model_builder.deploy(
            endpoint_name=endpoint_name,
            initial_instance_count=initial_instance_count,
            instance_type=endpoint_instance_type
        )  # Creates Endpoint Resource
        print(f"ModelBuilder deploy finished: {endpoint}")
        return endpoint

    def run_hyperparameter_tuning_job(
            self,
            epochs,
            hyperparameter_ranges,
            objective_metric_name,
            objective_type,
            max_jobs,
            max_parallel_jobs,
            noisy_data_path,
            clean_data_path
        ):
        """
        Configure the HyperparameterTuner for SageMaker hyperparameter tuning
        """

        tuner = HyperparameterTuner(
            model_trainer=self._build_model_trainer(
                hyperparameters={
                    "epochs": epochs  # Keep the number of epochs static across all tuning jobs
                }
            ),
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
            strategy="Bayesian",
            early_stopping_type="Auto",
            random_seed=42,
        )

        # Start the hyperparameter tuning job
        try:
            tuner.tune(
                inputs={
                    "noisy": noisy_data_path,
                    "clean": clean_data_path
                }
            )
            best_job_name = tuner.best_training_job()
            if best_job_name:
                print(f"Hyperparameter tuning job completed. Best training job: {best_job_name}")

                # Best hyperparameters can be retrieved from the best training job
                desc = tuner.describe()
                best_hyperparameters = desc.best_training_job.tuned_hyper_parameters
                print(f"Best hyperparameters: {best_hyperparameters}")
                return best_hyperparameters
            return None

        except Exception as e:
            print(f"Error occurred while starting hyperparameter tuning job: {e}")
            return None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run AWS training job.")

    # Hyperparameters for training
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for training.")
    parser.add_argument("--reconstruction_loss_weight", type=float, default=1000.0, help="Weight for the reconstruction loss in the combined loss function.")
    parser.add_argument("--base_filters", type=int, default=256, help="Number of filters in the first convolutional layer of the VAE.")
    parser.add_argument("--n_layers", type=int, default=5, help="Number of convolutional layers in the VAE.")
    parser.add_argument("--latent_space_dim", type=int, default=128, help="Dimensionality of the latent space in the VAE.")
    
    # SageMaker specific arguments
    parser.add_argument("--noisy_path_uri", type=str, required=True, help="S3 URI for noisy training data.")
    parser.add_argument("--clean_path_uri", type=str, required=True, help="S3 URI for clean training data.")
    parser.add_argument("--is_gpu", action="store_true", default=False, help="Flag to indicate whether to use GPU instance.")
    parser.add_argument("--instance_count", type=int, default=1, help="Number of instances for training.")
    
    # Hyperparameter tuning specific arguments
    parser.add_argument("--max_jobs", type=int, default=10, help="Maximum number of hyperparameter tuning jobs to run.")
    parser.add_argument("--max_parallel_jobs", type=int, default=2, help="Maximum number of hyperparameter tuning jobs to run in parallel.")
    
    # Model Deployment specific arguments
    parser.add_argument("--endpoint_instance_type", type=str, default="ml.m5.large", help="Instance type for ModelBuilder endpoint deploy.")
    parser.add_argument("--endpoint_instance_count", type=int, default=1, help="Instance count for endpoint deploy.")
    parser.add_argument("--endpoint_name", type=str, default=None, help="Optional endpoint name prefix (SDK may append a unique suffix).")
    parser.add_argument("--skip_deploy", action="store_true", help="If set, skip ModelBuilder deploy after final training.")
    args = parser.parse_args()

    # Get instance type
    instance_type = "ml.g5.2xlarge" if args.is_gpu else "ml.m5.large"

    training_job = AWSTrainingJob(
        framework="tensorflow",
        version="2.19",
        py_version="py312",
        instance_type=instance_type,
        instance_count=args.instance_count,
        base_job_name="tensorflow-training-job"
    )
    trainer, job_name = training_job.run_training_job(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        base_filters=args.base_filters,
        n_layers=args.n_layers,
        latent_space_dim=args.latent_space_dim,
        noisy_data_path=args.noisy_path_uri,
        clean_data_path=args.clean_path_uri
    )

    if trainer and not args.skip_deploy:
        training_job.deploy_model(
            job_name=job_name,
            endpoint_instance_type=args.endpoint_instance_type,
            initial_instance_count=args.endpoint_instance_count,
            endpoint_name=args.endpoint_name,
        )

    # # Hyperparameter tuning configuration
    # hyperparameter_ranges = {
    #     "learning_rate": ContinuousParameter(0.00001, 0.001),
    #     "batch_size": CategoricalParameter([16, 32, 64]),
    #     "reconstruction_loss_weight":CategoricalParameter([100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0]),
    #     "latent_space_dim": CategoricalParameter([16, 32, 64, 128, 256]),
    #     "base_filters": CategoricalParameter([64, 128, 256]),
    #     "n_layers": CategoricalParameter([3, 4, 5])
    # }

    # best_hyperparameters = training_job.run_hyperparameter_tuning_job(
    #     hyperparameter_ranges=hyperparameter_ranges,
    #     epochs=args.epochs,
    #     objective_metric_name="val_loss",
    #     objective_type="Minimize",
    #     max_jobs=args.max_jobs,
    #     max_parallel_jobs=args.max_parallel_jobs,
    #     noisy_data_path=args.noisy_path_uri,
    #     clean_data_path=args.clean_path_uri
    # )

    # if best_hyperparameters:
    #     # Number of epochs for the final training job
    #     EPOCHS = 150

    #     # Final training with best hyperparameters, then ModelBuilder deploy from that job.
    #     final_trainer = training_job.run_training_job(
    #         epochs=EPOCHS,
    #         batch_size=int(best_hyperparameters["batch_size"]),
    #         learning_rate=float(best_hyperparameters["learning_rate"]),
    #         reconstruction_loss_weight=float(best_hyperparameters["reconstruction_loss_weight"]),
    #         base_filters=int(best_hyperparameters["base_filters"]),
    #         n_layers=int(best_hyperparameters["n_layers"]),
    #         latent_space_dim=int(best_hyperparameters["latent_space_dim"]),
    #         noisy_data_path=args.noisy_path_uri,
    #         clean_data_path=args.clean_path_uri
    #     )

    #     if final_trainer and not args.skip_deploy:
    #         training_job.deploy_model(
    #             trainer=final_trainer,
    #             endpoint_instance_type=args.endpoint_instance_type,
    #             initial_instance_count=args.endpoint_instance_count,
    #             endpoint_name=args.endpoint_name,
    #         )

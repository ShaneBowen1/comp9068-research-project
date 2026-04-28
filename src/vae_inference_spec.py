"""Inference spec for the trained VAE model used by SageMaker ModelBuilder."""
import os
from urllib.parse import urlparse

import numpy as np
from sagemaker.serve.spec.inference_spec import InferenceSpec

import s3_utils
from vae import VAE


class VAEInferenceSpec(InferenceSpec):
    """Loads the saved VAE (parameters.pkl + model.weights.h5) and returns reconstructed spectrograms."""

    ARTIFACT_FILES = ("parameters.pkl", "model.weights.h5")
    DEFAULT_LOCAL_DIR = "/opt/ml/model"

    def __init__(self, s3_artifact_prefix_uri):
        super().__init__()
        self.s3_artifact_prefix_uri = s3_artifact_prefix_uri.rstrip("/")

    def load(self, model_dir):
        """
        Downloads the saved VAE artifacts from S3 and loads the model.
        """
        # ModelBuilder may pass an s3:// URI at build-time validation; at runtime
        # SageMaker passes a real local dir. Always resolve to a writable local dir.
        target_dir = model_dir if model_dir and not model_dir.startswith("s3://") else self.DEFAULT_LOCAL_DIR
        os.makedirs(target_dir, exist_ok=True)
        self._download_artifacts(target_dir)
        return VAE.load(target_dir)

    def invoke(self, input_object, model):
        """
        Invokes the VAE model with the input spectrogram and returns the reconstructed spectrogram.
        """
        x = np.asarray(input_object, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, ...]
        reconstructed, _z = model.reconstruct(x)
        return reconstructed

    def _download_artifacts(self, local_dir):
        """
        Downloads the parameters.pkl + model.weights.h5 from S3 and saves them to the local directory.
        """
        s3_client = s3_utils.S3Client(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "eu-west-1"),
        )
        parsed = urlparse(self.s3_artifact_prefix_uri)
        bucket_name = parsed.netloc
        key_prefix = parsed.path.lstrip("/")
        for filename in self.ARTIFACT_FILES:
            s3_client.download_file(
                bucket_name=bucket_name,
                object_name=f"{key_prefix}/{filename}",
                file_path=os.path.join(local_dir, filename),
            )

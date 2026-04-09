# COMP9068 Research Project
A Generative AI Approach for Audio Restoration due to Compression for Speech Enhancement

## Usage

1. Open up the terminal, go to the root directory (comp9068-research-project)

2. Build the Docker image:
    ```
    docker build -t test . -f Dockerfile
    ```

3. Run scripts:
    - transcode_audio.py
        - `--format` Specify audio format to transcode (e.g., mp3, opus) *Required
        - `--bitrate` Specify bitrate to transcode (e.g., 128) **Default=\[16\]
        - `--sample_rate` Specify sample rate to transcode (e.g., 22050) *Optional
        - `--channels` Specify number of audio channels  (e.g., 1 for mono, 2 for stereo) **Default=1
        - `--codec` Specify method for encoding (e.g., libopus) **Default=libopus
        - `--application` Application type for encoding (e.g., voip, audio, lowdelay) **Default=audio
        - `--samples` Specify number of samples (e.g., 100) **Default=all
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source test python src/transcode_audio.py --format opus
        ```

    - audio_analysis.py
        - `--format` Specify format to analyse (e.g., wav, opus) *Required
        - `--metric` Specify metric to calculate (e.g., PESQ, STOI) *Required
        - `--bitrate` Specify bitrate (e.g., 128) *Optional **Default=16
        - `--codec` Specify method for encoding (e.g., libopus) **Default=libopus
        - `--application` Application type for encoding (e.g., voip, audio, lowdelay) **Default=audio
        - `--samples` Specify number of samples (e.g., 100) **Default=all
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source test python src/audio_analysis.py --format opus --metric PESQ
        ```
    
    - preprocess.py

        # Use SageMaker default if not passed
        - `--frame_size` Frame size for STFT **Default=512
        - `--hop_length` Hop length for STFT **Default=256
        - `--duration` Duration of audio to load (in seconds) **Default=0.74
        - `--sample_rate` Sample rate for loading audio **Default=22050
        - `--mono` Whether to convert audio to mono **Default=True
        - `--samples` Number of samples to preprocess **Default=all
        - `--files_dir` Directory containing the audio files to preprocess *Required
        - `--spectrogram_save_dir` Directory to save the spectrogram features *Required
        - `--min_max_values_save_dir` Directory to save the min-max values for denormalisation *Required
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source --env-file .env test python src/preprocess.py
        ```
    
    - train.py
        - `--epochs` Specify number of epochs for training (e.g., 100) *Optional **Default=150
        - `--batch_size` Specify batch size for training  (e.g., 32) *Optional **Default=64
        - `--learning_rate` Specify learning rate for training (e.g. 0.005) *Optional **Default=0.0005
        - `--model_dir` Directory to save the trained model *Optional **Default=output/{TRAINING_JOB_NAME}
        - `--train_dir` Directory containing the training data *Optional **Default={SM_CHANNEL_TRAIN}
        ```
        docker run --rm -it --name test --volume ./data_source:/app/data_source --env-file .env test python src/train.py
        ```

    - aws_training_job.py
        - `--epochs` Specify number of epochs for training (e.g., 100) *Optional **Default=150
        - `--batch_size` Specify batch size for training  (e.g., 32) *Optional **Default=64
        - `--learning_rate` Specify learning rate for training (e.g. 0.005) *Optional **Default=0.0005
        - `--instance_count` Specify number of instance for training (e.g., 2) *Optional **Default=1
        - `--noisy_path_uri` S3 URI path for training data **Required
        - `--clean_path_uri` S3 URI path for clean training data **Required
        - `--is_gpu` Flag to indicate whether to use GPU instance *Optional **Default=False
        ```
        python src/aws_training_job.py --noisy_path_uri s3://comp9068-research-project-bucket/data_source/lj_speech/libopus/audio/4k/spectrograms/ --clean_path_uri s3://comp9068-research-project-bucket/data_source/lj_speech/clean/spectrograms/ --is_gpu
        ```

    - generate.py
    ```
    docker run --rm -it --name test --volume ./data_source:/app/data_source --env-file .env test python src/generate.py
    ```

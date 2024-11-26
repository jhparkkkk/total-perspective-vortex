import os
import argparse

from src.constants import N_FFT, DEFAULT_MONTAGE, DATA_DIR, TERMINAL_WIDTH
from src.EEGDataLoader import EEGDataLoader
from src.EEGDataVizualizer import EEGDataVizualizer
from src.EEGPreprocessor import EEGPreprocessor
from src.data_models import EEGData

import matplotlib.pyplot as plt

from mne.datasets import eegbci
from mne.channels import make_standard_montage

import mne

from sklearn.decomposition import PCA, FastICA

from mne.decoding import UnsupervisedSpatialFilter
import numpy as np

from mne.preprocessing import find_bad_channels_maxwell

import glob

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle 
import time
from sklearn.metrics import accuracy_score
def plot(file_path: [str]):
    """
    Load the EEG data and plot it raw and after preprocessing (filtering)
    """
    if len(file_path) != 1:
        raise ValueError("Only a single file path is allowed for plotting")
    file_path = file_path[0]
    # Load the data
    loader = EEGDataLoader()
    raw = loader.load_data(file_path=file_path)
    loader.describe_eeg_data(raw, file_path)

    # Visualize the data before preprocessing
    vizualizer = EEGDataVizualizer()
    vizualizer.plot_eeg(
        raw, title=f"Raw EEG Data for Subject {loader.subject_id} Run {loader.run_id}"
    )

    # Preprocess the data
    preprocessor = EEGPreprocessor()
    filtered_raw = preprocessor.filter_data(raw=raw)
    preprocessor.compute_psd(raw=filtered_raw)

    # Visualize the data after preprocessing
    vizualizer.plot_eeg(
        filtered_raw,
        title=f"Filtered EEG Data for Subject {loader.subject_id} Run {loader.run_id}",
    )


def train(file_paths: list[str]):
    # train entire dataset
    if not file_paths:
        file_paths = glob.glob(os.path.join(DATA_DIR, "S*", "*.edf"))

    all_features = []
    all_labels = []

    loader = EEGDataLoader()
    preprocessor = EEGPreprocessor()
    for file_path in file_paths:
        print("Loading data from: ", file_path)
        # load data
        raw = loader.load_data(file_path=file_path)
        events = loader.extract_events(raw)

        # preprocessing
        filtered_raw = preprocessor.preprocess(raw=raw)
        epochs = preprocessor.epoch_data(
            filtered_raw, events, loader.extract_run_id(file_path)
        )
        print("remaining epochs:", len(epochs))
        if len(epochs) == 0:
            print(f"Skipping file {file_path} (no valid epochs remaining).")
            continue

        # extract features
        features = preprocessor.extract_features(epochs)
        labels = epochs.events[:, -1]
        print("labels:", labels)

        all_features.append(features)
        all_labels.append(labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99)),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )
    scores = cross_val_score(pipeline, all_features, all_labels, cv=6)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", np.mean(scores))

    pipeline.fit(all_features, all_labels)

    with open("model.pkl", "wb") as model_file:
        pickle.dump(pipeline, model_file)
    print("Model saved to model.pkl")


def simulate_real_time_predictions(pipeline, features: np.ndarray, labels: np.ndarray, delay: float = 2.0):
    """
    Simulate real-time predictions on streaming data.
    
    Args:
        - pipeline: The trained pipeline for making predictions
        - features: Array of features for each epoch
        - labels: Corresponding labels for the chunks
        - delay: Time delay (in seconds) between each epoch
    """
    
    header = f"{'Epoch':<10}{'Prediction':<15}{'Truth':<10}{'Equal?':<10}"
    print(header)

    for i, (feature, label) in enumerate(zip(features, labels)):
        prediction = pipeline.predict(feature.reshape(1, -1))
        result = f"{i:<10}{prediction[0]:<15}{label:<10}{'True' if prediction[0] == label else 'False':<10}"
        print(result)
        time.sleep(delay)


def predict(file_paths: list[str]):
    if not file_paths:
        raise ValueError("No file paths provided for prediction")
    
    file_path = file_paths[0]
    with open("model.pkl", "rb") as model_file:
        pipeline = pickle.load(model_file)
    print("Model loaded from model.pkl")

    loader = EEGDataLoader()
    preprocessor = EEGPreprocessor()

    raw = loader.load_data(file_path=file_path)
    events = loader.extract_events(raw)
    filtered_raw = preprocessor.preprocess(raw=raw)
    epochs = preprocessor.epoch_data(
        filtered_raw, events, loader.extract_run_id(file_path)
    )
    print("remaining epochs:", len(epochs))

    features = preprocessor.extract_features(epochs)
    labels = epochs.events[:, -1]

    predictions = pipeline.predict(features)
    print("Predictions:", predictions)
    accuracy = accuracy_score(labels, predictions)
    print("Accuracy:", accuracy)


    simulate_real_time_predictions(pipeline, features, labels)
    

    pass


def create_file_path(subject_id: int, recording_ids: list[int]) -> list[str]:
    """
    Create a list of file paths based on subject ID and recording IDs.
    param:
        - subject_id: the subject ID
        - recording_ids: the list of recording IDs
    return:
        - file_paths: the list of file paths
    """
    if subject_id is None or not recording_ids:
        return []

    file_paths = [
        os.path.join(DATA_DIR, f"S{subject_id:03}", f"S{subject_id:03}R{rec_id:02}.edf")
        for rec_id in recording_ids
    ]

    return file_paths


def parse_arguments():
    """
    Parse the command line arguments
    """

    def validate_subject_id(arg: str) -> int:
        """
        Validate a single subject ID.
        """
        value = int(arg)
        if value < 1 or value > 109:
            raise argparse.ArgumentTypeError("Subject ID must be between 1 and 109")
        return value

    def validate_recording_id(arg: str) -> int:
        """
        Validate a single recording ID.
        """
        value = int(arg)
        if value < 1 or value > 14:
            raise argparse.ArgumentTypeError(
                "Each Recording ID must be between 1 and 14"
            )
        return value

    parser = argparse.ArgumentParser(description="EEG Data Analysis")

    parser.add_argument(
        "mode",
        type=str,
        choices=["plot", "train", "predict"],
        help="Mode of operation: visualize EEG, train model, make predictions",
    )
    parser.add_argument(
        "--subject",
        type=validate_subject_id,
        help="Enter the subject ID (between 1 and 109)",
    )
    parser.add_argument(
        "--recording",
        type=validate_recording_id,
        nargs="+",
        help="Enter a space-separated list of recording IDs (between 1 and 14)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("----------------- 🧠 EEG Data Analysis 🧠 -----------------")
    # parse cmd arguments
    args = parse_arguments()
    file_path = create_file_path(args.subject, args.recording)
    mode_map = {
        "plot": plot,
        "train": train,
        "predict": predict,
    }

    # try:
    mode_function = mode_map.get(args.mode)
    if mode_function is None:
        print("Invalid mode. Please choose from: plot, train, predict")

    print(file_path)
    mode_function(file_path)
    # except Exception as e:
    # print("An error occurred:")
    # print(e)
    # exit(1)

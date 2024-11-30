import os
import argparse

from src.constants import (
    N_FFT,
    DEFAULT_MONTAGE,
    DATA_DIR,
    TERMINAL_WIDTH,
    BAD_SUBJECTS,
    RUNS,
)
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
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import time


def plot(subject_id: list[int], run_id: list[int]):
    """
    Load the EEG data and plot it raw and after preprocessing (filtering)
    """
    if len(subject_id) != 1 or len(run_id) != 1:
        raise ValueError("Only a single file path is allowed for plotting")
    file_path = create_file_path(subject_id[0], run_id[0])

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
    raw = preprocessor.set_montage(raw)
    filtered_raw = preprocessor.filter_data(raw=raw)
    preprocessor.compute_psd(raw=filtered_raw)

    # Visualize the data after preprocessing
    vizualizer.plot_eeg(
        filtered_raw,
        title=f"Filtered EEG Data for Subject {loader.subject_id} Run {loader.run_id}",
    )


def train(subject_list: list[int], run_list: list[int]):

    all_features = []
    all_labels = []

    loader = EEGDataLoader()
    preprocessor = EEGPreprocessor()

    for subject in subject_list:
        raw = loader.load_raws(subject, run_list)
        preprocessed_raw = preprocessor.preprocess(raw)
        features, labels = preprocessor.extract_features_and_labels(preprocessed_raw)
        all_features.append(features)
        all_labels.append(labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99)),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", np.mean(cv_scores))

    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print("Test accuracy:", test_score)

    with open(f"model.pkl", "wb") as model_file:
        pickle.dump(pipeline, model_file)
    print("> Model saved to model.pkl")
    return np.mean(cv_scores), test_score


def run_experiment(args):
    """
    Wrapper for training a single experiment.
    """
    runs, subject_list = args
    return train(subject_list, runs)


def experiment(subject_list: list[int], run_list: list[int]):
    """
    Run experiments in parallel using processes.
    """
    subject_list = [s for s in range(1, 110) if s not in BAD_SUBJECTS]
    tasks = [(runs, subject_list) for runs in RUNS]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment, tasks))

    for i, (cv_score, test_score) in enumerate(results):
        print(f"Experiment {i + 1}: Test Accuracy = {test_score:.4f}")

    print(f"\nMean Accuracy: {np.mean([res[1] for res in results]):.4f}")


def simulate_real_time_predictions(
    pipeline, features: np.ndarray, labels: np.ndarray, delay: float = 0.1
):
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


def predict(subject_list: list[int], run_list: list[int]):

    for s in subject_list:
        for r in run_list:
            file_path = create_file_path(s, r)
            with open("model.pkl", "rb") as model_file:
                pipeline = pickle.load(model_file)
            print("Model loaded from model.pkl")

            loader = EEGDataLoader()
            preprocessor = EEGPreprocessor()

            raw = loader.load_data(file_path=file_path)
            filtered_raw = preprocessor.preprocess(raw=raw)
            features, labels = preprocessor.extract_features_and_labels(filtered_raw)

            predictions = pipeline.predict(features)
            print("Predictions:", predictions)
            print("Labels:", labels)

            accuracy = accuracy_score(labels, predictions)
            print("Accuracy:", accuracy)
            simulate_real_time_predictions(pipeline, features, labels)


def create_file_path(subject_id: int, run_id: int) -> str:
    """
    Create a list of file paths based on subject ID and recording IDs.
    param:
        - subject_id: the subject ID
        - recording_ids: the list of recording IDs
    return:
        - file_paths: the list of file paths
    """
    return os.path.join(
        DATA_DIR, f"S{subject_id:03}", f"S{subject_id:03}R{run_id:02}.edf"
    )

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
        "--mode",
        type=str,
        choices=["plot", "train", "predict", "experiment"],
        help="Mode of operation: visualize EEG, train model, make predictions, experiment with all subjects",
        default="experiment",
    )
    parser.add_argument(
        "--subject",
        type=validate_subject_id,
        nargs="+",
        help="Enter the subject ID (between 1 and 109)",
        default=list(range(1, 110)),
    )
    parser.add_argument(
        "--recording",
        type=validate_recording_id,
        nargs="+",
        help="Enter a space-separated list of recording IDs (between 3 and 14)",
        default=list(range(3, 15)),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("----------------- 🧠 EEG Data Analysis 🧠 -----------------")
    args = parse_arguments()

    mode_map = {
        "plot": plot,
        "train": train,
        "predict": predict,
        "experiment": experiment,
    }

    # try:
    mode_function = mode_map.get(args.mode)
    if mode_function is None:
        print("Invalid mode. Please choose from: plot, train, predict")

    # print(file_path)
    mode_function(args.subject, args.recording)
    # except Exception as e:
    # print("An error occurred:")
    # print(e)
    # exit(1)

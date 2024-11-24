import os
import argparse

from src.constants import N_FFT, DEFAULT_MONTAGE, DATA_DIR
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


def plot(file_path: str):
    # Load the data
    data_loader = EEGDataLoader()
    eeg_data: EEGData = data_loader.load_data(file_path=file_path)
    data_loader.describe_eeg_data()

    # Visualize the data
    eeg_vizualizer = EEGDataVizualizer()
    eeg_vizualizer.plot_eeg(eeg_data.raw)

    # Preprocess the data
    preprocessor = EEGPreprocessor()
    # psd = preprocessor.compute_psd(raw=eeg_data.raw)
    filtered_raw = preprocessor.filter_data(raw=eeg_data.raw)

    # Visualize the data
    eeg_vizualizer.plot_eeg(filtered_raw)


def train(file_path: str):
    if file_path:
        file_paths = [file_path]
    else:
        file_paths = glob.glob(os.path.join(DATA_DIR, "S*", "*.edf"))
    all_features = []
    all_labels = []

    data_loader = EEGDataLoader()
    preprocessor = EEGPreprocessor()
    for file_path in file_paths:
        print("Loading data from: ", file_path)
        eeg_data: EEGData = data_loader.load_data(file_path=file_path)
        filtered_raw = preprocessor.preprocess(raw=eeg_data.raw)
        epochs = preprocessor.epoch_data(filtered_raw, eeg_data.events)
        X = epochs.get_data()
        y = epochs.events[:, -1]

        # compatible with sklearn
        n_epochs, n_channels, n_times = X.shape
        print("Shape of X:", X.shape)
        reshaped_X = X.reshape(n_epochs, n_channels * n_times)

        all_features.append(reshaped_X)
        all_labels.append(y)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99)), 
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )
    scores = cross_val_score(pipeline, all_features, all_labels, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", np.mean(scores))


def predict():
    pass


def create_file_path(subject_id: int, recording_id: int):
    if subject_id is None or recording_id is None:
        return None
    return os.path.join(
        DATA_DIR, f"S{subject_id:03}", f"S{subject_id:03}R{recording_id:02}.edf"
    )


def parse_arguments():
    """
    Parse the command line arguments
    """

    def validate_subject_id(value):
        value = int(value)
        if value < 1 or value > 109:
            raise argparse.ArgumentTypeError("Subject ID must be between 1 and 109")
        return value

    def validate_recording_id(value):
        value = int(value)
        if value < 1 or value > 14:
            raise argparse.ArgumentTypeError("Recording ID must be between 1 and 14")
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
        help="Enter the recording ID (between 1 and 14)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("----------------- 🧠 EEG Data Analysis 🧠 -----------------")
    # parse cmd arguments
    args = parse_arguments()
    file_path = create_file_path(args.subject, args.recording)
    if file_path:
        print("Loading data from: ", file_path)

    mode_map = {
        "plot": plot,
        "train": train,
        "predict": predict,
    }

    mode_function = mode_map.get(args.mode)
    if mode_function is None:
        print("Invalid mode. Please choose from: plot, train, predict")

    mode_function(file_path)
   
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


def plot_mode(file_path: str):
    # Load the data
    data_loader = EEGDataLoader()
    eeg_data: EEGData = data_loader.load_data(file_path=file_path)
    data_loader.describe_eeg_data()

    # Visualize the data
    eeg_vizualizer = EEGDataVizualizer()
    eeg_vizualizer.plot_eeg(eeg_data.raw)

    # Preprocess the data
    eeg_preprocessor = EEGPreprocessor()
    # psd = eeg_preprocessor.compute_psd(raw=eeg_data.raw)
    filtered_raw = eeg_preprocessor.filter_data(raw=eeg_data.raw)

    # Visualize the data
    eeg_vizualizer.plot_eeg(filtered_raw)


def train_mode(file_path: str):
    if file_path:
        file_paths = [file_path]
    else:
        file_paths = glob.glob(os.path.join(DATA_DIR, "S*", "*.edf"))
    all_features = []
    all_labels = []

    data_loader = EEGDataLoader()
    eeg_preprocessor = EEGPreprocessor()
    for file_path in file_paths:
        print("Loading data from: ", file_path)
        eeg_data: EEGData = data_loader.load_data(file_path=file_path)
        filtered_raw = eeg_preprocessor.preprocess(raw=eeg_data.raw)
        epochs = eeg_preprocessor.epoch_data(filtered_raw, eeg_data.events)
        epochs.drop_bad()
        #epochs.plot_drop_log()
        print(f"Remaining epochs: {len(epochs)}")
        print(f"Bad channels: {filtered_raw.info['bads']}")
        #        epochs.plot(n_epochs=10, title="Retained Epochs", block=True)

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
            #("preprocessor", EEGPreprocessor()),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )
    scores = cross_val_score(pipeline, all_features, all_labels, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", np.mean(scores))


def predict_mode():
    pass


def create_file_path(subject_id: int, recording_id: int):
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
    print("args,", args)

    file_path = create_file_path(args.subject, args.recording)
    print("Loading data from: ", file_path)
    mode_map = {
        "plot": plot_mode,
        "train": train_mode,
        "predict": predict_mode,
    }

    mode_function = mode_map.get(args.mode)
    if mode_function is None:
        print("Invalid mode. Please choose from: plot, train, predict")

    mode_function(file_path)
    exit()
    # Load the data
    # file_path = input("Enter the path of the dataset: ")

    file_paths = glob.glob(os.path.join(DATA_DIR, "S*", "*.edf"))
    print(f"Found {len(file_paths)} files in the dataset.")
    exit()
    file_path = os.path.join("dataset", "S042", "S042R08.edf")
    print("Loading data from: ", file_path)

    data_loader = EEGDataLoader()
    eeg_data: EEGData = data_loader.load_data(file_path=file_path)
    data_loader.describe_eeg_data()

    # Visualize the data
    eeg_vizualizer = EEGDataVizualizer()
    eeg_vizualizer.plot_eeg(eeg_data.raw)

    # Preprocess the data
    eeg_preprocessor = EEGDataPreprocessing()
    psd = eeg_preprocessor.compute_psd(raw=eeg_data.raw)

    filtered_raw = eeg_preprocessor.filter_data(raw=eeg_data.raw)

    # Visualize the data
    eeg_vizualizer.plot_eeg(filtered_raw)

    # plt.show()

    good_channels = ["FC3", "FCZ", "FC4", "C3", "C1", "CZ", "C2", "C4"]
    bad_channels = [ch for ch in filtered_raw.ch_names if ch not in good_channels]
    filtered_raw.info["bads"] = bad_channels

    # Train the model
    picks = mne.pick_types(
        filtered_raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )
    epochs = mne.Epochs(
        filtered_raw, eeg_data.events, baseline=None, verbose=True, picks=picks
    )
    print(epochs)
    print(epochs.get_data().shape)
    print(epochs.get_data())
    X = epochs.get_data(copy=False)

    pca = UnsupervisedSpatialFilter(PCA(6), average=False)
    pca_data = pca.fit_transform(X)
    ev = mne.EvokedArray(
        np.mean(pca_data, axis=0),
        mne.create_info(6, epochs.info["sfreq"], ch_types="eeg"),
        tmin=-0.2,
    )
    ev.plot(show=False, window_title="PCA", time_unit="s")
    # plt.show()
    # Evaluate the model

    clf = LinearDiscriminantAnalysis()

    labels = epochs.events[:, -1]
    print("Shape of PCA data:", pca_data.shape)  # Should be (29, 6, 113)
    print("Shape of labels:", labels.shape)  # Should be (29,)
    if pca_data.shape[0] == len(labels):
        X = pca_data.reshape(pca_data.shape[0], -1)
        # Cross-validation
        scores = cross_val_score(clf, X, labels, cv=5)
        print("Cross-validation scores:", scores)
        print("Mean accuracy:", scores.mean())
    else:
        print("Mismatch between PCA data and labels after correction.")

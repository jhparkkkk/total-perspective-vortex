import os

from src.constants              import N_FFT, DEFAULT_MONTAGE, DATA_DIR
from src.EEGDataLoader          import EEGDataLoader
from src.EEGDataVizualizer      import EEGDataVizualizer
from src.EEGDataPreprocessing   import EEGDataPreprocessing
from src.data_models            import EEGData

import matplotlib.pyplot as plt

from mne.datasets import eegbci
from mne.channels import make_standard_montage

import mne

from sklearn.decomposition import PCA, FastICA

from mne.decoding import UnsupervisedSpatialFilter
import numpy as np

from mne.preprocessing import find_bad_channels_maxwell

import glob

if __name__ == "__main__":
    print("----------------- 🧠 EEG Data Analysis 🧠 -----------------")
    # Load the data
    #file_path = input("Enter the path of the dataset: ")

    file_paths = glob.glob(os.path.join(DATA_DIR, 'S*', '*.edf'))

    file_path = os.path.join('dataset', 'S032', 'S032R08.edf')
    print("Loading data from: ", file_path)
    
    data_loader = EEGDataLoader()
    eeg_data : EEGData = data_loader.load_data(file_path=file_path)
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
    
    #plt.show()

    good_channels = ["FC3", "FCZ", "FC4", "C3", "C1", "CZ", "C2", "C4"]
    bad_channels = [ch for ch in filtered_raw.ch_names if ch not in good_channels]
    filtered_raw.info["bads"] = bad_channels

    # Train the model
    picks = mne.pick_types(
        filtered_raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )
    epochs = mne.Epochs(filtered_raw, eeg_data.events, baseline=None, verbose=True, picks=picks)
    print(epochs)
    print(epochs.get_data().shape)
    print(epochs.get_data())
    X = epochs.get_data(copy=False)

    pca = UnsupervisedSpatialFilter(PCA(6), average=False)
    pca_data = pca.fit_transform(X)
    ev = mne.EvokedArray(
        np.mean(pca_data, axis=0),
        mne.create_info(6, epochs.info["sfreq"], ch_types="eeg"),
        tmin=-1,
    )
    ev.plot(show=False, window_title="PCA", time_unit="s")
    plt.show()
    # Evaluate the model

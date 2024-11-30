import numpy as np
import mne
from .data_models import EEGData
import matplotlib.pyplot as plt

from .constants import (
    GOOD_CHANNELS,
    N_FFT,
    DEFAULT_MONTAGE,
    LOW_FREQUENCY,
    HIGH_FREQUENCY,
    REF_CHANNELS,
    N_COMPONENTS,
)
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.decoding import UnsupervisedSpatialFilter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FastICA


class EEGPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess the EEG data
    """

    def __init__(
        self, l_freq=LOW_FREQUENCY, h_freq=HIGH_FREQUENCY, n_components=N_COMPONENTS
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.n_components = n_components

        pass

    def preprocess(self, raw):
        """
        Preprocess the raw data
        """
        raw = self.set_montage(raw)
        raw = self.filter_data(raw)
        raw = self.mark_bad_channels(raw)
        return raw

    def filter_data(self, raw):
        """
        Filter the data by keeping alpha and beta frequencies
        Alpha 8-13 Hz rest state, no movement
        Beta 13-30 Hz movement and cognitive tasks
        """
        raw.set_eeg_reference(ref_channels=REF_CHANNELS)
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
        return raw

    def mark_bad_channels(self, raw):
        """
        Remove the bad channels
        """
        good_channels = GOOD_CHANNELS
        bad_channels = [ch for ch in raw.ch_names if ch not in good_channels]
        raw.info["bads"] = bad_channels

        return raw

    def set_montage(self, raw):
        """
        Set the montage
        """
        eegbci.standardize(raw)
        montage = make_standard_montage(DEFAULT_MONTAGE)
        raw.set_montage(montage)
        return raw

    def compute_psd(self, raw):
        """
        Compute the power spectral density
        """
        psd = raw.compute_psd(method="multitaper", picks="eeg", fmin=1, fmax=80)
        psd.plot()
        return psd

    def extract_features_and_labels(self, raw) -> np.ndarray:
        """
        Extract features and labels from the EEG data.

        PSD represents how power of a signal varies across frequency.
        PSD extracts interpretable features from the EEG data.
        units: measured in microvolts squared per hertz (µV²/Hz)

        paremeters:
            - raw (mne.io.Raw): the raw EEG data
        returns:
            - features (np.ndarray): the features as the mean of the PSD
            - labels (np.ndarray): the labels for each epoch
        """
        try:
            epochs = self.epoch_data(raw)

            psd_data = epochs.compute_psd(
                fmin=self.l_freq, fmax=self.h_freq, method="multitaper"
            )

            psds, freqs = psd_data.get_data(return_freqs=True)

            # psds.shape = (n_epochs, n_channels, n_freqs)
            features = psds.mean(axis=2)

            labels = epochs.events[:, -1]
            return features, labels

        except Exception as e:
            print("Error extracting features:", e)
            raise e

    def epoch_data(self, raw):
        """
        Epoch the data
        """
        events, _ = mne.events_from_annotations(raw)

        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )

        event_id = dict(T1=1, T2=2)

        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=event_id,
            tmin=-1.0,
            tmax=2.0,
            picks=picks,
            baseline=(None),
            preload=True,
            verbose=True,
        )
        epochs.drop_bad()
        return epochs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass

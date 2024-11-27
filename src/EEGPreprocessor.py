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
        print("channel names in raw:", raw.ch_names)
        raw = self.filter_data(raw)
        # raw = self.mark_bad_channels(raw)
        print("channel names in raw:", len(raw.ch_names))

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
        # standardize the channel names
        eegbci.standardize(raw)
        montage = make_standard_montage(DEFAULT_MONTAGE)
        raw.set_montage(montage)

        # mark bad channels
        good_channels = GOOD_CHANNELS
        bad_channels = [ch for ch in raw.ch_names if ch not in good_channels]
        raw.info["bads"] = bad_channels
        return raw

    def compute_psd(self, raw):
        """
        Compute the power spectral density
        """
        eegbci.standardize(raw)
        montage = make_standard_montage(DEFAULT_MONTAGE)
        raw.set_montage(montage)
        psd = raw.compute_psd(method='multitaper', picks='eeg', fmin=1, fmax=80, n_fft=N_FFT)
        psd.plot()
        plt.show()
        return psd

    def extract_features(self, epochs: mne.epochs.Epochs) -> np.ndarray:
        """
        Extract features  from the epochs by computing the power spectral density (PSD)

        PSD represents how power of a signal varies across frequency.
        PSD extracts interpretable features from the EEG data.
        units: measured in microvolts squared per hertz (µV²/Hz)
        paremeters:
            - epochs: the preprocessed epochs data
        returns:
            - features: the extracted features as the mean of the PSD
        """
        try:
            psd_data = epochs.compute_psd(
                fmin=self.l_freq, fmax=self.h_freq, n_fft=N_FFT, method="welch"
            )

            psds, freqs = psd_data.get_data(return_freqs=True)

            features = psds.mean(axis=2)

            print("features extracted: shape", features.shape)

            return features
        except Exception as e:
            print("Error extracting features:", e)
            raise e

    def epoch_data(self, raw, events, run):
        """
        Epoch the data
        """
        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        print("Selected picks:", picks)
        if run in [1, 2]:
            reject = None 
            flat = None
        else: 
            reject = dict(eeg=150e-6)
            flat = dict(eeg=1e-6) 
        selected_channel_names = [raw.ch_names[i] for i in picks]
        print("Selected channel names:", selected_channel_names)
        epochs = mne.Epochs(
            raw,
            events,
            tmin=-1.0,
            tmax=4.0,
            picks=picks,
            baseline=(None, 0),
            preload=True,
            verbose=True,
            reject=reject,
            flat=flat,
        )
        epochs.drop_bad()
        return epochs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass

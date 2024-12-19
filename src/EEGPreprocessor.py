import numpy as np
import mne
from .data_models import EEGData
import matplotlib.pyplot as plt

from .constants import (
    GOOD_CHANNELS,
    REF_CHANNELS,
    DEFAULT_MONTAGE,
    LOW_FREQUENCY,
    HIGH_FREQUENCY,
    T_MIN,
    T_MAX,
)
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from sklearn.preprocessing import StandardScaler


class EEGPreprocessor:

    def __init__(self, l_freq=LOW_FREQUENCY, h_freq=HIGH_FREQUENCY):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.scaler = StandardScaler()

    def preprocess(self, raw: mne.io.edf.edf.RawEDF) -> mne.io.edf.edf.RawEDF:
        preprocessing_steps = [
            self.set_montage,
            self.filter_data,
            self._mark_bad_channels,
            self.normalize_data,
        ]

        for step in preprocessing_steps:
            raw = step(raw)
        return raw

    def filter_data(self, raw: mne.io.edf.edf.RawEDF) -> mne.io.edf.edf.RawEDF:
        return raw.filter(
            LOW_FREQUENCY,
            HIGH_FREQUENCY,
            fir_design="firwin",
            skip_by_annotation="edge",
            verbose=False,
        )

    def _mark_bad_channels(self, raw: mne.io.edf.edf.RawEDF) -> mne.io.edf.edf.RawEDF:
        bad_channels = [ch for ch in raw.ch_names if ch not in GOOD_CHANNELS]
        raw.info["bads"] = bad_channels
        return raw

    def set_montage(self, raw):
        eegbci.standardize(raw)
        montage = make_standard_montage(DEFAULT_MONTAGE)
        raw.set_montage(montage)
        raw.set_eeg_reference(ref_channels=REF_CHANNELS, verbose=False)
        return raw

    def normalize_data(self, raw):
        eeg_data = raw.get_data(picks="eeg")
        normalized_data = self.scaler.fit_transform(eeg_data.T).T
        raw._data[: len(normalized_data)] = normalized_data
        return raw

    def compute_psd(self, epochs):
        psd_data = epochs.compute_psd(
                fmin=LOW_FREQUENCY,
                fmax=HIGH_FREQUENCY,
                method="multitaper",
                verbose=False,
            )

        psds, _ = psd_data.get_data(return_freqs=True)
        return psds
    
    def epoch_data(self, raw):
        events, _ = mne.events_from_annotations(raw, verbose=False)

        eeg_channel_idx = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=dict(T1=2, T2=3),
            tmin=T_MIN,
            tmax=T_MAX,
            proj=False,
            picks=eeg_channel_idx,
            baseline=None,
            preload=True,
            verbose=False,
        )
        epochs.drop_bad()
        return epochs

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
            psd_data = self.compute_psd(epochs)
            
            # psds.shape = (n_epochs, n_channels, n_freqs)
            features = psd_data
            labels = epochs.events[:, -1] - 2
            return features, labels

        except Exception as e:
            print("Error extracting features:", e)
            raise e

    

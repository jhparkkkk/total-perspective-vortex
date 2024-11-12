import numpy as np
import mne
from .data_models import EEGData
import matplotlib.pyplot as plt

from .constants import N_FFT, DEFAULT_MONTAGE, LOW_FREQUENCY, HIGH_FREQUENCY, REF_CHANNELS
from mne.datasets import eegbci
from mne.channels import make_standard_montage

class EEGDataPreprocessing:
    def __init__(self):
        pass

    def filter_data(self, raw):
        """
        Filter the data by keeping alpha and beta frequencies
        Alpha 8-13 Hz rest state, no movement
        Beta 13-30 Hz movement and cognitive tasks
        """
        raw.set_eeg_reference(ref_channels=REF_CHANNELS)
        raw.filter(l_freq=LOW_FREQUENCY, h_freq=HIGH_FREQUENCY)
        return raw
        pass
    
    def compute_psd(self, raw):
        """
        Compute the power spectral density
        """
        eegbci.standardize(raw)  
        montage = make_standard_montage(DEFAULT_MONTAGE)
        raw.set_montage(montage)
        psd = raw.compute_psd(fmin=1, fmax=80, n_fft=N_FFT)
        psd.plot()
        plt.show()
        return psd
        pass

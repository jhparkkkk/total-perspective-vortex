import matplotlib.pyplot as plt
import numpy as np
from .EEGPreprocessor import EEGPreprocessor

class EEGDataVizualizer:
    def __init__(self):
        """
        Initialize the EEGEventVizualizer with EEGData
        """

    def plot_eeg(self, raw, title=None):
        """
        Plot the EEG data
        """
        fig = raw.plot(show=True)

        if title:
            fig.canvas.manager.set_window_title(title)
        plt.show()

    def plot_psd(self, raw):
        """
        Plot the power spectral density of the EEG data
        """
        fmax= raw.info["sfreq"] / 2 * 0.9
        fig = raw.compute_psd(method="multitaper", verbose=False, fmin=1, fmax=fmax)
        fig.plot()

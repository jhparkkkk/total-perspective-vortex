import matplotlib.pyplot as plt
import numpy as np
from .data_models import EEGData


class EEGDataVizualizer:
    def __init__(self):
        """
        Initialize the EEGEventVizualizer with EEGData
        """

    def plot_eeg(self, raw):
        """
        Plot the EEG data
        """

        raw.plot()
        plt.show()

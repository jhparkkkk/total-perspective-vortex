import matplotlib.pyplot as plt
import numpy as np
from .data_models import EEGData

class EEGDataVizualizer:
    def __init__(self, eeg_data: EEGData):
        """
        Initialize the EEGEventVizualizer with EEGData
        """
        self.eeg_data = eeg_data
    
    def plot_eeg(self):
        """
        Plot the EEG data
        """

        if self.eeg_data.events is None:
            raise ValueError("Events not loaded yet.")

        self.eeg_data.raw_eeg_data.plot()
        plt.show()
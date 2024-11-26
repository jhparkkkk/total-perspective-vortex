import matplotlib.pyplot as plt
import numpy as np
from .data_models import EEGData


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

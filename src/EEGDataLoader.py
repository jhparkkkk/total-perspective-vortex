import mne


class EEGdataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        pass

    def load_data(self):
        self.raw_data = mne.io.read_raw_fif(self.file_path, preload=True)

import mne


class EEGDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        pass

    def load_data(self):
        self.raw_data = mne.io.read_raw_edf(self.file_path, preload=True)
        
        print("Data loaded successfully")
        #print(self.raw_data.info)
        
        print("Data shape: ", self.raw_data.get_data().shape)
        print("Data type: ", type(self.raw_data))
        
        print("Data channels: ", self.raw_data.ch_names)
        print("Data sampling frequency: ", self.raw_data.info['sfreq'])
        print("Data duration: ", self.raw_data.times[-1])
        print("Data events: ", self.raw_data.annotations)

        print("Data events: ", self.raw_data.annotations.description)
        print("Data info: ", self.raw_data.info)
        
        events, event_id = mne.events_from_annotations(self.raw_data)
        print("Events: ", events)


        return self.raw_data

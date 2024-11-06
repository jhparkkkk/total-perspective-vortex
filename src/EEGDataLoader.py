import mne
import mne.io.edf.edf
import shutil
from typing import Optional
from dataclasses import dataclass

RawEDF = mne.io.edf.edf.RawEDF

TERMINAL_WIDTH = shutil.get_terminal_size().columns


@dataclass
class ChannelInfo:
    channels: 'list[str]'
    highpass_Hz: float
    lowpass_Hz: float

@dataclass
class EEGData:
    subject_id: int
    run_id: int
    timestamp: Optional[str] = None
    duration_seconds: Optional[int] = None
    sampling_frequency: Optional[int] = None
    events: 'numpy.ndarray' = None

event_lookup = {
    (1, 2): {1: "rest"},
    (3, 4, 7, 8, 11, 12): {1: "rest", 2: "left fist", 3: "right fist"},
    (5, 6, 9, 10, 13, 14): {1: "rest", 2: "both fists", 3: "both feet"}
}

class EEGDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data : RawEDF = None

    def load_data(self):
        """
        Load the EEG data from the .edf file path
        """
        self.raw_data = mne.io.read_raw_edf(self.file_path, preload=True)
        self.ChannelInfo = self._extract_channel_info()
        self.EEGData = self._extract_eeg_data()
        
    
    def describe_channel_info(self):
        """
        Print the channel information
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded yet.")
        
        header = " Channel Information "
        print("\n" + header.center(TERMINAL_WIDTH, "-"))
        print(f"Highpass:{self.ChannelInfo.highpass_Hz} Hz")
        print(f"Lowpass: {self.ChannelInfo.lowpass_Hz} Hz")

        for i in range(0, len(self.ChannelInfo.channels), 4):
            print(self.ChannelInfo.channels[i:i+4])

    def describe_eeg_data(self):

        if self.raw_data is None:
            raise ValueError("Raw data not loaded yet.")
        
        header = " EEG Data Information "
        print("\n" + header.center(TERMINAL_WIDTH, "-"))
        print("Subject ID: ", self.EEGData.subject_id)
        print("Run ID: ", self.EEGData.run_id)
        print("Timestamp: ", self.EEGData.timestamp)
        print("Duration: ", self.EEGData.duration_seconds, " seconds")

        print("\nEvents:")
        print(f"{'Onset (sample)':<15} {'Timestamp (s)':<15} {'Previous Code':<15} {'Event Type':<15} {'Decoded Event Type':<30}")
        print("-" * TERMINAL_WIDTH)
        task = self._decode_event(self.EEGData.run_id)
        for event in self.EEGData.events:
            onset, prev_code, event_type = event
            timestamp = onset / self.EEGData.sampling_frequency
            print(f"{onset:<15} {timestamp:<15} {prev_code:<15} {event_type:<15} {task[int(event_type)]:<30}")

    def _extract_events(self) -> 'numpy.ndarray':
        """
        Extract the events from the raw .edf data
        Each event is represented by a row with 3 integers: [start, previous, event_type]
        event_type are mapped like this:
          - T0 = 1: resting state, baseline
          - T1 = 2: first task 
          - T2 = 3: second task
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded yet.")
        events, _ = mne.events_from_annotations(self.raw_data)
        return events

    def _extract_eeg_data(self) -> EEGData:
        """
        Extract the EEG data from the raw .edf data
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded yet.")
        
        subject_id = self.file_path.split('/')[-2][1:]
        run_id= self.file_path.split('/')[-1].split('.')[0][-2:]

        return EEGData(
            subject_id=int(subject_id),
            run_id=int(run_id),
            timestamp=self.raw_data.info['meas_date'],
            duration_seconds=self.raw_data.times[-1],
            sampling_frequency=self.raw_data.info['sfreq'],
            events=self._extract_events()
        )    
    
    def _extract_channel_info(self) -> ChannelInfo:
        """
        Extract the channel information from the raw .edf data
        - channels: electrodes capturing electrical signals from specific brain areas
        - highpass: high frequency noise filter (Hz Max)
        - lowpass: low frequency noise filter (Hz Min)
        """
        return ChannelInfo(
            channels=self.raw_data.ch_names,
            highpass_Hz=self.raw_data.info['highpass'],
            lowpass_Hz=self.raw_data.info['lowpass']
        )

    def _decode_event(self, run_id: int) -> str:
        """
        Decode the event id into a human readable format
        """
        for key, value in event_lookup.items():
            if run_id in key:
                return value
        return("Unknown Task")
        
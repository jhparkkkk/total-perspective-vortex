import mne
from mne.io.edf.edf import RawEDF
from .data_models import ChannelInfo, EEGData
from .constants import TERMINAL_WIDTH, EVENT_LOOKUP

from .utils import create_file_path


class EEGDataLoader:
    def __init__(self):
        self.raw_data: RawEDF = None

    def load_data(self, file_path: str) -> RawEDF:
        """
        Load the EEG data from the .edf file path
        """
        try:
            return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            raise ValueError("Subject ID not provided.")

    def load_raws(self, subject: int, runs: list[int]) -> RawEDF:
        if subject is None:
            raise ValueError("Subject ID not provided.")
        raws = []
        for run in runs:
            raw = self.load_data(create_file_path(subject, run))
            raws.append(raw)
        raws = mne.concatenate_raws(raws)
        return raws

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
            print(self.ChannelInfo.channels[i : i + 4])

    def describe_eeg_data(self, raw, file_path):

        # if raw is None:
        #    raise ValueError("Raw data not loaded yet.")

        self.ChannelInfo = self._extract_channel_info(raw)
        self.EEGData = self._extract_eeg_data(raw, file_path)

        header = " EEG Data Information "
        print("\n" + header.center(TERMINAL_WIDTH, "-"))
        print("Subject ID: ", self.EEGData.subject_id)
        print("Run ID: ", self.EEGData.run_id)
        print("Date: ", self.EEGData.date)
        print("Duration: ", self.EEGData.duration_seconds, " seconds")

        print(f"\n{len(self.EEGData.events)} events found.\n")
        print(
            f"{'Onset (sample)':<15} {'Timestamp (s)':<15} {'Previous Code':<15} {'Event Type':<15} {'Decoded Event Type':<30}"
        )
        print("-" * TERMINAL_WIDTH)
        task = self._decode_event(self.EEGData.run_id)
        for event in self.EEGData.events:
            onset, prev_code, event_type = event
            timestamp = onset / self.EEGData.sampling_frequency
            print(
                f"{onset:<15} {timestamp:<15} {prev_code:<15} {event_type:<15} {task[int(event_type)]:<30}"
            )

    def extract_events(self, raw) -> "numpy.ndarray":
        """
        Extract the events from the raw .edf data
        Each event is represented by a row with 3 integers: [start, previous, event_type]
        event_type are mapped like this:
          - T1 = 2: first task
          - T2 = 3: second task
        """
        if raw is None:
            raise ValueError("Raw data not loaded yet.")
        events, _ = mne.events_from_annotations(raw, verbose=False)
        return events

    def _extract_eeg_data(self, raw, file_path) -> EEGData:
        """
        Extract the EEG data from the raw .edf data
        """
        if raw is None:
            raise ValueError("Raw data not loaded yet.")

        self.subject_id = file_path.split("/")[-2][1:]
        self.run_id = file_path.split("/")[-1].split(".")[0][-2:]

        return EEGData(
            subject_id=int(self.subject_id),
            run_id=int(self.run_id),
            date=raw.info["meas_date"],
            duration_seconds=raw.times[-1],
            sampling_frequency=raw.info["sfreq"],
            channel_info=self.ChannelInfo,
            events=self.extract_events(raw),
            raw=raw,
            eeg_data=raw.get_data(),
            timestamps=raw.times,
        )

    def _extract_channel_info(self, raw) -> ChannelInfo:
        """
        Extract the channel information from the raw .edf data
        - channels: electrodes capturing electrical signals from specific brain areas
        - highpass: high frequency noise filter (Hz Max)
        - lowpass: low frequency noise filter (Hz Min)
        """
        return ChannelInfo(
            channels=raw.ch_names,
            highpass_Hz=raw.info["highpass"],
            lowpass_Hz=raw.info["lowpass"],
        )

    def _decode_event(self, run_id: int) -> str:
        """
        Decode the event type using event_lookup into a human readable format
        """
        for key, value in EVENT_LOOKUP.items():
            if run_id in key:
                return value
        return "Unknown Task"

    def extract_run_id(self, file_path: str) -> int:
        """
        Extract the run_id from the file path
        """
        return int(file_path.split("/")[-1].split(".")[0][-2:])

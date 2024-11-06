from dataclasses import dataclass
from typing import Optional

@dataclass
class ChannelInfo:
    channels: 'list[str]'
    highpass_Hz: float
    lowpass_Hz: float

@dataclass
class EEGData:
    subject_id: int
    run_id: int
    date: Optional[str] = None
    duration_seconds: Optional[int] = None
    sampling_frequency: Optional[int] = None
    channel_info: Optional[ChannelInfo] = None
    events: 'numpy.ndarray' = None
    raw_eeg_data: 'numpy.ndarray' = None
    eeg_data: 'numpy.ndarray' = None
    timestamps: 'numpy.ndarray' = None

from dataclasses import dataclass
from typing import Optional
from numpy import ndarray

@dataclass
class ChannelInfo:
    channels: "list[str]"
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
    events: ndarray = None
    raw: ndarray = None
    eeg_data: ndarray = None
    timestamps: ndarray = None

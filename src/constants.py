DATA_DIR = "dataset"
SUBJECT = "S001"
RECORDING = "S001R04"

GOOD_CHANNELS = ["FC3", "FCz", "FC4", "C3", "C1", "Cz", "C2", "C4"]

LOW_FREQUENCY = 8
HIGH_FREQUENCY = 30
REF_CHANNELS = "average"
DEFAULT_MONTAGE = "standard_1020"
N_COMPONENTS = 8
N_FFT = 800

import shutil

TERMINAL_WIDTH = shutil.get_terminal_size().columns

DATA_DIR = "dataset"
SUBJECT = "S001"
RECORDING = "S001R04"

BAD_SUBJECTS = [38, 88, 89, 92, 93, 94, 100, 104, 106]

RUNS = [
    [3, 7, 11],
    [4, 8, 12],
    [5, 9, 13],
    [6, 10, 14],
    #[3, 7, 11, 4, 8, 12],
    #[5, 9, 13, 6, 10, 14],
]

GOOD_CHANNELS = ["FC3", "FCz", "FC4", "C3", "C1", "Cz", "C2", "C4"]
# ROIC3/ROIC4 (Region of Interest autour de C3 et C4)
#GOOD_CHANNELS = ["FC3", "C5", "C3", "C1", "CP3", "FC4", "C2", "C4", "C6", "CP4"]
# ROICp3/ROICp4 (Region of Interest autour de CP3 et CP4)
# GOOD_CHANNELS = [ "C3", "CP5", "CP3", "CP1", "P3", "CP2", "C4", "CP6", "P4"]


LOW_FREQUENCY = 7
HIGH_FREQUENCY = 30
REF_CHANNELS = "average"
DEFAULT_MONTAGE = "standard_1020"
N_COMPONENTS = 8
N_FFT = 600

import shutil

TERMINAL_WIDTH = shutil.get_terminal_size().columns

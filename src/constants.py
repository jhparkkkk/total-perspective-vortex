DATA_DIR = "dataset"

# BAD_SUBJECTS = [38, 88, 89, 92, 93, 94, 100, 104, 106]
# BAD_SUBJECTS = [14, 34, 37, 41, 51, 64, 69, 72, 73, 74, 76, 88, 92, 100, 102, 104, 109]
BAD_SUBJECTS = []

RUNS = [
    [3, 7, 11],
    [4, 8, 12],
    [5, 9, 13],
    [6, 10, 14],
    [3, 7, 11, 4, 8, 12],
    [5, 9, 13, 6, 10, 14],
]

EVENT_LOOKUP = {
    (1, 2): {1: "Rest"},
    (3, 7, 11): {
        1: "Rest",
        2: "Real Left Hand Movement",
        3: "Real Right Hand Movement",
    },
    (4, 8, 12): {
        1: "Rest",
        2: "Imagery Left Hand Movement",
        3: "Imagery Right Hand Movement",
    },
    (5, 9, 13): {1: "Rest", 2: "Real Fists Movement", 3: "Real Feet Movement"},
    (6, 10, 14): {1: "Rest", 2: "Imagery Fists Movement", 3: "Imagery Feet Movement"},
}

# ROIC3/ROIC4
GOOD_CHANNELS = ["FC3", "C5", "C3", "C1", "CP3", "FC4", "C2", "C4", "C6", "CP4"]
# ROICp3/ROICp4
# GOOD_CHANNELS = [ "C3", "CP5", "CP3", "CP1", "P3", "CP2", "C4", "CP6", "P4"]


LOW_FREQUENCY = 7
HIGH_FREQUENCY = 40
T_MIN = -1.0
T_MAX = 4.0
REF_CHANNELS = "average"
DEFAULT_MONTAGE = "standard_1020"

N_COMPONENTS = 10
TEST_SIZE = 0.2
N_SPLITS = 5

import shutil

TERMINAL_WIDTH = shutil.get_terminal_size().columns

import os
import argparse


from .constants import TERMINAL_WIDTH, DATA_DIR


def parse_arguments():
    """
    Parse the command line arguments
    """

    def validate_subject_id(arg: str) -> int:
        """
        Validate a single subject ID.
        """
        value = int(arg)
        if value < 1 or value > 109:
            raise argparse.ArgumentTypeError("Subject ID must be between 1 and 109")
        return value

    def validate_recording_id(arg: str) -> int:
        """
        Validate a single recording ID.
        """
        value = int(arg)
        if value < 1 or value > 14:
            raise argparse.ArgumentTypeError(
                "Each Recording ID must be between 1 and 14"
            )
        return value

    parser = argparse.ArgumentParser(description="EEG Data Analysis")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["plot", "train", "predict", "experiment"],
        help="Mode of operation: visualize EEG, train model, make predictions, experiment with all subjects",
        default="experiment"
    )
    parser.add_argument(
        "--subject",
        type=validate_subject_id,
        nargs="+",
        help="Enter the subject ID (between 1 and 109)",
        default=list(range(1, 110)),
    )
    parser.add_argument(
        "--recording",
        type=validate_recording_id,
        nargs="+",
        help="Enter a space-separated list of recording IDs (between 3 and 14)",
        default=list(range(3, 15)),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset directory",
        required=False,
        default=None,
    )
    return parser.parse_args()

def set_dataset_path(dataset_path):
    global DATA_DIR
    if dataset_path is None:
        dataset_path = input("Enter the path to the dataset: ").strip()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The specified dataset path does not exist: {dataset_path}")
    DATA_DIR = dataset_path


def create_file_path(subject_id: int, run_id: int) -> str:
    """
    Create a list of file paths based on subject ID and recording IDs.
    param:
        - subject_id: the subject ID
        - recording_ids: the list of recording IDs
    return:
        - file_paths: the list of file paths
    """
    return os.path.join(
        DATA_DIR, f"S{subject_id:03}", f"S{subject_id:03}R{run_id:02}.edf"
    )


def display_header(mode: str, subjects: list[int], runs: list[int]) -> None:
    """
    Display the header information
    """
    header = " Brain-Computer Interface (BCI) using EEG Data Analysis "
    print("\n" + header.center(TERMINAL_WIDTH, "-"))

    label_width = 12
    print(f"{'Mode:'.ljust(label_width)}{mode.upper()}")
    print(f"{'Subject(s):'.ljust(label_width)}{', '.join(map(str, subjects))}")
    print(f"{'Run(s):'.ljust(label_width)}{', '.join(map(str, runs))}")


def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] An error occurred in {func.__name__}: {e}")
            return None

    return wrapper


def log(message: str, active: bool = True):
    if active:
        print(message)

import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.CSP import CSP as myCSP
from src.EEGDataLoader import EEGDataLoader
from src.EEGPreprocessor import EEGPreprocessor
from src.EEGDataVizualizer import EEGDataVizualizer
from src.utils import (
    parse_arguments,
    set_dataset_path,
    create_file_path,
    display_header,
    handle_exceptions,
    log,
)
from src.constants import BAD_SUBJECTS, RUNS, TEST_SIZE, N_COMPONENTS, N_SPLITS

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


def plot(subject_id: list[int], run_id: list[int]):
    """
    Load the EEG data and plot it raw and after preprocessing (filtering)
    """
    if len(subject_id) != 1 or len(run_id) != 1:
        raise ValueError("Only a single file path is allowed for plotting")

    file_path = create_file_path(subject_id[0], run_id[0])

    # Load the data
    loader = EEGDataLoader()
    raw = loader.load_data(file_path=file_path)
    loader.describe_eeg_data(raw, file_path)

    # Visualize the data before preprocessing
    visualizer = EEGDataVizualizer()
    visualizer.plot_eeg(
        raw, title=f"Raw EEG Data for Subject {loader.subject_id} Run {loader.run_id}"
    )

    # Preprocess the data
    preprocessor = EEGPreprocessor()
    raw = preprocessor.set_montage(raw)
    filtered_raw = preprocessor.filter_data(raw=raw)

    # Visualize the power spectral density
    visualizer.plot_psd(raw=filtered_raw)

    # Visualize the data after preprocessing
    visualizer.plot_eeg(
        raw,
        title=f"Filtered EEG Data for Subject {loader.subject_id} Run {loader.run_id}",
    )


@handle_exceptions
def train(subject_list: list[int], run_list: list[int], log_active: bool = True):
    """
    Train a model for given subjects and runs.
    """
    logger = lambda msg: log(msg, log_active)

    # Load and preprocess data
    loader = EEGDataLoader()
    preprocessor = EEGPreprocessor()
    X, y = [], []

    for subject in subject_list:
        logger(f"Loading and preprocessing data for Subject {subject}...")
        raw = loader.load_raws(subject, run_list)
        preprocessed_raw = preprocessor.preprocess(raw)
        features, labels = preprocessor.extract_features_and_labels(preprocessed_raw)
        X.append(features)
        y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    logger(f"Data loaded: {X.shape[0]} epochs, {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    pipeline = Pipeline(
        [
            ("CSP", myCSP(n_components=N_COMPONENTS)),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )

    cv = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
    logger(f"Cross-validation mean accuracy: {np.mean(cv_scores) * 100:.2f}%")

    pipeline.fit(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    logger(f"Test accuracy: {test_accuracy * 100:.2f}%")

    with open("model.pkl", "wb") as model_file:
        pickle.dump(pipeline, model_file)

    logger("Model saved: model.pkl")

    return np.mean(cv_scores), test_accuracy


@handle_exceptions
def experiment(subject_list: list[int], run_list: list[int]):
    """
    Execute training for each subject and compute the mean score
    over each subject, by type of experiment runs.
    """
    subject_list = [s for s in range(1, 110) if s not in BAD_SUBJECTS]

    run_scores = {i: [] for i in range(len(RUNS))}

    print("Starting experiments...")

    for run_id, runs in enumerate(RUNS):
        print(f"\nProcessing Experiment {run_id + 1}: Runs {runs}")
        cv_score, test_score = train(subject_list, runs, log_active=True)
        run_scores[run_id].append(test_score)

    mean_scores_per_run = []
    for run_id, scores in run_scores.items():
        mean_score = np.mean(scores)
        mean_scores_per_run.append(mean_score)
        print(f"Experiment {run_id + 1}: Mean Accuracy = {mean_score:.4f}")

    global_mean_accuracy = np.mean(mean_scores_per_run)
    print(f"\nGlobal mean accuracy across runs: {global_mean_accuracy:.4f}")

    if global_mean_accuracy >= 0.6:
        print("Global accuracy is greater than 60%.")
    else:
        print("Global accuracy is less than 60%.")


def _run_experiment(args):
    """
    Wrapper for training a single experiment.
    """
    runs, subject_list = args
    return train(subject_list, runs, log_active=False)


def _experiment(subject_list: list[int], run_list: list[int]):
    """
    Execute training for each subject and compute the mean score
    over each subject, by type of experiment runs.
    """
    subject_list = [s for s in range(1, 110) if s not in BAD_SUBJECTS]

    run_scores = {i: [] for i in range(len(RUNS))}

    tasks = [(runs, subject_list) for runs in RUNS]

    print("Starting experiments...")
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(run_experiment, tasks),
                total=len(tasks),
                desc="Experiments progress",
            )
        )

    for run_id, (cv_score, test_score) in enumerate(results):
        run_scores[run_id].append(test_score)

    mean_scores_per_run = []
    for run_id, scores in run_scores.items():
        mean_score = np.mean(scores)
        mean_scores_per_run.append(mean_score)
        print(f"Experiment {run_id + 1}: Mean Accuracy = {mean_score:.4f}")

    global_mean_accuracy = np.mean(mean_scores_per_run)
    print(f"\nGlobal mean accuracy accross runs {global_mean_accuracy:.4f}")

    if global_mean_accuracy >= 0.6:
        print("Global accuracy is greater than 60%.")
    else:
        print("Global accuracy is less than 60%.")


def stream_predictions(
    pipeline, features: np.ndarray, labels: np.ndarray, delay: float = 0.1
):
    """
    Simulate real-time predictions on streaming data.

    Args:
        - pipeline: The trained pipeline for making predictions
        - features: Array of features for each epoch
        - labels: Corresponding labels for the chunks
        - delay: Time delay (in seconds) between each epoch
    """

    header = f"{'Epoch':<10}{'Prediction':<15}{'Truth':<10}{'Equal?':<10}"
    print(header)

    for i, (feature, label) in enumerate(zip(features, labels)):
        prediction = pipeline.predict(feature.reshape(1, feature.shape[0], -1))
        result = f"{i:<10}{prediction[0]:<15}{label:<10}{'True' if prediction[0] == label else 'False':<10}"
        print(result)
        time.sleep(delay)


@handle_exceptions
def predict(subject_list: list[int], run_list: list[int]):
    loader = EEGDataLoader()
    preprocessor = EEGPreprocessor()

    for subject in subject_list:
        for run in run_list:
            file_path = create_file_path(subject, run)
            with open("model.pkl", "rb") as model_file:
                pipeline = pickle.load(model_file)
            print("Model successfully loaded from model.pkl")

            raw = loader.load_data(file_path=file_path)
            filtered_raw = preprocessor.preprocess(raw=raw)
            x, y = preprocessor.extract_features_and_labels(filtered_raw)

            y_pred = pipeline.predict(x)
            accuracy = accuracy_score(y, y_pred)
            stream_predictions(pipeline, x, y)

            print("\nPrediction Results:")
            print(f"  - Number of epochs: {len(x)}")
            print(f"  - True Labels: {y.tolist()}")
            print(f"  - Model'subject Predictions: {y_pred.tolist()}")
            print(f"  - Accuracy: {accuracy:.2%}")

@handle_exceptions
def main():
    args = parse_arguments()
    set_dataset_path(args.dataset_path)
    display_header(args.mode, args.subject, args.recording)

    mode_map = {
        "plot": plot,
        "train": train,
        "predict": predict,
        "experiment": experiment,
    }

    mode_function = mode_map.get(args.mode)
    if mode_function is None:
        print("Invalid mode. Please choose from: plot, train, predict, experiment")

    mode_function(args.subject, args.recording)

if __name__ == "__main__":
    main()

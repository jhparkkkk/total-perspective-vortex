import os
from src.EEGDataLoader import EEGDataLoader

from src.EEGDataVizualizer import EEGDataVizualizer
from src.data_models import EEGData

if __name__ == "__main__":
    print("----------------- 🧠 EEG Data Analysis 🧠 -----------------")
    # Load the data
    #file_path = input("Enter the path of the dataset: ")
    file_path = os.path.join('dataset', 'S001', 'S001R14.edf')
    print("Loading data from: ", file_path)
    data_loader = EEGDataLoader(file_path=file_path)
    data_loader.load_data()
    data_loader.describe_channel_info()
    data_loader.describe_eeg_data()


    event_vizualizer = EEGDataVizualizer(eeg_data=data_loader.EEGData)
    event_vizualizer.plot_eeg()
    # Preprocess the data
    # Train the model
    # Evaluate the model

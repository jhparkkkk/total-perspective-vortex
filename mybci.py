import os
from src.EEGDataLoader import EEGDataLoader



if __name__ == "__main__":
    print("----------------- 🧠 EEG Data Analysis 🧠 -----------------")
    # Load the data
    #file_path = input("Enter the path of the dataset: ")
    file_path = os.path.join('dataset', 'S001', 'S001R06.edf')
    print("Loading data from: ", file_path)
    data_loader = EEGDataLoader(file_path=file_path)
    data_loader.load_data()
    data_loader.describe_channel_info()
    data_loader.describe_eeg_data()


    
    # Preprocess the data
    # Train the model
    # Evaluate the model

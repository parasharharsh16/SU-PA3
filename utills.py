import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import warnings

warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths, self.labels = self.load_data()
        self.cut=64600 # take ~4 sec audio (64600 samples)
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        features = self.loads_audio(file_path)
        return features, label

    def load_data(self):
        file_paths = []
        labels = []
        for label, folder in enumerate(['Real', 'Fake']):
            folder_path = os.path.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                file_paths.append(os.path.join(folder_path, file))
                labels.append(label)
        return file_paths, labels
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x	
    
    def loads_audio(self, file_path):
        audio, sr = librosa.load(file_path,sr=16000)
        X_pad = self.pad(audio,self.cut)
        x_inp= torch.tensor(X_pad)
        return x_inp

# def calculate_eer():

# def evaluate():
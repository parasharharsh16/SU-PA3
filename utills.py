import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import warnings
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import wandb
import random

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
        label_encoded = []
        for label, folder in enumerate(['Real', 'Fake']):
            folder_path = os.path.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                file_paths.append(os.path.join(folder_path, file))
                if folder == 'Real':
                    labels.append(1)
                else:
                    labels.append(0)
        for label in labels:
            one_hot_label = torch.zeros(2)  # 2 classes: Real and Fake
            one_hot_label[label] = 1
            label_encoded.append(one_hot_label)

        return file_paths, label_encoded
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
    
class FORDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths, self.labels = self.load_data()
        self.cut=64600
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
        label_encoded = []
        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                file_paths.append(os.path.join(folder_path, file))
                if folder == 'real':
                    labels.append(1)
                else:
                    labels.append(0)
        for label in labels:
            one_hot_label = torch.zeros(2)  # 2 classes: Real and Fake
            one_hot_label[label] = 1
            label_encoded.append(one_hot_label)
        return file_paths, label_encoded
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

def calculate_auc(all_embeddings, all_labels,accuracy,eer,eer_threshold,model_name="pre-trained"):
    # Convert tensors to numpy arrays
    embeddings_np = all_embeddings.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    positive_class_index = 0 #DF
    scores_positive = embeddings_np[:, positive_class_index]
    labels_positive = labels_np[:, positive_class_index]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels_positive, scores_positive)

    # Calculate AUC
    auc_score = auc(fpr, tpr)

    # Calculate AUC
    auc_score = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.suptitle(f"Accuracy: {accuracy}, EER: {eer}, EER Threshold: {eer_threshold}")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{model_name}.png")
    plt.show()
    return auc



def calculate_eer(genuine_scores, impostor_scores):
    # genuine_scores = genuine_scores.cpu().numpy()
    # impostor_scores = impostor_scores.cpu().numpy()
    num_genuine_pairs = len(genuine_scores)
    num_impostor_pairs = len(impostor_scores)

    eer = 0.0
    min_difference = float('inf')
    eer_threshold = 0.0

    for threshold in np.arange(min(min(genuine_scores), min(impostor_scores)), max(max(genuine_scores), max(impostor_scores)), 0.001):
        far = sum(impostor_scores >= threshold) / num_impostor_pairs
        frr = sum(genuine_scores < threshold) / num_genuine_pairs
        difference = abs(far - frr)

        if difference < min_difference:
            min_difference = difference
            eer = (far + frr) / 2
            eer_threshold = threshold

    return eer, eer_threshold


def label_smoothing(targets, epsilon=0.1):
    n_classes = len(targets)#.size(1)
    smoothed_labels = (1.0 - epsilon) * targets + epsilon / n_classes
    return smoothed_labels

def finetune(train_loader,eval_loader, model,device,lr=0.001,epochs = 5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    wandb.init(
    # set the wandb project where this run will be logged
    project="SU-Programming-Assignment-3",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "xlsr2_300m",
    "dataset": "for-2sec",
    "epochs": epochs,
    }
    )
    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    # Freeze all layers except the last two
    # Freeze all layers except the last two
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    # Freeze all layers except the last two
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Identify the last two layers and unfreeze them
    last_two_layers = list(model.children())[-2:]
    for layer in last_two_layers:
        for param in layer.parameters():
            param.requires_grad = True       
    model.train()
    
    for i in range(epochs):
        running_loss = 0
        num_total = 0.0
        correct_predictions = 0
        pbar = tqdm(train_loader, desc=f"Epoch {i+1}")
        for batch_x, batch_y in pbar:
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            smooth_label = label_smoothing(batch_y)
            optimizer.zero_grad()
            
            # Forward pass
            batch_out = model(batch_x)
            
            # Compute loss
            batch_loss = criterion(batch_out, smooth_label)
            
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
            
            running_loss += (batch_loss.item() * batch_size)
            # Calculate accuracy
            idx = 0
            _, predicted_classes = torch.max(batch_out, 1)
            _, lable_class = torch.max(batch_y, 1)
            for pred, label in zip(predicted_classes, lable_class):
                correct_predictions += (pred == label).item()
                idx += 1
            
            num_total += len(predicted_classes)
            pbar.set_postfix({'loss': running_loss / num_total, 'accuracy': correct_predictions / num_total})
        loss = running_loss / num_total
        _,_, acc, _, _ = eval(eval_loader, model, device)
        wandb.log({"acc": acc, "loss": loss})

    wandb.finish()
    return model

def eval(eval_loader, model, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    num_total = 0.0
    correct_predictions = 0
    genuine = []
    impostor = []
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluation")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            
            _, predicted_classes = torch.max(batch_out, 1)
            _, lable_class = torch.max(batch_y, 1)
            #correct_predictions += (predicted_classes == lable_class).sum().item()
            idx = 0
            for pred, label in zip(predicted_classes, lable_class):
                correct_predictions += (pred == label).item()
                if label.item() == 1:
                    genuine.append((batch_out[idx])[0].detach().cpu().numpy())
                else:
                    impostor.append((batch_out[idx])[0].detach().cpu().numpy())
                idx += 1
            
            num_total += len(predicted_classes)
            # Append embeddings and labels
            all_embeddings.append(batch_out)
            all_labels.append(batch_y)
    accuracy = correct_predictions / num_total

    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeddings, all_labels, accuracy, genuine, impostor
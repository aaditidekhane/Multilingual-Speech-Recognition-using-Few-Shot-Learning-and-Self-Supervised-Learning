import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Data Preprocessing
def preprocess_audio(waveform, sr=16000, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=waveform.numpy().flatten(), sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

class MetaAudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        mel_spec = preprocess_audio(waveform, sr=sample_rate)
        return torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0), label

# Load Meta Dataset
meta_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='./', download=True, subset='training')
dataset = MetaAudioDataset(meta_dataset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Self-Supervised Learning Model
class SimpleSSLModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SimpleSSLModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * (input_dim // 4) * (input_dim // 4), embedding_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Matching Networks
class MatchingNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(MatchingNetwork, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, support_set, query_set):
        support_embeddings = self.embed(support_set)
        query_embeddings = self.embed(query_set)
        similarities = self.compute_similarities(support_embeddings, query_embeddings)
        return similarities

    def embed(self, x):
        return x
    
    def compute_similarities(self, support_embeddings, query_embeddings):
        support_embeddings = support_embeddings.detach().numpy()
        query_embeddings = query_embeddings.detach().numpy()
        similarities = cosine_similarity(query_embeddings, support_embeddings)
        return torch.tensor(similarities, dtype=torch.float32)

# Training and Evaluation
def train_ssl_model(ssl_model, dataloader, epochs=10, lr=0.001):
    optimizer = optim.Adam(ssl_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        ssl_model.train()
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            outputs = ssl_model(inputs)
            loss = criterion(outputs, inputs.view(outputs.size(0), -1))
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def evaluate_few_shot_model(ssl_model, matching_net, support_set, query_set, support_labels, query_labels):
    ssl_model.eval()
    support_embeddings = ssl_model(support_set)
    query_embeddings = ssl_model(query_set)
    
    similarities = matching_net(support_embeddings, query_embeddings)
    _, predicted_labels = torch.max(similarities, dim=1)
    
    accuracy = (predicted_labels == query_labels).float().mean()
    return accuracy.item()

# Example usage
input_dim = 128  # Mel spectrogram dimensions
embedding_dim = 256

ssl_model = SimpleSSLModel(input_dim=input_dim, embedding_dim=embedding_dim)
matching_net = MatchingNetwork(embedding_dim=embedding_dim)

train_ssl_model(ssl_model, dataloader, epochs=10)

# Create support and query sets
support_indices = random.sample(range(len(dataset)), 5)
query_indices = random.sample(range(len(dataset)), 5)

support_set = torch.stack([dataset[i][0] for i in support_indices])
query_set = torch.stack([dataset[i][0] for i in query_indices])
support_labels = torch.tensor([dataset[i][1] for i in support_indices])
query_labels = torch.tensor([dataset[i][1] for i in query_indices])

accuracy = evaluate_few_shot_model(ssl_model, matching_net, support_set, query_set, support_labels, query_labels)
print(f'Few-shot learning accuracy: {accuracy:.2f}')

# Import necessary libraries
import torch
import torch.nn as nn

"""This is a library of baseline models"""

glove_file = "glove.840B.300d.txt"

# Bi-LSTM Model for PyTorch
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        final = self.fc(lstm_out[-1])
        return final
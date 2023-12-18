# Import necessary libraries
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

"""This is a library of baseline models"""

# Bi-LSTM Model for PyTorch
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_matrix, d_embed, d_hidden, d_out, dropout = 0.5, num_layers = 2, bidirectional = True):
        super(BiLSTMModel, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(d_embed, d_hidden, dropout = dropout, bidirectional=bidirectional, num_layers = num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * d_hidden * 2,d_out)
        )

    def forward(self, x, seq_lengths):
        x = self.embeddings(x)
        # Sort x and seq_lengths in descending order
        # This is required for packing the sequence
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]
        # Pack the sequence
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True)
        # Pass the packed sequence through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True,total_length = x.size()[1])
        _, unperm_idx = perm_idx.sort(0)
        #unperm_idx = unperm_idx.to(self.device)
        output = output.index_select(0, unperm_idx)
        #This takes all the outputs across the cells
        mean_pooled = torch.mean(output, dim=1)
        max_pooled, _ = torch.max(output, dim=1)
        output = torch.cat((mean_pooled,max_pooled),dim=1)
        output = self.fc(output)
        return output
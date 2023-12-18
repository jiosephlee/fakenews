import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset
from utils import augment_with_noise
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import BiLSTMModel
from utils import get_glove_mapping
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

NUM_EPOCHS = 7

class Vocabulary:
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.index = 2  # Start indexing from 2

    def add_word(self, word):
        if word not in self.word_to_id:
            self.word_to_id[word] = self.index
            self.id_to_word[self.index] = word
            self.index += 1

    def get_id(self, word):
        return self.word_to_id.get(word, self.word_to_id["<UNK>"])
 
# Training Function
def train_rnn(device, model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        training_loss = 0.0
        # Training
        model.train()
        for sequences, lengths, labels in train_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
            # Forward Pass
            output = model(sequences, lengths)
            loss = criterion(output, labels)
            # Backpropogate & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # For logging purposes
            training_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        validation_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for sequences, lengths, labels in val_loader:
                    sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
                    outputs = model(sequences, lengths)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
            val_loss_values.append(validation_loss / len(val_loader))
            val_error.append(100-100*val_correct/val_total)
        # Log Model Performance  
        train_loss_values.append(training_loss)
        train_error.append(100-100*train_correct/train_total)
        print(f'Epoch {epoch+1}, Training Loss: {training_loss/len(train_loader)}, Validation Error: {val_error[-1]}, Training Error: {train_error[-1]}')
        for op_params in optimizer.param_groups:
            op_params['lr'] = op_params['lr'] * 0.6
    return train_error,train_loss_values, val_error, val_loss_values

# Function to convert sentences to sequences of indices
def text_to_sequence(texts, vocab_map):
    sequences = []
    for sentence in texts:
        seq = [vocab_map.get_id(word) for word in sentence]
        sequences.append(seq)
    return sequences

def get_embedding_matrix(vocab, n_embed, d_embed, glove_map, randomize_init = False):
    """
    Initialize the embedding matrix

    INPUT:
    n_embed         - size of the dictionary of embeddings
    d_embed         - the size of each embedding vector
    glove_map       - the map you created storing all of embeddings you will need from GloVE
    randomize_init  - if True, ignore the embeddings from glove_map and intilize all embeddings to random guassian noise (np.random.normal will be useful).

    OUTPUT:
    embedding_matrix  - a numpy matrix of mapping from word id to embedding

    """
    if randomize_init:
        return np.random.normal(0, 1, (n_embed, d_embed))
    embedding_matrix = np.zeros((n_embed, d_embed))
    for key in sorted(vocab.id_to_word.keys()):
        word = vocab.id_to_word[key]
        vector = glove_map[word] if word in glove_map else np.random.normal(0, 1, d_embed)
        embedding_matrix[key, np.arange(d_embed)] = vector
    return embedding_matrix

def pad_sequences(sequences, max_len=None, pad_id=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.full((len(sequences), max_len), pad_id, dtype=int)
    sequence_lengths = np.zeros(len(sequences), dtype=int)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_sequences[i, :length] = seq[:length]
        sequence_lengths[i] = length
    return torch.tensor(padded_sequences,dtype=torch.int64), torch.tensor(sequence_lengths,dtype=torch.int64)

class SequenceDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]
    
if __name__ == "__main__":
    # Load Device
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("------------Loading Data-----------")
    # Load the LIAR dataset
    dataset = load_dataset("liar")
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    # Augment the training dataset!!!
    #print("---------Augmenting--------")
    # vocab_set = set()
    # train_texts = [item['statement'] for item in dataset['train']]
    # for text in train_texts:
    #     vocab_set.update(word_tokenize(text))
    # print(train_dataset)
    # temp = [row for row in train_dataset]
    # augmented_dataset = augment_with_noise(temp[:500],train_dataset, vocab_set)
    
    # print(augmented_dataset)
    # # Concatenate the original train_dataset with the augmented_dataset
    # train_dataset = concatenate_datasets([train_dataset, augmented_dataset])

    print("------------Processing Data-----------")
    # Tokenize the sentences to get a vocabulary
    def tokenize_dataset(dataset):
        return [word_tokenize(sentence['statement']) for sentence in dataset]
    train_sequences = tokenize_dataset(train_dataset)
    valid_sequences = tokenize_dataset(valid_dataset)

    # Build vocab that handles going from token -> id
    vocab = Vocabulary()
    vocab_set = set()
    for sentence in train_sequences:
        for word in sentence:
            vocab.add_word(word)
            vocab_set.add(word)

    # Convert texts to sequences of indices using vocab
    train_sequences = text_to_sequence(train_sequences, vocab)
    valid_sequences = text_to_sequence(valid_sequences, vocab)

    # Glove Mapping that gives word -> embeddings 
    glove_map = get_glove_mapping(vocab_set,"glove.840B.300d.txt")

    # Parameters
    d_out = 6  # the number of output classes of the model
    n_embed = len(vocab.word_to_id) # the total number of word embeddings in the input layer
    d_embed = 300 # the dimensionality of each word embedding

    # Generate a word embeddings: id -> embedding using Glove & Vocab
    embedding_matrix = torch.tensor(get_embedding_matrix(vocab, n_embed, d_embed, glove_map, randomize_init = False), dtype=torch.float32)

    # Padding Sequence 
    train_sequences, train_lengths = pad_sequences(train_sequences)
    valid_sequences, valid_lengths = pad_sequences(valid_sequences)
    
    # Preparing input
    train_labels = torch.tensor(np.array([item['label'] for item in train_dataset]), dtype=torch.long)
    valid_labels = torch.tensor(np.array([item['label'] for item in valid_dataset]), dtype=torch.long)
    # Create datasets and dataloaders
    train_dataset = SequenceDataset(train_sequences, train_lengths, train_labels)
    valid_dataset = SequenceDataset(valid_sequences, valid_lengths, valid_labels)

    batch_size = 64  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    print("------------Training-----------")
    d_hidden = 150
    model = BiLSTMModel(embedding_matrix,d_embed,d_hidden,d_out,dropout=0.6,num_layers=2) 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 5e-4
    weight_decay=1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_error,train_loss_values, val_error, val_loss_values = train_rnn(device, model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS)

    # Plot the training error
    plt.figure(figsize=(10, 5))
    plt.plot(val_error, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Validation Error')
    plt.legend()
    plt.show()
    plt.savefig('rnn_validation_error.png')  # This will save the plot as an image

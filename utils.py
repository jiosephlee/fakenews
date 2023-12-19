from torch.utils.data import Dataset as DatasetClass
from datasets import Dataset
import torch 
import numpy as np
import copy
import random
import nltk
nltk.download('averaged_perceptron_tagger')


glove_file = "glove.840B.300d.txt"

class TextDatasetContext(DatasetClass):
    def __init__(self, data, tokenizer):
        """
        data: A list of tuples, where each tuple contains the text and its corresponding label.
        tokenizer: Tokenizer to be used for tokenizing the text.
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract text and label from the data structure
        text, label = self.data[idx]['context'] + ': ' + self.data[idx]['statement'],self.data[idx]['label']

        # Tokenize the text
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokenized_text['input_ids'][0]
        attention_mask = tokenized_text['attention_mask'][0]

        # Convert label to a tensor
        label = torch.tensor(label)

        return input_ids, attention_mask, label
    
class TextDatasetContextAndBinary(Dataset):
    def __init__(self, data, tokenizer):
        """
        data: A list of tuples, where each tuple contains the text and its corresponding label.
        tokenizer: Tokenizer to be used for tokenizing the text.
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract text and label from the data structure
        text, label = self.data[idx]['context'] + ': ' + self.data[idx]['statement'], self.data[idx]['label']

        # Convert multi-class label to binary label
        # Grouping first three labels as '0' and last three labels as '1'
        binary_label = 0 if label < 3 else 1

        # Tokenize the text
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokenized_text['input_ids'][0]
        attention_mask = tokenized_text['attention_mask'][0]

        # Convert binary label to a tensor
        binary_label = torch.tensor(binary_label)

        return input_ids, attention_mask, binary_label


class TextDatasetBase(DatasetClass):
    def __init__(self, data, tokenizer):
        """
        data: A list of tuples, where each tuple contains the text and its corresponding label.
        tokenizer: Tokenizer to be used for tokenizing the text.
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract text and label from the data structure
        text, label = self.data[idx]['statement'],self.data[idx]['label']

        # Tokenize the text
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokenized_text['input_ids'][0]
        attention_mask = tokenized_text['attention_mask'][0]

        # Convert label to a tensor
        label = torch.tensor(label)

        return input_ids, attention_mask, label
    
def noise_text(text, vocabulary, noise_ratio=0.15):
    tokens = text.split()
    noised_tokens = []
    for token in tokens:
        if random.random() < noise_ratio:
            # Apply noise - here, we simply mask the token, but you can modify this
            noised_tokens.append(random.choice(list(vocabulary)))
        else:
            noised_tokens.append(token)
    return ' '.join(noised_tokens)

def noise_text_with_pos(text, vocabulary, noise_ratio=0.15):
    # Tokenize and POS tag the input text
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)

    # Create a dictionary to store vocabulary words by their POS tags
    vocab_by_pos = {}
    for word, pos in nltk.pos_tag(vocabulary):
        if pos not in vocab_by_pos:
            vocab_by_pos[pos] = []
        vocab_by_pos[pos].append(word)

    # Replace tokens with same POS words from the vocabulary
    noised_tokens = []
    for word, tag in tagged_tokens:
        if random.random() < noise_ratio and tag in vocab_by_pos:
            # Replace with a word of the same POS
            noised_tokens.append(random.choice(vocab_by_pos[tag]))
        else:
            noised_tokens.append(word)

    return ' '.join(noised_tokens)

def delete_text(text, vocabulary, noise_ratio=0.15):
    tokens = text.split()
    noised_tokens = []
    for token in tokens:
        if random.random() < noise_ratio:
            continue
        else:
            noised_tokens.append(token)
    return ' '.join(noised_tokens)

def noise_text(text, vocabulary, noise_ratio=0.15):
    tokens = text.split()
    noised_tokens = []
    for token in tokens:
        if random.random() < noise_ratio:
            # Apply noise - here, we simply mask the token, but you can modify this
            noised_tokens.append(random.choice(list(vocabulary)))
        else:
            noised_tokens.append(token)
    return ' '.join(noised_tokens)

def augment_with_noise(sample_dataset, train_dataset, vocab_set):
    augmented_train_data = []
    for item in sample_dataset:
        augmented_item = copy.deepcopy(item)
        original_text = item['statement']
        noised_text = noise_text(original_text, vocab_set)
        augmented_item['statement'] = noised_text
        augmented_train_data.append(augmented_item)
    augmented_dataset = Dataset.from_dict({key: [d[key] for d in augmented_train_data] for key in train_dataset.features}, features=train_dataset.features) 
    return augmented_dataset

def augment_with_noise_pos(sample_dataset, train_dataset, vocab_set):
    augmented_train_data = []
    for item in sample_dataset:
        augmented_item = copy.deepcopy(item)
        original_text = item['statement']
        noised_text = noise_text_with_pos(original_text, vocab_set)
        augmented_item['statement'] = noised_text
        augmented_train_data.append(augmented_item)
    augmented_dataset = Dataset.from_dict({key: [d[key] for d in augmented_train_data] for key in train_dataset.features}, features=train_dataset.features) 
    return augmented_dataset

def augment_with_deletion(sample_dataset, train_dataset, vocab_set):
    augmented_train_data = []
    for item in sample_dataset:
        augmented_item = copy.deepcopy(item)
        original_text = item['statement']
        noised_text = delete_text(original_text, vocab_set)
        augmented_item['statement'] = noised_text
        augmented_train_data.append(augmented_item)
    augmented_dataset = Dataset.from_dict({key: [d[key] for d in augmented_train_data] for key in train_dataset.features}, features=train_dataset.features) 
    return augmented_dataset

#takes about 1 minute to read through the whole file and find the words we need.
def get_glove_mapping(vocab, file):
    """
    Gets the mapping of words from the vocabulary to pretrained embeddings

    INPUT:
    vocab       - set of vocabulary words
    file        - file with pretrained embeddings

    OUTPUT:
    glove_map   - mapping of words in the vocabulary to the pretrained embedding

    """

    glove_map = {}
    with open(file,'rb') as fi:
        count = 0
        for l in fi:
            try:
                #### STUDENT CODE HERE ####
                line_str = bytes.decode(l, 'utf-8')
                count += 1
                tokens = line_str.split(" ")
                word = tokens[0]
                if word in vocab:
                  # avoid computing the vector if the word is not in vocab
                  vector = np.array(tokens[1:], dtype='float')
                  glove_map[word] = vector
                ## only include vectors that are found in the vocabulary.
                #### STUDENT CODE ENDS HERE ####
            except:
                #some lines have urls, we don't need them.
                pass
    return glove_map


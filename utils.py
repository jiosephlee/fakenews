from torch.utils.data import Dataset
import torch 

class TextDatasetContext(Dataset):
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
    
class TextDatasetBase(Dataset):
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
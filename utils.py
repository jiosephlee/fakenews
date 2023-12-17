from torch.utils.data import Dataset
import torch 

glove_file = "glove.840B.300d.txt"

class TextDataset(Dataset):
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
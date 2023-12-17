from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import torch
from utils import get_glove_mapping
from nltk.tokenize import word_tokenize

NUM_EPOCHS = 100

# Training Function
def train(device, model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        training_loss = 0.0
        for op_params in optimizer.param_groups:
            op_params['lr'] = op_params['lr'] * 0.6
        # Training
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            # Forward Pass
            output = model(data)
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
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
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
        print(f'Epoch {epoch+1}, Training Loss: {training_loss}, Validation Error: {val_error[-1]}, Error: {train_error[-1]}')
    return train_error,train_loss_values, val_error, val_loss_values

if __name__ == "__main__":
    # Load the LIAR dataset
    dataset = load_dataset("liar")

    # Separate features and labels
    train_texts = [item['statement'] for item in dataset['train']]
    train_labels = [item['label'] for item in dataset['train']]
    valid_texts = [item['statement'] for item in dataset['validation']]
    valid_labels = [item['label'] for item in dataset['validation']]

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_texts)
    y_train = train_labels

    # Transform the validation data
    X_val = vectorizer.transform(valid_texts)
    y_val = valid_labels

    # SVM Model
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    # Logistic Regression Model
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Evaluate SVM Model
    svm_predictions = svm_model.predict(X_val)
    svm_accuracy = accuracy_score(y_val, svm_predictions)
    print("SVM Validation Accuracy:", svm_accuracy)

    # Evaluate Logistic Regression Model
    logistic_predictions = logistic_model.predict(X_val)
    logistic_accuracy = accuracy_score(y_val, logistic_predictions)
    print("Logistic Regression Validation Accuracy:", logistic_accuracy)

    # RNN

    # Tokenize the texts to create a vocabulary set
    vocab_set = set()
    for text in train_texts:
        vocab_set.update(word_tokenize(text.lower()))

    # Glove Mapping for RNN

    vocab_set = set(word for sublist in train_texts for word in sublist)
    glove_map = get_glove_mapping(vocab_set,"/content/glove.840B.300d.txt")
    glove_keys = glove_map.keys()

    d_out = 6  # the number of output classes of the model
    n_embed = len(vocab_set) # the total number of word embeddings in the input layer
    d_embed = 300 # the dimensionality of each word embedding

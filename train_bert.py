# Core ML Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import TextDatasetBase, TextDatasetContext, augment_with_noise, augment_with_deletion, augment_with_noise_pos
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# Additional Libraries
import sys
import csv
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CyclicLR
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

NUM_EPOCHS = 7

# Training Function for BERT Model
def train_bert_cyclical(device, model, train_loader, val_loader, optimizer, num_epochs):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    #swa_model = AveragedModel(model)
    best_val_error = 100
    swa_start = num_epochs // 2
    scheduler = None
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        training_loss = 0.0
        count = 0
        if epoch == swa_start:
            scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=5e-5, step_size_up=500, step_size_down=1500, mode='exp_range', cycle_momentum=False)
        # Training
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # Run the BERT Model
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Backpropogate & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            if epoch >= swa_start:
                scheduler.step()
            # Update SWA after swa_start epochs
            #if epoch >= swa_start:
                #swa_model.update_parameters(model)
            # For logging purposes
            training_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            count += 1
            if count % 50 == 0:
                print(f'Epoch {epoch+1} in progress, Training Loss: {training_loss/count}')
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        validation_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for input_ids, attention_mask, labels in valid_loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    # Run the BERT Model
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    _, predicted = torch.max(outputs.logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    validation_loss += loss.item()
            val_loss_values.append(validation_loss / len(val_loader))
            val_error.append(100-100*val_correct/val_total)
        if epoch < swa_start:
            for op_params in optimizer.param_groups:
                op_params['lr'] = op_params['lr'] * 0.8
        # Log Model Performance  
        train_loss_values.append(training_loss/len(train_loader))
        train_error.append(100-100*train_correct/train_total)
        if val_error[-1] < best_val_error:
            best_val_error = val_error[-1]
            # Save the best model
            torch.save(model.state_dict(), 'best_model_checkpoint.pth')
        print(f'Epoch {epoch+1}, Training Loss: {training_loss/len(train_loader)}, Validation Error: {val_error[-1]}, Training Error: {train_error[-1]}')
        # Update batch normalization in SWA model
        #torch.optim.swa_utils.update_bn(train_loader, swa_model)
    return train_error,train_loss_values, val_error, val_loss_values

# Training Function for BERT Model
def train_bert(device, model, train_loader, val_loader, optimizer, num_epochs):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    best_val_error = 100
    swa_model = AveragedModel(model)
    swa_start = 1
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        training_loss = 0.0
        count = 0
        # Training
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # Run the BERT Model
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Backpropogate & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # For logging purposes
            training_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            count += 1
            if count % 50 == 0:
                print(f'Epoch {epoch+1} in progress, Training Loss: {training_loss/count}')
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        validation_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for input_ids, attention_mask, labels in valid_loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    # Run the BERT Model
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    _, predicted = torch.max(outputs.logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    validation_loss += loss.item()
            val_loss_values.append(validation_loss / len(val_loader))
            val_error.append(100-100*val_correct/val_total)
        for op_params in optimizer.param_groups:
            op_params['lr'] = op_params['lr'] * 0.9
        # Log Model Performance  
        train_loss_values.append(training_loss/len(train_loader))
        train_error.append(100-100*train_correct/train_total)
        if val_error[-1] < best_val_error:
            best_val_error = val_error[-1]
            # Save the best model
            torch.save(model.state_dict(), 'best_model_checkpoint.pth')
        print(f'Epoch {epoch+1}, Training Loss: {training_loss/len(train_loader)}, Validation Error: {val_error[-1]}, Training Error: {train_error[-1]}')
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
    return train_error,train_loss_values, val_error, val_loss_values, swa_model

if __name__ == "__main__":
    # Load the LIAR dataset
    dataset = load_dataset("liar")

    # The dataset consists of training, validation, and test splits
    train_dataset = dataset["train"]
    print(train_dataset)
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    vocab_set = set()
    train_texts = [item['statement'] for item in dataset['train']]
    for text in train_texts:
        vocab_set.update(word_tokenize(text.lower()))
    print(vocab_set)
    # Augment the training dataset!!!
    print("---------Augmenting--------")
    temp = [row for row in train_dataset]
    num_of_aug = int(sys.argv[2])
    print("...")
    augmented_dataset = augment_with_noise(temp[:num_of_aug],train_dataset, vocab_set)
    print("...")
    augmented_dataset_2 = augment_with_noise_pos(temp[num_of_aug:2*num_of_aug],train_dataset, vocab_set)
    print("...")
    augmented_dataset_3 = augment_with_deletion(temp[2*num_of_aug:3*num_of_aug],train_dataset, vocab_set)
    if len(augmented_dataset) > 0:
        print(augmented_dataset[0:5])
        print(augmented_dataset_2[0:5])
        print(augmented_dataset_3[0:5])
    # Concatenate the original train_dataset with the augmented_dataset
    train_dataset = concatenate_datasets([train_dataset, augmented_dataset, augmented_dataset_2, augmented_dataset_3])
    print("---------Finished Augmenting--------")

    # Load Device
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Bert Models
    # Check if the model name is provided as a command-line argument
    if len(sys.argv) != 3:
        print("Usage: python filename.py [model_name]")
        print("model_name: bert-base-cased, distilbert-base-cased, or roberta-base")
        sys.exit(1)

    model_name = sys.argv[1]
    # Validate the model name
    valid_models = ['bert-base-cased', 'distilbert-base-cased', 'roberta-base']

    if model_name not in valid_models:
        print("Invalid model name. Choose from bert-base-cased, distilbert-base-cased, or roberta-base.")
        sys.exit(1)
    
    # Initialize Pre-trained Models & Hyperparameters
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    model = model.to(device)
    lr = 5e-5
    weight_decay=1e-4
    batch_size = 16
    if model_name == 'roberta-base':
        lr = 2e-5
        weight_decay=1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Generate Loaders & Modify Data
    train_dataset = TextDatasetContext(train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TextDatasetContext(valid_dataset, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TextDatasetContext(test_dataset, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    print("------------Training-----------")
    train_error,train_loss_values, val_error, val_loss_values, swa_model = train_bert(device, model, train_loader, valid_loader, optimizer, NUM_EPOCHS)

    print("-------------Saving Results--------")
    # Plot Training and Validation Error
    plt.figure(figsize=(10, 5))
    plt.plot(train_error, label='Training Error')
    plt.plot(val_error, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f'Training and Validation Error of {model_name}-{num_of_aug}-RandomReplacement\nlr={lr}, weight_decay={weight_decay}, batch_size={batch_size}')
    plt.legend()
    plt.savefig(f'train_val_error_{model_name}_{num_of_aug}.png')
    plt.show()

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss of {model_name}-{num_of_aug}-RandomReplacement\nlr={lr}, weight_decay={weight_decay}, batch_size={batch_size}')
    plt.legend()
    plt.savefig(f'train_val_loss_{model_name}_{num_of_aug}.png')
    plt.show()

    with open(f'training_results_{model_name}_{num_of_aug}-RandomReplacement.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['Epoch', 'Training Error', 'Training Loss', 'Validation Error', 'Validation Loss'])

        # Write the data
        for epoch, (tr_err, tr_loss, val_err, val_loss) in enumerate(zip(train_error, train_loss_values, val_error, val_loss_values), 1):
            writer.writerow([epoch, tr_err, tr_loss, val_err, val_loss])

        print("Data written to training_results.csv")

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy_1 = 100 * test_correct / test_total
    print(f'Last Version, Test Accuracy: {test_accuracy_1}%')

    swa_model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = swa_model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy_2 = 100 * test_correct / test_total
    print(f'Averaged Model Test Accuracy: {test_accuracy_2}%')

    # Load the best model
    model.load_state_dict(torch.load('best_model_checkpoint.pth'))
     # Test the model
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy_3 = 100 * test_correct / test_total
    print(f'Best Val Model, Test Accuracy: {test_accuracy_3}%')

    # The string to be written to the file
    string_to_write = f"Last Version, Test Accuracy: {test_accuracy_1}%\nAveraged Model Test Accuracy: {test_accuracy_2}%\nBest Val Model, Test Accuracy: {test_accuracy_3}%"

    # Open a file in write mode
    with open(f'{model_name}_{num_of_aug}_RandomReplacement.txt', 'w') as file:
        # Write the string to the file
        file.write(string_to_write)
        print(f'Accuracies written to {model_name}_{num_of_aug}_RandomReplacement.txt')
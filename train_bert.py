import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import TextDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

NUM_EPOCHS = 10

# Training Function for BERT Model
def train_bert(device, model, train_loader, val_loader, num_epochs):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
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
            op_params['lr'] = op_params['lr'] * 0.35
        # Log Model Performance  
        train_loss_values.append(training_loss/len(train_loader))
        train_error.append(100-100*train_correct/train_total)
        print(f'Epoch {epoch+1}, Training Loss: {training_loss/len(train_loader)}, Validation Error: {val_error[-1]}, Training Error: {train_error[-1]}')
    return train_error,train_loss_values, val_error, val_loss_values

if __name__ == "__main__":
    # Load the LIAR dataset
    dataset = load_dataset("liar")

    # The dataset consists of training, validation, and test splits
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Load Device
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Bert Models
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=6)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # Generate Loaders & Modify Data
    train_dataset = TextDataset(train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataset = TextDataset(valid_dataset, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # Train the model
    print("------------Training-----------")
    train_error,train_loss_values, val_error, val_loss_values = train_bert(device, model, train_loader, valid_loader, NUM_EPOCHS)

    # Plot the training error
    plt.figure(figsize=(10, 5))
    plt.plot(val_error, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Validation Error')
    plt.legend()
    plt.show()
    plt.savefig('validation_error.png')  # This will save the plot as an image

    # Save the model
    torch.save(model.state_dict(), 'model_state.pth')
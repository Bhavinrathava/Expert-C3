import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import torch



# Preprocess the data

def load_data(data_file):
    """
    Load the data from the file and return the text and labels
    """
    data = pd.read_csv(data_file)
    texts = data['Question'].tolist()
    labels = data['Label'].tolist()

    return texts, labels

texts, labels = load_data('query_new.csv')
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.15)

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Enconde the texts
train_encoodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512)
val_encoodings = tokenizer(val_texts, padding=True, truncation=True, max_length=512)

# Convert the data to tensors
def convert_to_tensors(encoodings, labels):
    input_ids = torch.tensor(encoodings['input_ids'])
    attention_mask = torch.tensor(encoodings['attention_mask'])
    labels = torch.tensor(labels)

    return input_ids, attention_mask, labels

# Convert your training and validation data into tensors
train_input_ids, train_attention_mask, train_labels = convert_to_tensors(train_encoodings, train_labels)
val_input_ids, val_attention_mask, val_labels = convert_to_tensors(val_encoodings, val_labels)

# Create a DataLoader
def create_data_loader(input_ids, attention_mask, labels, batch_size, sampler):
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return dataloader

train_dataloader = create_data_loader(train_input_ids, train_attention_mask, train_labels, 32, RandomSampler)
val_dataloader = create_data_loader(val_input_ids, val_attention_mask, val_labels, 32, SequentialSampler)

# Load the model fro sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
    )

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configure the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)


# Trainig loop
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )

# function to calculate the accuracy
def accuracy(preds, labels):
    pred = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    return np.mean(pred == labels)

# Traning and validation loop
for epoch in range(epochs):
    # Training loop
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        print(step, end = " ", flush=True)
        # add the batch to the device
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        # zero the gradients
        model.zero_grad()
        # forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        # backward pass
        loss.backward()
        # clip the gradients to 1.0 to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update the weights
        optimizer.step()
        scheduler.step()

    # Validation loop
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        total_val_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_val_accuracy += accuracy(logits, label_ids)

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_accuracy = total_val_accuracy / len(val_dataloader)

    print(f'Epoch: {epoch+1}')
    print(f'Training loss: {avg_train_loss}')
    print(f'Validation loss: {avg_val_loss}')
    print(f'Validation accuracy: {avg_val_accuracy}')
    print('-----------------')

# Save the model
model.save_pretrained('./model_save/')

### Test the model
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=512)
test_input_ids, test_attention_mask, test_labels = convert_to_tensors(test_encodings, test_labels)
test_dataloader = create_data_loader(test_input_ids, test_attention_mask, test_labels, batch_size=32, sampler=SequentialSampler)

model.eval()
# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

total_test_accuracy = 0
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
    # extract the values from PyTorch tensors and convert them into NumPy arrays.
    logits = logits.detach().cpu().numpy()
    label_ids = labels.to('cpu').numpy()
    total_test_accuracy += accuracy(logits, label_ids)
    
    # Convert logits to predicted labels
    predicted_labels.extend(np.argmax(logits, axis=1))
    true_labels.extend(label_ids)

avg_test_accuracy = total_test_accuracy/len(test_dataloader)
print(f'Calculated Test accuracy: {avg_test_accuracy}')

accuracy_rate = accuracy_score(true_labels, predicted_labels)
precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
print(f"Accuracy: {accuracy_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-1 score: {fscore}")

# Calculated Test accuracy: 0.9776785714285714
# Accuracy: 0.9768518518518519
# Precision: 0.9763779527559056
# Recall: 0.9841269841269841
# F-1 score: 0.9802371541501976
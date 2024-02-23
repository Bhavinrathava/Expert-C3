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

texts, labels = load_data('query.csv')
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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
epochs = 4
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

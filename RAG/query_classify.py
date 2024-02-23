from transformers import BertForSequenceClassification, BertTokenizer
import torch

def load_model_tokenizer(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return model, tokenizer

def predict_query_label(query):
    # Load model
    model, tokenizer = load_model_tokenizer('../Query_Classification/model_save/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenize the input sentence
    inputs = tokenizer(query, padding=True, truncation=True, max_length=512)
    
    # Move input tensors to the device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    
    # Get predicted label (assuming binary classification)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    
    return predicted_label
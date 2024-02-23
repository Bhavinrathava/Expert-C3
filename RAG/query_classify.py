from transformers import BertForSequenceClassification, BertTokenizer
import torch

def load_model_tokenizer(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return model, tokenizer

def predict_query_label(query):
    # load model
    model, tokenizer = load_model_tokenizer('../Query_Classification/model_save/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs = tokenizer(query, padding=True, truncation=True, max_length=512)

    # if tokenizer returns a list, take the first element
    if isinstance(inputs, list):
        inputs = inputs[0]

    # move input tensors to the device
    input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(device)
    
    #input_ids = inputs['input_ids'].to(device)
    #attention_mask = inputs['attention_mask'].to(device)
    
    # perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    probabilities = torch.softmax(logits, dim=1)
    
    # get predicted label
    predicted_label = torch.argmax(probabilities, dim=1).item()
    
    return predicted_label

if __name__ == '__main__':
    #get answer from RAG
    query = "What is northwestern University's policy on Gifts?"
    label = predict_query_label(query)
    print(label)

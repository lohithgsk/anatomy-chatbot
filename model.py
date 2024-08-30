import os
import PyPDF2
import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text):
    cleaned_text = text.replace('\n', ' ').replace('\r', '').strip()
    return ' '.join(cleaned_text.split())

def store_data(text):
    data = {'text': text}
    with open('extracted_data.json', 'w') as f:
        json.dump(data, f)

def prepare_training_data(text):
    chunk_size = 384  # Adjust based on your needs
    stride = 128
    chunks = []
    for i in range(0, len(text) - chunk_size + 1, stride):
        chunks.append(text[i:i+chunk_size])
    return chunks


def fine_tune_model(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    # Prepare input data
    inputs = tokenizer(texts, return_tensors="pt", max_length=384, truncation=True, padding=True)
    
    # Create dummy answers (we're not actually training on specific QA pairs)
    start_positions = torch.zeros(len(texts), dtype=torch.long)
    end_positions = torch.zeros(len(texts), dtype=torch.long)

    # Create dataset and dataloader
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], start_positions, end_positions)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Training setup
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, start_positions, end_positions = batch
            
            outputs = model(input_ids, attention_mask=attention_mask, 
                            start_positions=start_positions, 
                            end_positions=end_positions)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}/{num_epochs} completed")

    return model, tokenizer

def get_response(question, context, model, tokenizer, device):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()
    
    if not answer or answer.isspace():
        return "I'm sorry, I couldn't find a specific answer to that question in the provided context. Could you please rephrase or ask another question?"
    
    return answer

pdf_path = '/content/anatomy_vol_2.pdf'

# Extract and clean data
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(raw_text)

# Store data
store_data(cleaned_text)

# Prepare training data
texts = prepare_training_data(cleaned_text)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = fine_tune_model(texts)

model.to(device)

while True:
    question = input("Ask a question about human anatomy (or type 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    # Find the most relevant context
    relevant_context = max(texts, key=lambda x: sum(1 for word in question.lower().split() if word in x.lower()))
    
    try:
        answer = get_response(question, relevant_context, model, tokenizer, device)
        print(f"A: {answer}")
    except Exception as e:
        print(f"An error occurred: {str(e)}. Please try asking another question.")
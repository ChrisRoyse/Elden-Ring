import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging

# Set up logging to track progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Elden Ring dataset
eldenring_df = pd.read_csv('c:/python39/eldenring_data.csv')

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('c:/python39/margit_bert_model')
tokenizer = BertTokenizer.from_pretrained('c:/python39/margit_bert_tokenizer')

# Make sure to set the model to evaluation mode
model.eval()

# Function to classify a message as Margit-related or not, including the score
def classify_message(message):
    inputs = tokenizer(message, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_label = torch.argmax(logits, dim=-1).item()
    score = probabilities[0][predicted_label].item()
    return predicted_label, score

# Create a list to store the results
results = []

# Classify messages and add them to the results
logging.info("Starting classification of messages.")
for index, row in eldenring_df.iterrows():
    message = row['messageText']
    label, score = classify_message(message)
    
    # Add progress logging every 100 messages
    if index % 100 == 0:
        logging.info(f"Processing message {index}/{len(eldenring_df)}: {message}")
    
    # If label is 1 (Margit-related), store it with its score
    if label == 1:  # 1 indicates a Margit-related message based on training
        logging.info(f"Classified as Margit-related: {message} (Score: {score:.4f})")
        results.append({
            'messageText': message,
            'predicted_label': label,
            'label_score': score,
            'explanation': 'Classified as Margit-related due to model prediction score and context.'
        })

# Convert the results to a DataFrame
output_df = pd.DataFrame(results)

# Save the results to margittesting.csv
output_df.to_csv('c:/python39/margittesting.csv', index=False)

logging.info("Classification complete. Results saved to margittesting.csv")

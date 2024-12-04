```markdown
# The Number One LLC - Player Insights Automation

Welcome to the repository for **The Number One LLC**, where we leverage cutting-edge machine learning models to transform player feedback into actionable insights. This project encompasses data collection, sentiment analysis, and message paraphrasing to help game developers make data-driven decisions that enhance player satisfaction and game quality.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training the BERT Model](#training-the-bert-model)
- [Classifying Messages](#classifying-messages)
- [Rewriting Messages with GPT-2](#rewriting-messages-with-gpt-2)
- [Usage](#usage)
- [Results](#results)
- [Contact](#contact)
- [License](#license)

## Features

- **Massive Data Collection**: Capture millions of live chat messages daily across major streaming platforms.
- **Real-Time Analytics**: Access immediate, actionable insights to make informed decisions swiftly.
- **Player-to-Player Insights**: Understand genuine player sentiments by analyzing their interactions.
- **Data-Driven Decisions**: Reduce risks by basing your game development on solid, unbiased data.
- **Custom Consulting**: Receive personalized support and tailored reports to meet your specific needs.
- **Filtered Feedback**: Our AI filters out trolling and irrelevant noise, delivering only what matters.

## Project Structure

```
.
├── data
│   ├── goodmessage.csv
│   ├── badmessage.csv
│   ├── eldenring_data.csv
├── models
│   ├── margit_bert_model
│   ├── margit_bert_tokenizer
├── results
│   ├── margittesting.csv
│   ├── margit_bert_model
│   ├── margit_bert_tokenizer
├── scripts
│   ├── train_bert.py
│   ├── classify_messages.py
│   ├── rewrite_messages.py
├── rewritten
│   ├── goodmessagerewritten.csv
├── README.md
├── requirements.txt
```

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/number-one-llc-player-insights.git
   cd number-one-llc-player-insights
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Ensure that your data files are placed in the `data/` directory:

- `goodmessage.csv`: Contains positive messages labeled as Margit-related.
- `badmessage.csv`: Contains negative messages labeled as non-Margit-related.
- `eldenring_data.csv`: Contains messages from Elden Ring to be classified.

**CSV Structure:**

- `goodmessage.csv` should have a column named `posmessage`.
- `badmessage.csv` should have a column named `badmessage`.
- `eldenring_data.csv` should have a column named `messageText`.

## Training the BERT Model

The `train_bert.py` script trains a BERT model to classify messages as Margit-related or not.

### Script: `train_bert.py`

```python
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the datasets
pos_df = pd.read_csv('data/goodmessage.csv')
neg_df = pd.read_csv('data/badmessage.csv')

# Prepare the data
pos_df['label'] = 1  # Label for Margit-related messages
neg_df['label'] = 0  # Label for non-Margit messages

# Combine the datasets
df = pd.concat([pos_df[['posmessage', 'label']].rename(columns={'posmessage': 'text'}),
                neg_df[['badmessage', 'label']].rename(columns={'badmessage': 'text'})])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the data into training and validation sets
train_test = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define metrics for evaluation
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
print("Training BERT model. This might take some time...")
trainer.train()

# Save the model and tokenizer
model.save_pretrained('models/margit_bert_model')
tokenizer.save_pretrained('models/margit_bert_tokenizer')

print("Training completed and model saved.")
```

### How It Works

1. **Data Loading & Preparation**: Reads positive and negative messages, assigns labels, and combines them into a single dataset.
2. **Tokenization**: Uses BERT tokenizer to process text data.
3. **Model Training**: Fine-tunes BERT for sequence classification with specified training arguments.
4. **Saving**: Saves the trained model and tokenizer for later use.

### Running the Training Script

```bash
python scripts/train_bert.py
```

## Classifying Messages

The `classify_messages.py` script uses the trained BERT model to classify messages from Elden Ring data and saves the results.

### Script: `classify_messages.py`

```python
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging

# Set up logging to track progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Elden Ring dataset
eldenring_df = pd.read_csv('data/eldenring_data.csv')

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('models/margit_bert_model')
tokenizer = BertTokenizer.from_pretrained('models/margit_bert_tokenizer')

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
output_df.to_csv('results/margittesting.csv', index=False)

logging.info("Classification complete. Results saved to margittesting.csv")
```

### How It Works

1. **Loading Data & Model**: Loads the Elden Ring messages and the pre-trained BERT model.
2. **Classification**: Iterates through each message, classifies it, and logs progress.
3. **Saving Results**: Stores Margit-related messages with their confidence scores in `margittesting.csv`.

### Running the Classification Script

```bash
python scripts/classify_messages.py
```

## Rewriting Messages with GPT-2

The `rewrite_messages.py` script uses GPT-2 to generate paraphrased versions of positive messages.

### Script: `rewrite_messages.py`

```python
import os
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Define the input and output file paths
input_file = 'data/goodmessage.csv'
output_file = 'rewritten/goodmessagerewritten.csv'

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: The file {input_file} does not exist.")
    exit(1)

# Load the CSV file
df = pd.read_csv(input_file)

# Make sure the 'posmessage' column exists
if 'posmessage' not in df.columns:
    print("Error: The 'posmessage' column does not exist in the input file.")
    exit(1)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Initialize the text generation pipeline with truncation and pad_token_id set
paraphraser = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    truncation=True
)

# Initialize the list to store rewritten messages
rewritten_messages = []

# Iterate over each message in the 'posmessage' column
for message in df['posmessage']:
    # Generate three paraphrases
    for _ in range(3):
        rewrites = paraphraser(message, max_new_tokens=50, num_return_sequences=1, do_sample=True)
        rewritten_messages.append(rewrites[0]['generated_text'])

# Create a new DataFrame with the rewritten messages
new_df = pd.DataFrame({'rewritten_message': rewritten_messages})

# Save the rewritten messages to a new CSV file
new_df.to_csv(output_file, index=False)

print(f"Rewritten messages have been saved to {output_file}.")
```

### How It Works

1. **Data Loading**: Reads positive messages from `goodmessage.csv`.
2. **Paraphrasing**: Uses GPT-2 to generate three paraphrased versions of each message.
3. **Saving Results**: Stores the paraphrased messages in `goodmessagerewritten.csv`.

### Running the Rewriting Script

```bash
python scripts/rewrite_messages.py
```

## Usage

1. **Training the Model**

   Ensure that `goodmessage.csv` and `badmessage.csv` are properly formatted and placed in the `data/` directory.

   ```bash
   python scripts/train_bert.py
   ```

2. **Classifying Messages**

   After training, classify messages from `eldenring_data.csv`.

   ```bash
   python scripts/classify_messages.py
   ```

3. **Rewriting Messages**

   Generate paraphrased versions of positive messages.

   ```bash
   python scripts/rewrite_messages.py
   ```

## Results

- **Trained BERT Model**: Saved in `models/margit_bert_model` and `models/margit_bert_tokenizer`.
- **Classification Results**: Stored in `results/margittesting.csv`.
- **Rewritten Messages**: Available in `rewritten/goodmessagerewritten.csv`.

## Contact

For any inquiries or support, please contact us:

- **Phone**: (785) 307-4445
- **Email**: [thenumberonellc@gmail.com](mailto:thenumberonellc@gmail.com)

## License

© 2024 The Number One LLC. All rights reserved.

```

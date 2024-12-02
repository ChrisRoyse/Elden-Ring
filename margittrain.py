import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the datasets
pos_df = pd.read_csv('c:/python39/goodmessage.csv')
neg_df = pd.read_csv('c:/python39/badmessage.csv')

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
    save_strategy="epoch",  # Match the evaluation strategy with the save strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,  # Log progress every 10 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
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
model.save_pretrained('c:/python39/margit_bert_model')
tokenizer.save_pretrained('c:/python39/margit_bert_tokenizer')

print("Training completed and model saved.")

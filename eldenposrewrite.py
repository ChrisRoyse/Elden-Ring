import os
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Define the input and output file paths
input_file = 'c:/python39/goodmessage.csv'
output_file = 'c:/python39/goodmessagerewritten.csv'

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

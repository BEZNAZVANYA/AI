from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from datasets import load_from_disk
import torch

# Load the dataset
dataset = load_from_disk(r'D:\.vscode\Cril\Git\AI\data\prepared_dataset')

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['context'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the data loader
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=8)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} completed')

# Save the model and tokenizer
model.save_pretrained(r'D:\.vscode\Cril\Git\AI\models\trained_model')
tokenizer.save_pretrained(r'D:\.vscode\Cril\Git\AI\models\trained_model')

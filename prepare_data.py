from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
import torch
import os
import numpy as np

load_dotenv()

# Load tokenizer from env
model_name = os.getenv('MODEL_NAME')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset in streaming mode
dataset = load_dataset("allenai/dolma", split="train", streaming=True)

def prepare_sequences(dataset, tokenizer, max_length=512, target_tokens=1000000000, output_file="data.bin"):
    """Stream, tokenize, and save data efficiently to binary file"""
    current_tokens = []
    total_tokens = 0
    sequences_saved = 0
    
    # Open binary file for writing
    with open(output_file, "wb") as f:
        for example in dataset:
            # Tokenize the text
            tokens = tokenizer.encode(example['text'], add_special_tokens=False)
            current_tokens.extend(tokens)
            
            # Create sequences of max_length and write to disk immediately
            while len(current_tokens) >= max_length:
                sequence = current_tokens[:max_length]
                # Write as uint16 to save space (assuming vocab_size < 65536)
                seq_array = np.array(sequence, dtype=np.uint16)
                f.write(seq_array.tobytes())
                
                current_tokens = current_tokens[max_length:]
                total_tokens += max_length
                sequences_saved += 1
                
                if total_tokens >= target_tokens:
                    return sequences_saved, total_tokens
                    
            # Progress update
            if sequences_saved % 1000 == 0 and sequences_saved > 0:
                print(f"Saved {sequences_saved} sequences, {total_tokens:,} tokens to {output_file}")
    
    return sequences_saved, total_tokens

# Prepare training data (1B tokens)
print("Preparing training data...")
train_seqs, train_tokens = prepare_sequences(
    dataset, tokenizer, 512, 1000000000, "train_data.bin"
)

# Prepare test data (100k tokens) - restart dataset
dataset_test = load_dataset("allenai/dolma", split="train", streaming=True)
# Skip to avoid overlap with training data
for _ in range(train_seqs):
    next(iter(dataset_test))
    
print("Preparing test data...")
test_seqs, test_tokens = prepare_sequences(
    dataset_test, tokenizer, 512, 100000, "test_data.bin"
)

print(f"Training: {train_seqs} sequences, {train_tokens:,} tokens -> train_data.bin")
print(f"Test: {test_seqs} sequences, {test_tokens:,} tokens -> test_data.bin")
print(f"Vocab size: {tokenizer.vocab_size}, Max token value: {max(tokenizer.get_vocab().values())}")

# Create a simple loader script
loader_code = '''import numpy as np
import torch

def load_data(filename, seq_len=512):
    """Load pretokenized binary data"""
    data = np.fromfile(filename, dtype=np.uint16)
    data = data.reshape(-1, seq_len)
    return torch.from_numpy(data.astype(np.int64))

# Usage:
# train_data = load_data("train_data.bin")
# test_data = load_data("test_data.bin")
'''

with open("data_loader.py", "w") as f:
    f.write(loader_code)
    
print("Created data_loader.py for loading pretokenized data")
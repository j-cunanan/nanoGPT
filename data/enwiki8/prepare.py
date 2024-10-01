import os
import pickle
import numpy as np
from datasets import load_dataset

# Load the enwik8 dataset in raw format
dataset = load_dataset("enwik8", "enwik8-raw")
data = dataset['train']['text'][0]  # Get the raw string data
print(f"Length of dataset in characters: {len(data):,}")

# Convert string to bytes
data_bytes = data.encode('utf-8')
print(f"Length of dataset in bytes: {len(data_bytes):,}")

# Get all the unique bytes that occur in this data
unique_bytes = sorted(set(data_bytes))
vocab_size = len(unique_bytes)
print(f"Number of unique bytes: {vocab_size}")

# Create a mapping from bytes to integers
stoi = {ch: i for i, ch in enumerate(unique_bytes)}
itos = {i: ch for i, ch in enumerate(unique_bytes)}

def encode(s):
    return [stoi[c] for c in s]  # encoder: take bytes, output a list of integers

def decode(l):
    return bytes([itos[i] for i in l])  # decoder: take a list of integers, output bytes

# Create the train, val, and test splits
train_data = data_bytes[:90000000]  # 90M
val_data = data_bytes[90000000:95000000]  # 5M
test_data = data_bytes[95000000:]  # 5M

# Encode all splits to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"Train has {len(train_ids):,} bytes")
print(f"val has {len(val_ids):,} bytes")
print(f"Test has {len(test_ids):,} bytes")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# Save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


# Length of dataset in characters: 99,621,832
# Length of dataset in bytes: 100,000,000
# Number of unique bytes: 205
# Train has 90,000,000 bytes
# val has 5,000,000 bytes
# Test has 5,000,000 bytes
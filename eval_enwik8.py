import os
import time
import torch
import numpy as np
from model import GPTConfig, GPT

# Evaluation settings
out_dir = 'out-enwiki8-char-parallel'
dataset = 'enwiki8'
batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Set up the context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# Data loading function
data_dir = os.path.join('data', dataset)
def get_data(split):
    return np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')

# Load the trained model
def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model, checkpoint

# Evaluation function
@torch.no_grad()
def evaluate_full_set(model, split):
    model.eval()
    data = get_data(split)
    total_loss = 0
    num_batches = len(data) // (batch_size * block_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size * block_size
        end_idx = start_idx + batch_size * block_size
        batch_data = data[start_idx:end_idx].reshape(batch_size, block_size)
        
        x = torch.from_numpy(batch_data.astype(np.int64)).to(device)
        y = torch.from_numpy(np.roll(batch_data, -1, axis=1).astype(np.int64)).to(device)
        
        with ctx:
            logits, loss = model(x, y)
        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_batches} batches")
    
    return total_loss / num_batches

if __name__ == '__main__':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    model, checkpoint = load_model(ckpt_path)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Iteration: {checkpoint['iter_num']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

    # Evaluate the model on full val and test sets
    for split in ['val', 'test']:
        t0 = time.time()
        loss = evaluate_full_set(model, split)
        t1 = time.time()
        
        print(f"\nFull {split} set loss: {loss:.4f}")
        print(f"Evaluation time: {(t1-t0):.2f} seconds")

    print("\nModel configuration:")
    for k, v in model.config.__dict__.items():
        print(f"{k}: {v}")

    # Print PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")

    # Print data statistics
    for split in ['val', 'test']:
        data = get_data(split)
        print(f"{split} data shape: {data.shape}")
        print(f"{split} data min: {data.min()}, max: {data.max()}")


# Iteration: 953500
# Best validation loss: 1.2476
# Processed 100/305 batches
# Processed 200/305 batches
# Processed 300/305 batches

# Full val set loss: 1.3126
# Evaluation time: 6.98 seconds
# Processed 100/305 batches
# Processed 200/305 batches
# Processed 300/305 batches

# Full test set loss: 1.3145
# Evaluation time: 6.82 seconds

# Model configuration:
# block_size: 256
# vocab_size: 205
# n_layer: 12
# n_head: 12
# n_embd: 384
# dropout: 0.2
# bias: False
# conv_block: True
# parallel: True

# PyTorch version: 2.6.0.dev20240929+cu124
# val data shape: (5000000,)
# val data min: 0, max: 204
# test data shape: (5000000,)
# test data min: 0, max: 203
import os
import time
import torch
import numpy as np
from model import GPTConfig, GPT

# Evaluation settings
out_dir = 'out-enwiki8-char-mod'
eval_interval = 250
eval_iters = 200
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
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
def estimate_loss(model):
    out = {}
    model.eval()  # Ensure the model is in eval mode
    for split in ['train', 'val', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

if __name__ == '__main__':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    model, checkpoint = load_model(ckpt_path)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Iteration: {checkpoint['iter_num']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

    print(f"\nInitial model mode: {'eval' if not model.training else 'train'}")

    # Evaluate the model
    t0 = time.time()
    losses = estimate_loss(model)
    t1 = time.time()
    
    print(f"Train loss: {losses['train']:.4f}")
    print(f"Validation loss: {losses['val']:.4f}")
    print(f"Test loss: {losses['test']:.4f}")
    print(f"Evaluation time: {(t1-t0)*1000:.2f}ms")

    print("\nModel configuration:")
    for k, v in model.config.__dict__.items():
        print(f"{k}: {v}")

    # Try a single forward pass for each dataset
    for split in ['train', 'val', 'test']:
        X, Y = get_batch(split)
        model.eval()
        with ctx:
            logits, loss = model(X, Y)
        print(f"\nSingle batch {split} loss: {loss.item():.4f}")

    # Final check of model mode
    print(f"\nFinal model mode: {'eval' if not model.training else 'train'}")

    # Print PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")

    # Print data statistics
    for split in ['train', 'val', 'test']:
        data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        print(f"{split} data shape: {data.shape}")
        print(f"{split} data min: {data.min()}, max: {data.max()}")

    # Calculate bits per byte
    print("\nBits per byte:")
    for split, loss in losses.items():
        print(f"{split}: {loss:.4f}")
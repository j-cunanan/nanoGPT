# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwiki8-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'enwiki8-char'
# wandb_run_name = 'mini-gpt'
wandb_run_name = 'conv-gpt'

dataset = 'enwiki8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
# n_layer = 6
# n_head = 6
n_layer = 12
n_head = 12
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# Tokens per iteration:
# This is now batch_size * context_length = 64 * 256 = 16,384 tokens per iteration.
# Iterations per epoch:
# total_tokens / tokens_per_iter = 90,000,000 / 16,384 ≈ 5,493.16 iterations.
max_iters = 600000
lr_decay_iters = 600000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

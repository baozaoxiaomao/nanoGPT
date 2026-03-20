# Train config for Tianlongbabu (天龙八部) character-level GPT model
# Adapted for nanoGPT, compatible with UTF-8 encoding

# ===================== Basic Config =====================
# Output directory for model checkpoints (ckpt.pt will be saved here)
out_dir = 'out-tianlong-char'
# Evaluate loss every N iterations
eval_interval = 250
# Number of iterations per evaluation (on val set)
eval_iters = 200
# Print training log every N iterations
log_interval = 10
# Save checkpoint only when val loss improves (avoid redundant saves)
always_save_checkpoint = False

# ===================== Dataset Config =====================
# Dataset folder name (must match data/tianlong)
dataset = 'tianlong'
# Gradient accumulation steps (increase for low VRAM GPUs)
gradient_accumulation_steps = 1
# Batch size (reduce to 16/8 if GPU OOM, 32 is safe for most GPUs)
batch_size = 32
# Context window size (max tokens model can process)
block_size = 256

# ===================== Model Architecture =====================
# Number of Transformer layers (reduce to 4 for low VRAM)
n_layer = 6
# Number of attention heads (match n_embd: 384/6=64 per head)
n_head = 6
# Embedding dimension (feature size, multiple of n_head)
n_embd = 384
# Dropout rate (prevent overfitting, 0.2 is universal)
dropout = 0.2

# ===================== Optimizer Config =====================
# Learning rate (1e-3 works for small models)
learning_rate = 1e-3
# Max training iterations (5000 for GPU, 2000 for CPU)
max_iters = 5000
# Learning rate decay iterations (match max_iters)
lr_decay_iters = 5000
# Minimum learning rate (lower bound for decay)
min_lr = 1e-4
# AdamW beta2 (momentum parameter, larger for small datasets)
beta2 = 0.99
# Warmup iterations (gradually increase lr for first N steps)
warmup_iters = 100

# ===================== Logging Config (Optional) =====================
# Disable wandb logging (for classroom training)
wandb_log = False
wandb_project = 'tianlong-char'
wandb_run_name = 'mini-gpt-tianlong'
# 模型输出目录（训练后生成的权重存放在此）
out_dir = 'out-poemtext-char'
# 评估间隔（训练过程中多久验证一次）
eval_interval = 250
# 评估迭代次数
eval_iters = 200
# 日志打印间隔
log_interval = 10
# 是否总是保存检查点
always_save_checkpoint = False
# 数据集文件夹名（对应data/下的poemtext）
dataset = 'poemtext'
# 梯度累积步数
gradient_accumulation_steps = 1
# 批次大小
batch_size = 64
# 上下文长度
block_size = 256

# 模型核心参数（nanoGPT基础架构）
n_layer = 6  # Transformer层数
n_head = 6   # 注意力头数
n_embd = 384 # 嵌入维度
dropout = 0.2# 随机失活率

# 优化器参数
learning_rate = 1e-3  # 学习率
max_iters = 5000      # 最大训练迭代次数
lr_decay_iters = 5000 # 学习率衰减迭代次数
min_lr = 1e-4         # 最小学习率
beta2 = 0.99          # Adam优化器beta2参数
warmup_iters = 100    # 学习率预热迭代次数
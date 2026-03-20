import os
import tiktoken
import numpy as np

# -------------------------- 核心配置：绝对路径指定天龙八部文本 --------------------------
# 替换成你自己的tianlong.txt绝对路径！
input_file_path = r'E:\CLASS\big modle\nano\nanoGPT\data\tianlong\tianlong.txt'
output_dir = r'E:\CLASS\big modle\nano\nanoGPT\data\tianlong'  # 输出目录（当前目录，即data/tianlong/）

# 创建输出目录（确保目录存在，exist_ok=True避免重复创建报错）
os.makedirs(output_dir, exist_ok=True)

# 读取原始文本
print(f"Reading input file: {input_file_path}")
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 用GPT-2的tokenizer编码文本（和nanoGPT训练逻辑一致）
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(text)  # 普通编码，不处理特殊token
tokens = np.array(tokens, dtype=np.uint16)

# 划分训练集（90%）和验证集（10%）
n = len(tokens)
train_tokens = tokens[:int(0.9*n)]
val_tokens = tokens[int(0.9*n):]

# 保存为二进制文件（模型训练必需）
train_path = os.path.join(output_dir, 'train.bin')
val_path = os.path.join(output_dir, 'val.bin')

with open(train_path, 'wb') as f:
    f.write(train_tokens.tobytes())
with open(val_path, 'wb') as f:
    f.write(val_tokens.tobytes())

# 打印统计信息，确认预处理成功
print(f"✅ Total tokens in tianlong.txt: {n}")
print(f"✅ Train set tokens: {len(train_tokens)} (90%)")
print(f"✅ Val set tokens: {len(val_tokens)} (10%)")
print(f"✅ Saved train.bin to: {train_path}")
print(f"✅ Saved val.bin to: {val_path}")
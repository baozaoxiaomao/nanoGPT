
# nanoGPT
![nanoGPT](assets/nanogpt.jpg)

---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```


Dependencies:
- `pytorch` (deep learning framework)
- `numpy` (numerical computing)
- `transformers` (for GPT-2 tokenizer/weights)
- `datasets` (for data loading)
- `tiktoken` (fast BPE tokenizer)
- `wandb` (optional, for logging)
- `tqdm` (progress bars)

---

# 📚 案例一：诗词&天龙八部文本生成实践
本实践基于nanoGPT框架，完成**古诗词生成**与**《天龙八部》武侠风格文本生成**两个任务，从数据准备、模型训练到推理采样全流程可复现，适配GPU/CPU环境，适合nanoGPT初学者入门。

## 一、实践目标
1. 基于5.8万首古诗词数据集，训练可自动生成诗词的GPT模型
2. 基于《天龙八部》全本小说文本，训练可生成武侠风格内容的GPT模型
3. 掌握nanoGPT的核心流程：数据预处理 → 配置文件编写 → 模型训练 → 推理采样

## 二、环境准备
1. 项目克隆（若未克隆）
```bash
git clone https://github.com/baozaoxiaomao/nanoGPT.git
cd nanoGPT
```

2. 依赖安装（已在上方 install 部分完成，若缺失可重新执行）

```
pip install torch numpy transformers datasets tiktoken tqdm
```
注：所有依赖版本建议 < 3.0，避免环境冲突；CPU 环境自动适配，无需额外配置。

## 三、数据准备
1. 数据集准备
古诗词数据集：tang_poet.txt（5.8 万首全唐诗 / 宋词，纯文本格式）
《天龙八部》数据集：tianlong.txt（金庸《天龙八部》全本小说，纯文本格式，约 124 万字符）
2. 创建数据目录
在项目data/目录下，分别为两个数据集创建专属文件夹：
```
# 古诗词数据目录
mkdir -p data/poemtext
# 天龙八部数据目录
mkdir -p data/tianlongtext
```
将tang_poet.txt放入data/poemtext/，tianlong.txt放入data/tianlongtext/。
3. 数据预处理（核心步骤）
为两个数据集分别编写prepare.py，完成文本切分、BPE 编码、二进制文件生成，生成训练 / 验证集train.bin/val.bin。
3.1 古诗词数据预处理
在data/poemtext/目录下新建prepare.py，写入以下代码：
```python

import os
import tiktoken
import numpy as np

# 加载古诗词数据集
input_file_path = 'tang_poet.txt'
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# 9:1 切分训练集/验证集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 使用GPT-2 BPE分词器编码
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"训练集token数: {len(train_ids):,}")
print(f"验证集token数: {len(val_ids):,}")

# 保存为uint16格式二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```

运行预处理脚本：
```
cd data/poemtext
python prepare.py
```
运行成功后，data/poemtext/目录下会生成train.bin和val.bin。

3.2 天龙八部数据预处理
在data/tianlongtext/目录下新建同名prepare.py，仅修改输入文件名为tianlong.txt，其余代码完全一致：
```python
import os
import tiktoken
import numpy as np

# 加载天龙八部数据集
input_file_path = 'tianlong.txt'
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# 9:1 切分训练集/验证集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 使用GPT-2 BPE分词器编码
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"训练集token数: {len(train_ids):,}")
print(f"验证集token数: {len(val_ids):,}")

# 保存为uint16格式二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```
运行脚本：
```
cd data/tianlongtext
python prepare.py
```
生成对应train.bin/val.bin，完成数据准备。

## 四、模型训练
1. 编写训练配置文件
在项目config/目录下，为两个数据集分别创建训练配置文件，定义模型结构、训练超参数。
1.1 古诗词模型配置：train_poemtext_char.py
```python

# 输出目录（模型权重保存路径）
out_dir = 'out-poemtext-char'
# 评估间隔
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
# 数据集名称（对应data/下的文件夹名）
dataset = 'poemtext'
# 训练批次/上下文窗口
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
# 模型结构参数
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
# 学习率/训练迭代数
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
# 设备（自动适配GPU/CPU，无需手动修改）
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False if device == 'cpu' else True
```
1.2 天龙八部模型配置：train_tianlongtext_char.py
仅修改out_dir和dataset，其余参数可复用（也可根据需求调优）：
```python

# 输出目录（模型权重保存路径）
out_dir = 'out-tianlongtext-char'
# 评估间隔
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
# 数据集名称（对应data/下的文件夹名）
dataset = 'tianlongtext'
# 训练批次/上下文窗口
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
# 模型结构参数
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
# 学习率/训练迭代数
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
# 设备（自动适配GPU/CPU，无需手动修改）
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False if device == 'cpu' else True
```
2. 执行训练命令
返回项目根目录，根据环境选择训练命令：
2.1 古诗词模型训练
GPU 环境（推荐，速度快）
```
python train.py config/train_poemtext_char.py
```
CPU 环境（适合无 GPU 设备，已调小参数）
```
python train.py config/train_poemtext_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```
2.2 天龙八部模型训练
GPU 环境
```
python train.py config/train_tianlongtext_char.py
```
CPU 环境
```
python train.py config/train_tianlongtext_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```
训练说明：训练过程中会输出迭代次数iter、损失值loss，若train loss和val loss持续下降，说明训练正常；训练完成后，模型权重ckpt.pt会保存在对应out_dir目录下。
## 五、模型推理 & 文本生成
训练完成后，使用sample.py进行推理，生成对应风格的文本内容。
1. 古诗词生成
GPU 环境
```
python sample.py --out_dir=out-poemtext-char
```
CPU 环境
```
python sample.py --out_dir=out-poemtext-char --device=cpu
```
运行后，控制台会直接输出模型生成的古诗词内容。
2. 天龙八部风格文本生成
GPU 环境
```
python sample.py --out_dir=out-tianlongtext-char
```
CPU 环境
```
python sample.py --out_dir=out-tianlongtext-char --device=cpu
```
运行后，控制台会输出武侠风格的小说内容。

## 六、核心参数说明（调优指南）
| 参数 | 含义 | 调优建议 |
|------|------|----------|
| `out_dir` | 模型权重、日志保存目录 | 不同数据集需区分目录，避免覆盖 |
| `dataset` | 数据集目录（对应`data/`下的文件夹） | 必须与预处理后的数据集目录一致 |
| `batch_size` | 单次训练批次大小 | CPU 环境建议调小（8-16），GPU 可适当增大 |
| `block_size` | 上下文窗口长度（参考前 N 个字符生成） | 越大模型上下文能力越强，显存占用越高 |
| `n_layer/n_head/n_embd` | 模型层数、注意力头数、嵌入维度 | 决定模型大小，参数越大效果越好，训练越慢 |
| `learning_rate` | 学习率 | 小模型建议 1e-3~5e-4，大模型建议 1e-4~5e-5 |
| `max_iters` | 最大训练迭代次数 | 迭代数越多，模型拟合越好，注意避免过拟合 |

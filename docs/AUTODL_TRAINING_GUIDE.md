# Civis Lucri-Faber AutoDL 训练指南

## 一、AutoDL环境准备

### 1.1 选择镜像

AutoDL镜像推荐：
- **PyTorch 2.0 + Python 3.10** (推荐)
- CUDA 11.8 / 12.0

### 1.2 GPU选择建议

| 训练规模 | 推荐GPU | 成本参考 |
|---------|--------|---------|
| 小规模测试 | RTX 3090 (24GB) | ~1.5元/小时 |
| 中等训练 | RTX 4090 (24GB) | ~2.5元/小时 |
| 大规模训练 | A100 (40GB/80GB) | ~10-15元/小时 |

### 1.3 连接实例后执行

```bash
# 更新系统
apt-get update && apt-get upgrade -y

# 检查CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 二、项目部署

### 2.1 克隆项目

```bash
# 方式1: 从本地打包上传 (推荐)
# 在本地执行:
cd D:/civis_lucri_faber
tar -czvf civis_lucri_faber.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='electron_app' --exclude='*.vmdk' .

# 上传到AutoDL后解压:
cd /root
tar -xzvf civis_lucri_faber.tar.gz
cd civis_lucri_faber

# 方式2: 从Git克隆 (如果有远程仓库)
git clone https://your-repo-url.git
cd civis_lucri_faber
```

### 2.2 安装依赖

```bash
# 创建虚拟环境 (可选但推荐)
conda create -n civis python=3.10 -y
conda activate civis

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn
pip install tqdm loguru python-dotenv

# 可选依赖
pip install pynvml  # GPU监控
pip install wandb   # 实验追踪
pip install tensorboard  # 可视化
```

### 2.3 验证安装

```bash
python -c "
import torch
import sys
sys.path.insert(0, '/root/civis_lucri_faber')
from core.curiosity import CuriosityEngine
from core.information_gain import TrueInformationGainCalculator
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print('Imports OK!')
"
```

---

## 三、训练脚本

### 3.1 基础训练命令

```bash
# 使用现有train_gpu.py
python train_gpu.py --epochs 100 --batch_size 32 --lr 1e-3 --output_dir ./checkpoints

# 从checkpoint恢复
python train_gpu.py --epochs 200 --resume ./checkpoints/best_model.pt
```

### 3.2 完整训练脚本 (推荐)

创建 `/root/civis_lucri_faber/train_autodl.py`:

```python
"""
Civis Lucri-Faber AutoDL训练脚本
=================================

Usage:
    python train_autodl.py --mode full --epochs 100
    python train_autodl.py --mode curiosity_only --epochs 50
    python train_autodl.py --mode info_gain --epochs 50
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 项目路径
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.curiosity import CuriosityEngine, LearnedNoveltyEngine
from core.information_gain import TrueInformationGainCalculator
from core.meta_learning import FirstOrderMAML
from core.basal_ganglia import BasalGangliaSystem
from core.sleep import MemoryReplayer


class CivisTrainer:
    """Civis完整训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练模式
        self.mode = args.mode
        
        # 初始化组件
        self._init_components()
        
        # 训练历史
        self.history = {
            'loss': [],
            'info_gain': [],
            'novelty': [],
            'gpu_memory': [],
            'epoch_time': [],
        }
        
    def _init_components(self):
        """初始化训练组件"""
        
        # 信息增益世界模型 (需要训练)
        self.info_gain = TrueInformationGainCalculator(
            state_dim=64,
            action_dim=16,
            latent_dim=32,
            lr=self.args.lr,
            device=self.device
        )
        
        # 好奇心引擎 (可选训练)
        self.curiosity = LearnedNoveltyEngine(
            vocab_size=100,
            embedding_dim=64,
            hidden_dim=64
        ).to(self.device)
        self.curiosity_optimizer = optim.Adam(self.curiosity.parameters(), lr=self.args.lr)
        
        # 元学习 MAML (需要训练)
        if self.mode in ['full', 'meta']:
            self.maml = FirstOrderMAML(
                model_dim=64,
                inner_lr=0.01,
                outer_lr=self.args.lr
            ).to(self.device)
        
        # 记忆回放系统 (睡眠期间训练)
        self.memory_replayer = MemoryReplayer(batch_size=self.args.batch_size)
        
        # 离线学习模型
        self.offline_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(self.device)
        self.offline_optimizer = optim.Adam(self.offline_model.parameters(), lr=self.args.lr)
        
        # 混合精度
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
    def generate_synthetic_data(self, n_samples=1000):
        """生成合成训练数据"""
        states = np.random.randn(n_samples, 64).astype(np.float32)
        actions = np.random.randint(0, 16, size=(n_samples,)).astype(np.int64)
        rewards = np.random.randn(n_samples, 1).astype(np.float32)
        next_states = np.random.randn(n_samples, 64).astype(np.float32)
        
        return states, actions, rewards, next_states
    
    def train_info_gain(self, states, actions, rewards, next_states):
        """训练信息增益世界模型"""
        total_loss = 0.0
        
        dataset = TensorDataset(
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
        )
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        
        for batch in loader:
            s, a, r, ns = batch
            s = s.to(self.device)
            a = a.to(self.device)
            r = r.to(self.device)
            ns = ns.to(self.device)
            
            # 训练世界模型
            result = self.info_gain.train_step()
            total_loss += result.get('loss', 0)
        
        return total_loss / len(loader)
    
    def train_curiosity(self, goal_pairs):
        """训练好奇心引擎"""
        if len(goal_pairs) < 2:
            return 0.0
        
        result = self.curiosity.train_step(goal_pairs)
        return result.get('loss', 0)
    
    def train_offline(self, memories):
        """睡眠期间离线训练"""
        for memory in memories:
            state_tensor = torch.FloatTensor(memory['state']).to(self.device)
            reward_tensor = torch.FloatTensor([memory['reward']]).to(self.device)
            
            pred = self.offline_model(state_tensor)
            loss = nn.functional.mse_loss(pred, reward_tensor)
            
            self.offline_optimizer.zero_grad()
            loss.backward()
            self.offline_optimizer.step()
            
        return loss.item()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        epoch_start = time.time()
        
        # 生成数据
        n_samples = self.args.batch_size * 10
        states, actions, rewards, next_states = self.generate_synthetic_data(n_samples)
        
        # 添加到记忆回放
        for i in range(n_samples):
            self.memory_replayer.add_experience(
                states[i], str(actions[i]), rewards[i].item(), next_states[i]
            )
        
        epoch_losses = {}
        
        # 1. 信息增益训练
        if self.mode in ['full', 'info_gain']:
            ig_loss = self.train_info_gain(states, actions, rewards, next_states)
            epoch_losses['info_gain_loss'] = ig_loss
        
        # 2. 好奇心训练
        if self.mode in ['full', 'curiosity']:
            goal_pairs = [(states[i], next_states[i]) for i in range(min(50, n_samples))]
            c_loss = self.train_curiosity(goal_pairs)
            epoch_losses['curiosity_loss'] = c_loss
        
        # 3. 离线学习 (模拟睡眠)
        if self.mode in ['full', 'offline']:
            sampled_memories = self.memory_replayer.sample(self.args.batch_size)
            memory_data = [{'state': m.state, 'reward': m.reward} for m in sampled_memories]
            o_loss = self.train_offline(memory_data)
            epoch_losses['offline_loss'] = o_loss
        
        # 记录GPU状态
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            self.history['gpu_memory'].append(gpu_mem)
            torch.cuda.reset_peak_memory_stats()
        
        epoch_time = time.time() - epoch_start
        self.history['epoch_time'].append(epoch_time)
        
        return epoch_losses
    
    def validate(self):
        """验证"""
        with torch.no_grad():
            states, actions, rewards, next_states = self.generate_synthetic_data(100)
            
            # 计算信息增益
            total_ig = 0.0
            for i in range(100):
                reward_obj = self.info_gain.compute_reward(
                    states[i], int(actions[i]), 
                    rewards[i].item(), next_states[i],
                    use_intrinsic=True
                )
                total_ig += reward_obj.information_gain
            
            return total_ig / 100
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'args': vars(self.args),
            'info_gain_state': self.info_gain.world_model.state_dict(),
            'info_gain_optimizer': self.info_gain.optimizer.state_dict(),
            'curiosity_state': self.curiosity.state_dict(),
            'offline_model_state': self.offline_model.state_dict(),
            'history': self.history,
        }
        
        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()
        
        # 保存最新
        latest_path = self.output_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Best model saved at epoch {epoch}")
        
        # 定期保存
        if epoch % 10 == 0:
            epoch_path = self.output_dir / f'model_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, path):
        """加载checkpoint"""
        if not path.exists():
            print(f"[Warning] Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.info_gain.world_model.load_state_dict(checkpoint['info_gain_state'])
        self.info_gain.optimizer.load_state_dict(checkpoint['info_gain_optimizer'])
        self.curiosity.load_state_dict(checkpoint['curiosity_state'])
        self.offline_model.load_state_dict(checkpoint['offline_model_state'])
        
        if 'scaler_state' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        self.history = checkpoint.get('history', self.history)
        
        print(f"[Checkpoint] Loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[History] Saved to {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Civis Lucri-Faber AutoDL Training')
    
    # 训练模式
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'info_gain', 'curiosity', 'offline', 'meta'],
                        help='Training mode')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./autodl_checkpoints')
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("\n" + "="*60)
    print("Civis Lucri-Faber AutoDL Training")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Output: {args.output_dir}")
    
    # 创建训练器
    trainer = CivisTrainer(args)
    
    # Resume
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        start_epoch = trainer.load_checkpoint(resume_path) + 1
    
    # 训练循环
    best_val_ig = 0.0
    total_start = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # 训练
        losses = trainer.train_epoch(epoch)
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        print(f"  [Losses] {loss_str}")
        
        # 验证
        if epoch % args.log_interval == 0:
            val_ig = trainer.validate()
            print(f"  [Val] Info Gain: {val_ig:.4f}")
            trainer.history['info_gain'].append(val_ig)
            
            # 保存最佳
            is_best = val_ig > best_val_ig
            if is_best:
                best_val_ig = val_ig
            
            trainer.save_checkpoint(epoch, is_best=is_best)
        
        # GPU状态
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / 1e9
            print(f"  [GPU] Memory: {current_mem:.2f} GB")
        
        epoch_time = trainer.history['epoch_time'][-1]
        print(f"  [Time] {epoch_time:.2f}s")
    
    # 完成
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best Info Gain: {best_val_ig:.4f}")
    
    # 保存历史
    trainer.save_history()
    
    # 最终checkpoint
    trainer.save_checkpoint(args.epochs)


if __name__ == '__main__':
    main()
```

---

## 四、训练命令

### 4.1 不同模式训练

```bash
# 完整训练 (所有模块)
python train_autodl.py --mode full --epochs 100 --batch_size 64 --lr 1e-3

# 仅信息增益世界模型
python train_autodl.py --mode info_gain --epochs 50

# 仅好奇心引擎
python train_autodl.py --mode curiosity --epochs 50

# 离线学习 (模拟睡眠)
python train_autodl.py --mode offline --epochs 30

# 恢复训练
python train_autodl.py --mode full --epochs 200 --resume ./autodl_checkpoints/latest_model.pt
```

### 4.2 资源监控命令

```bash
# GPU监控 (每1秒刷新)
watch -n 1 nvidia-smi

# Python内监控
python -c "
import torch
import time
while True:
    mem = torch.cuda.memory_allocated() / 1e9
    max_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'Current: {mem:.2f}GB | Peak: {max_mem:.2f}GB')
    time.sleep(1)
"
```

---

## 五、保存与下载

### 5.1 训练完成后打包

```bash
cd /root/civis_lucri_faber

# 打包checkpoints
tar -czvf checkpoints.tar.gz autodl_checkpoints/

# 打包完整训练结果
tar -czvf training_results.tar.gz \
    autodl_checkpoints/ \
    training_history.json \
    civis_metrics.csv
```

### 5.2 下载到本地

```bash
# 方式1: AutoDL文件管理器下载
# 登录AutoDL控制台 → 文件管理 → 找到文件 → 下载

# 方式2: SCP下载 (本地执行)
scp -P <端口> root@<地址>:/root/civis_lucri_faber/training_results.tar.gz ./autodl_results/

# 方式3: 使用AutoDL的OSS存储
# 上传到OSS后从控制台下载
```

---

## 六、进阶配置

### 6.1 WandB实验追踪

```bash
# 安装
pip install wandb

# 登录
wandb login

# 在训练脚本中添加
import wandb
wandb.init(project="civis-lucri-faber", name="autodl-run-001")

# 训练循环中记录
wandb.log({
    "loss": loss,
    "info_gain": val_ig,
    "gpu_memory": gpu_mem,
    "epoch": epoch
})
```

### 6.2 TensorBoard可视化

```bash
# 安装
pip install tensorboard

# 在训练脚本中添加
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/autodl')

# 训练循环中记录
writer.add_scalar('Loss/info_gain', loss, epoch)
writer.add_scalar('Val/info_gain', val_ig, epoch)

# 启动TensorBoard
tensorboard --logdir ./runs --port 6006

# AutoDL端口映射
# 控制台 → 自定义服务 → 开启6006端口 → 通过外部链接访问
```

### 6.3 多GPU训练

```python
# 在train_autodl.py中添加
import torch.distributed as dist
import torch.nn.parallel.distributed as DDP

# 初始化分布式
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 包装模型
model = DDP(model, device_ids=[local_rank])

# 运行命令
torchrun --nproc_per_node=2 train_autodl.py --mode full
```

---

## 七、成本估算

### 7.1 小规模训练 (测试)

| 配置 | 预估时间 | 预估成本 |
|------|---------|---------|
| RTX 3090, 50 epochs | ~30分钟 | ~0.75元 |
| RTX 4090, 50 epochs | ~20分钟 | ~0.83元 |

### 7.2 中等规模训练

| 配置 | 预估时间 | 预估成本 |
|------|---------|---------|
| RTX 3090, 200 epochs | ~2小时 | ~3元 |
| RTX 4090, 200 epochs | ~1.5小时 | ~3.75元 |

### 7.3 大规模训练

| 配置 | 预估时间 | 预估成本 |
|------|---------|---------|
| A100 40GB, 500 epochs | ~3小时 | ~30元 |
| A100 80GB, 1000 epochs | ~6小时 | ~90元 |

---

## 八、常见问题

### Q1: CUDA内存不足

```bash
# 减小batch_size
python train_autodl.py --batch_size 16

# 清理缓存
torch.cuda.empty_cache()
```

### Q2: 导入错误

```bash
# 确保路径正确
cd /root/civis_lucri_faber
export PYTHONPATH=/root/civis_lucri_faber:$PYTHONPATH
```

### Q3: 训练中断恢复

```bash
# 使用resume参数
python train_autodl.py --resume ./autodl_checkpoints/latest_model.pt
```

### Q4: 数据上传慢

```bash
# 使用压缩
tar -czvf project.tar.gz --exclude='*.pyc' --exclude='__pycache__' .

# 或使用AutoDL的数据盘 (/root/autodl-tmp)
# 数据盘是SSD，读写更快
```

---

## 九、推荐训练流程

```bash
# Day 1: 环境测试 + 小规模训练
python train_autodl.py --mode info_gain --epochs 10 --batch_size 32

# Day 2: 信息增益完整训练
python train_autodl.py --mode info_gain --epochs 100 --batch_size 64

# Day 3: 好奇心引擎训练
python train_autodl.py --mode curiosity --epochs 50

# Day 4: 离线学习 (睡眠模拟)
python train_autodl.py --mode offline --epochs 30

# Day 5: 完整集成训练
python train_autodl.py --mode full --epochs 100 --resume ./autodl_checkpoints/latest_model.pt
```

---

**祝训练顺利！有问题随时交流。**
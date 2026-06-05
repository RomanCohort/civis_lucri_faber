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
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

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

        return total_loss / len(loader) if len(loader) > 0 else 0

    def train_curiosity(self, goal_pairs):
        """训练好奇心引擎"""
        if len(goal_pairs) < 2:
            return 0.0

        result = self.curiosity.train_step(goal_pairs)
        return result.get('loss', 0)

    def train_offline(self, memories):
        """睡眠期间离线训练"""
        total_loss = 0.0
        count = 0

        for memory in memories:
            state_tensor = torch.FloatTensor(memory['state']).to(self.device)
            reward_tensor = torch.FloatTensor([memory['reward']]).to(self.device)

            pred = self.offline_model(state_tensor)
            loss = nn.functional.mse_loss(pred, reward_tensor)

            self.offline_optimizer.zero_grad()
            loss.backward()
            self.offline_optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0

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
            return 0

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
        loaded_epoch = trainer.load_checkpoint(resume_path)
        if loaded_epoch > 0:
            start_epoch = loaded_epoch + 1

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
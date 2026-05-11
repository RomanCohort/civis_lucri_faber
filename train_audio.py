"""
听觉皮层训练 - 对比学习版
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class AudioDataset(Dataset):
    def __init__(self, n_samples=200):
        # 模拟音频数据 - 不同类型有不同特征
        self.audio = []
        self.labels = []
        for i in range(n_samples):
            label = i % 4  # 0=平静, 1=开心, 2=悲伤, 3=愤怒
            # 生成不同情感的音频特征
            if label == 0:  # 平静 - 低频
                audio = np.sin(np.linspace(0, 10, 8000)) * 0.1 + np.random.randn(8000) * 0.02
            elif label == 1:  # 开心 - 高频快
                audio = np.sin(np.linspace(0, 30, 8000)) * 0.15 + np.random.randn(8000) * 0.03
            elif label == 2:  # 悲伤 - 中频慢
                audio = np.sin(np.linspace(0, 5, 8000)) * 0.1 + np.random.randn(8000) * 0.02
            else:  # 愤怒 - 噪声
                audio = np.random.randn(8000) * 0.15
            self.audio.append(audio.astype(np.float32))
            self.labels.append(label)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return {
            'audio': torch.tensor(self.audio[idx]),
            'label': self.labels[idx],
        }


def contrastive_loss(z_i, z_j, temperature=0.1):
    """SimCLR对比损失"""
    batch_size = z_i.size(0)
    # 归一化
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    # 相似度矩阵
    sim = torch.cat([z_i, z_j], dim=0) @ torch.cat([z_j, z_i], dim=0).T / temperature
    # 对角是正样本
    labels = torch.arange(batch_size).to(z_i.device)
    labels = torch.cat([labels, labels], dim=0)
    # 对比损失
    loss = F.cross_entropy(sim, labels)
    return loss


def train():
    print("=== 听觉皮层训练 (对比学习) ===")

    base = os.path.dirname(__file__)
    audit = load(os.path.join(base, 'core', 'auditory_cortex.py'), 'auditory_cortex')

    dataset = AudioDataset(200)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"数据: {len(dataset)}")

    # 模型
    model = audit.create_auditory_cortex(n_filters=8)

    # 投影头 (对比学习)
    projection = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

    opt = torch.optim.AdamW(list(model.parameters()) + list(projection.parameters()), lr=5e-4)

    # 训练
    model.train()
    for epoch in range(5):
        losses = []
        for batch in loader:
            audio = batch['audio']

            # 两个增广视图
            aug1 = audio + torch.randn_like(audio) * 0.05
            aug2 = audio + torch.randn_like(audio) * 0.05

            # 编码
            feat1 = model(aug1)['features']
            feat2 = model(aug2)['features']

            # 投影
            z1 = projection(feat1)
            z2 = projection(feat2)

            # 对比损失
            loss = contrastive_loss(z1, z2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}: loss={sum(losses)/len(losses):.4f}")

    # 保存
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/auditory.pt')
    print("保存到 checkpoints/auditory.pt")
    print("训练完成!")


if __name__ == "__main__":
    train()
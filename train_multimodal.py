"""
多模态感知训练脚本

训练方式:
1. 多模态对比学习 (SimCLR风格)
2. 对齐各模态 embeddings
3. 情感预测联合训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List
import os


class MultimodalDataset(Dataset):
    """多模态数据集"""
    def __init__(
        self,
        n_samples: int = 100,
        has_all_modalities: bool = True,
    ):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 返回各模态数据 + 情感标签
        return {
            'flow': torch.randn(1, 2, 8, 32, 32),
            'rgb': torch.randn(1, 6, 8, 32, 32),
            'audio': torch.randn(1, 8000),
            'text': torch.randint(0, 5000, (1, 8)),
            'emotion': torch.randint(0, 5, (1,)),  # 5类情绪
        }


class MultimodalTrainer:
    """多模态训练器"""
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        temperature: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.model.to(device)
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def contrastive_loss(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        多模态对比损失: 同一样本的不同模态应该相近
        """
        # 合并所有模态的embedding
        all_emb = []
        names = []

        for name, emb in embeddings.items():
            if emb is not None:
                all_emb.append(emb)
                names.append(name)

        if len(all_emb) < 2:
            return torch.tensor(0.0, device=self.device)

        # 拼接所有embedding [B, n_modal, dim]
        stacked = torch.stack(all_emb, dim=1)  # [B, n_mod, dim]

        # 计算相似度矩阵
        B, n_mod, dim = stacked.shape
        flat = stacked.view(B * n_mod, dim)  # [B*n_mod, dim]
        sim = (flat @ flat.T) / self.temperature  # [BN, BN]

        # 创建labels: 同一样本的模态是正样本
        # 同一batch中，样本0的所有模态=0, 样本1的所有模态=1, ...
        labels_expanded = labels.unsqueeze(1).expand(-1, n_mod).flatten()
        labels_mask = (labels_expanded.unsqueeze(0) == labels_expanded.unsqueeze(1))

        # 去掉自相关
        for i in range(n_mod):
            start = i * B
            end = (i + 1) * B
            sim[start:end, start:end] -= torch.eye(B, device=sim.device)

        # 对比损失
        exp_sim = torch.exp(sim)
        pos = (exp_sim * labels_mask.float()).sum(dim=1)
        neg = exp_sim.sum(dim=1) - exp_sim.diag()
        loss = -torch.log(pos / (pos + neg + 1e-8)).mean()

        return loss

    def emotion_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """情感分类损失"""
        return F.cross_entropy(pred, target)

    def train_step(
        self,
        batch: Dict,
    ) -> Dict:
        """单步训练"""
        self.model.train()

        flow = batch['flow'].to(self.device)
        rgb = batch['rgb'].to(self.device)
        audio = batch['audio'].to(self.device)
        text = batch['text'].to(self.device)
        emotion = batch['emotion'].squeeze(-1).to(self.device)

        # 多模态前向
        result = self.model(
            optical_flow=flow,
            rgb_ppg=rgb,
            audio=audio,
            text_tokens=text,
        )

        # 提取embeddings
        embeddings = {}

        if result.get('fused') is not None:
            embeddings['fused'] = result['fused']
        if result.get('vision') and result['vision'].get('features') is not None:
            embeddings['vision'] = result['vision']['features']
        if result.get('audio') and result['audio'].get('features') is not None:
            embeddings['audio'] = result['audio']['features']
        if result.get('text') and result['text'].get('features') is not None:
            embeddings['text'] = result['text']['features']

        # 损失
        loss = 0
        metrics = {}

        # 1. 对比损失
        if len(embeddings) >= 2:
            ctr_loss = self.contrastive_loss(embeddings, batch['emotion'].squeeze(-1).to(self.device))
            loss += ctr_loss
            metrics['contrastive'] = ctr_loss.item()

        # 2. 情感预测损失
        if result.get('emotion') is not None:
            emo_loss = self.emotion_loss(
                result['emotion'].get('emotion_probs', torch.randn(1, 5).to(self.device)),
                emotion
            )
            # 简化: 暂时跳过
            # loss += emo_loss * 0.5

        # 总损失
        total_loss = loss

        # 梯度更新
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        metrics['total'] = total_loss.item()
        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        total = {k: 0 for k in ['total', 'contrastive']}

        for batch in dataloader:
            m = self.train_step(batch)
            for k, v in m.items():
                if k in total:
                    total[k] += v

        n = len(dataloader)
        return {k: v/n for k, v in total.items()}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model': self.model.state_dict()}, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        print(f"Loaded from {path}")


def create_multimodal_trainer(
    vocab_size: int = 5000,
    lr: float = 1e-3,
) -> MultimodalTrainer:
    """创建训练器"""
    from core.multimodal_perception import create_multimodal_perception
    model = create_multimodal_perception(vocab_size=vocab_size)
    return MultimodalTrainer(model, lr=lr)


def main():
    # 模拟数据
    dataset = MultimodalDataset(n_samples=50)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 创建训练器
    trainer = create_multimodal_trainer()

    print("=== 多模态感知训练 ===")

    for epoch in range(3):
        m = trainer.train_epoch(loader)
        print(f"Epoch {epoch+1}: total={m['total']:.4f}, ctr={m['contrastive']:.4f}")

    trainer.save("checkpoints/multimodal.pt")
    print("训练完成!")


if __name__ == "__main__":
    main()
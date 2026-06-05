"""
Civis Lucri-Faber 真实数据训练脚本
===================================

使用 ShareGPT + GoEmotions 数据集训练所有可训练模块

Usage:
    # 完整训练 (所有模块)
    python train_real.py --mode full --epochs 50

    # 单模块训练
    python train_real.py --mode info_gain --epochs 30
    python train_real.py --mode curiosity --epochs 30
    python train_real.py --mode emotion --epochs 30
    python train_real.py --mode meta --epochs 20

    # 从checkpoint恢复
    python train_real.py --mode full --epochs 100 --resume ./checkpoints/latest.pt
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.curiosity import LearnedNoveltyEngine
from core.information_gain import TrueInformationGainCalculator
from core.meta_learning import FirstOrderMAML, Task
from core.limbic import AmygdalaNucleus
from core.language_cortex import LanguageCortex
from core.sleep import MemoryReplayer

# ─── Sentence-Transformers 文本编码器 ────────────────────────────────

class TextEncoder(nn.Module):
    """用sentence-transformers生成语义embedding，投影到64维"""
    def __init__(self, model_name='all-MiniLM-L6-v2', output_dim=64):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(model_name)
            self.embed_dim = self.sbert.get_sentence_embedding_dimension()  # 384
            self.sbert.eval()
            # 冻结sbert参数
            for param in self.sbert.parameters():
                param.requires_grad = False
            print(f"[TextEncoder] Loaded {model_name}, dim={self.embed_dim}")
        except ImportError:
            print("[TextEncoder] sentence-transformers not found, using fallback")
            self.sbert = None
            self.embed_dim = 384
        # 可训练投影层
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )
        # 缓存
        self._cache = {}

    def encode_text(self, text: str) -> np.ndarray:
        """编码单条文本 → 384维embedding"""
        if text in self._cache:
            return self._cache[text]
        if self.sbert is not None:
            emb = self.sbert.encode(text, convert_to_numpy=True)
        else:
            # fallback: 手工特征填充到384维
            emb = np.zeros(self.embed_dim, dtype=np.float32)
            for c in text.lower()[:200]:
                idx = ord(c) % self.embed_dim
                emb[idx] += 1.0
            emb /= max(len(text), 1)
        self._cache[text] = emb
        return emb

    def encode_batch(self, texts: list) -> np.ndarray:
        """批量编码"""
        uncached = [t for t in texts if t not in self._cache]
        if uncached and self.sbert is not None:
            embeddings = self.sbert.encode(uncached, convert_to_numpy=True, batch_size=64)
            for t, emb in zip(uncached, embeddings):
                self._cache[t] = emb
        return np.array([self._cache.get(t, np.zeros(self.embed_dim, dtype=np.float32)) for t in texts])

    def forward(self, text: str) -> torch.Tensor:
        """文本 → 64维神经状态 (可训练)"""
        emb = self.encode_text(text)
        emb_tensor = torch.FloatTensor(emb).unsqueeze(0)  # [1, 384]
        return self.projector(emb_tensor).squeeze(0)  # [64]

    def forward_tensor(self, emb: torch.Tensor) -> torch.Tensor:
        """384维embedding → 64维 (批量)"""
        return self.projector(emb)  # [B, 64]


# ─── GoEmotions 标签映射 ───────────────────────────────────────────
GO_EMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral',
]

# 映射到CLF的5类基础情绪: joy, sadness, anger, fear, neutral (与AmygdalaNucleus一致)
BASIC_EMOTION_MAP = {
    'admiration': 'joy', 'amusement': 'joy', 'approval': 'joy',
    'caring': 'joy', 'excitement': 'joy', 'gratitude': 'joy',
    'joy': 'joy', 'love': 'joy', 'optimism': 'joy', 'pride': 'joy',
    'relief': 'joy', 'surprise': 'neutral', 'curiosity': 'neutral',
    'realization': 'neutral',
    'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
    'disgust': 'anger',
    'fear': 'fear', 'nervousness': 'fear', 'embarrassment': 'fear',
    'sadness': 'sadness', 'disappointment': 'sadness', 'grief': 'sadness',
    'remorse': 'sadness',
    'confusion': 'sadness', 'desire': 'joy',
    'neutral': 'neutral',
}
BASIC_EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'neutral']


def text_to_state(text, dim=64):
    """文本→64维状态向量 (fallback: 手工特征)"""
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    words = text.lower().split()
    for c in text.lower():
        if 'a' <= c <= 'z':
            vec[ord(c) - ord('a')] += 1.0
    total_chars = sum(vec[:26]) + 1e-8
    vec[:26] /= total_chars
    emo_kw = {
        26: ['happy','love','great','good','joy','excited','wonderful','amazing','glad','pleased'],
        27: ['sad','miss','lost','cry','pain','sorry','grief','disappointed','lonely','hurt'],
        28: ['angry','hate','mad','furious','annoyed','disgusted','rage','irritated','frustrated'],
        29: ['fear','scared','worry','afraid','anxious','nervous','terrified','panic','dread'],
        30: ['okay','fine','normal','maybe','alright','neutral','whatever','hmm'],
    }
    for idx, kws in emo_kw.items():
        for w in words:
            if w.strip('.,!?;:"\'') in kws:
                vec[idx] += 1.0
    vec[26:31] /= max(len(words), 1)
    common = ['i','you','the','is','not','have','do','be','what','how',
              'why','can','will','would','should','feel','think','know','want','like']
    for w in words:
        w2 = w.strip('.,!?;:"\'')
        if w2 in common:
            vec[31 + common.index(w2)] += 1.0
    vec[31:51] /= max(len(words), 1)
    vec[51] = min(len(text) / 200.0, 1.0)
    vec[52] = min(len(words) / 40.0, 1.0)
    vec[53] = text.count('!') / max(len(text), 1)
    vec[54] = text.count('?') / max(len(text), 1)
    vec[55] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    vec[56] = min(sum(len(w) for w in words) / max(len(words), 1) / 8.0, 1.0)
    vec[57] = len(set(words)) / max(len(words), 1)
    vec[58] = min(len([w for w in words if len(w) > 6]) / max(len(words), 1) * 5, 1.0)
    vec[59] = 1.0 if any(w.strip('.,!?;:"\'') in ['no','never','dont','can\'t','won\'t'] for w in words) else 0.0
    vec[60] = 1.0 if text.endswith('?') else 0.0
    vec[61] = 1.0 if text.endswith('!') else 0.0
    vec[62] = min(max(len(words), 0) / 20.0, 1.0)
    vec[63] = min(len([w for w in words if w.isupper()]) / max(len(words), 1) * 3, 1.0)
    return vec


def text_to_tokens(text, vocab_size=10000, max_len=32):
    """文本→token序列 (字符级hash)"""
    tokens = []
    for ch in text[:max_len]:
        tokens.append(ord(ch) % vocab_size)
    while len(tokens) < max_len:
        tokens.append(0)  # padding
    return torch.LongTensor(tokens[:max_len])


class RealDataTrainer:
    """使用真实数据集的训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = args.mode

        # 加载数据集
        self._load_datasets()

        # 初始化组件
        self._init_components()

        # 训练历史
        self.history = {
            'loss': [], 'info_gain': [], 'emotion_acc': [],
            'val_emotion_acc': [], 'language_loss': [],
            'gpu_memory': [], 'epoch_time': [],
        }

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def _load_datasets(self):
        """加载本地数据集"""
        data_dir = PROJECT_ROOT / 'data'

        # GoEmotions (切分训练/验证 80/20)
        go_path = data_dir / 'go_emotions'
        if go_path.exists():
            go_full = load_from_disk(str(go_path))
            split = go_full.train_test_split(test_size=0.2, seed=42)
            self.go_emotions = split['train']
            self.go_emotions_val = split['test']
            print(f"[Data] GoEmotions: {len(self.go_emotions)} train, {len(self.go_emotions_val)} val")
        else:
            print(f"[Warning] GoEmotions not found at {go_path}")
            self.go_emotions = None
            self.go_emotions_val = None

        # ShareGPT
        sg_path = data_dir / 'sharegpt'
        if sg_path.exists():
            self.sharegpt = load_from_disk(str(sg_path))
            print(f"[Data] ShareGPT loaded: {len(self.sharegpt)} samples")
        else:
            print(f"[Warning] ShareGPT not found at {sg_path}")
            self.sharegpt = None

    def _init_components(self):
        """初始化训练组件"""
        lr = self.args.lr

        # 1. 信息增益世界模型
        self.info_gain = TrueInformationGainCalculator(
            state_dim=64, action_dim=16, latent_dim=32,
            lr=lr, device=self.device
        )

        # 2. 好奇心引擎
        self.curiosity = LearnedNoveltyEngine(
            vocab_size=10000, embedding_dim=64, hidden_dim=128
        ).to(self.device)
        self.curiosity_optimizer = optim.Adam(self.curiosity.parameters(), lr=lr)

        # 3. 情绪识别 (从AmygdalaNucleus提取)
        self.emotion_net = AmygdalaNucleus(input_dim=64).to(self.device)
        self.emotion_optimizer = optim.Adam(self.emotion_net.parameters(), lr=lr)

        # 3.5 语言皮层
        self.language_cortex = LanguageCortex().to(self.device)
        self.language_optimizer = optim.Adam(self.language_cortex.parameters(), lr=lr)

        # 4. 元学习 MAML
        if self.mode in ['full', 'meta']:
            self.maml = FirstOrderMAML(
                input_dim=64, output_dim=5,
                hidden_dim=64, inner_lr=0.01, outer_lr=lr
            ).to(self.device)

        # 5. 记忆回放 (离线学习)
        self.memory_replayer = MemoryReplayer(batch_size=self.args.batch_size)
        self.offline_model = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        ).to(self.device)
        self.offline_optimizer = optim.Adam(self.offline_model.parameters(), lr=lr)

        # 0. 文本编码器 (sentence-transformers → 64维)
        self.text_encoder = TextEncoder().to(self.device)
        self.text_encoder_optimizer = optim.Adam(
            self.text_encoder.projector.parameters(), lr=lr
        )

    # ─── GoEmotions 情绪训练 ──────────────────────────────────────

    def _precompute_sbert_embeddings(self):
        """预计算所有SBERT embedding (只执行一次)"""
        if self.go_emotions is not None:
            print("[TextEncoder] Pre-computing GoEmotions train embeddings...")
            self.go_train_embeddings = self.text_encoder.encode_batch(
                list(self.go_emotions['text'])
            )
            print(f"  Train: {len(self.go_train_embeddings)} embeddings done")

        if self.go_emotions_val is not None:
            print("[TextEncoder] Pre-computing GoEmotions val embeddings...")
            self.go_val_embeddings = self.text_encoder.encode_batch(
                list(self.go_emotions_val['text'])
            )
            print(f"  Val: {len(self.go_val_embeddings)} embeddings done")

    def prepare_emotion_data(self):
        """从GoEmotions准备情绪训练数据 (使用预计算的SBERT embedding + label smoothing)"""
        if self.go_emotions is None:
            return None, None

        # 使用预计算的embedding (避免每个epoch重复编码)
        if not hasattr(self, 'go_train_embeddings'):
            self._precompute_sbert_embeddings()

        embeddings = self.go_train_embeddings
        labels = self.go_emotions['labels']

        # 只在第一次计算标签目标 (带label smoothing)
        if not hasattr(self, '_emotion_targets'):
            SMOOTH = 0.15
            emotion_targets = []
            valence_targets = []
            arousal_targets = []

            valence_map = {'joy': 0.8, 'neutral': 0.0, 'sadness': -0.7, 'anger': -0.6, 'fear': -0.5}
            arousal_map = {'joy': 0.6, 'neutral': 0.3, 'sadness': 0.3, 'anger': 0.9, 'fear': 0.85}

            for lbls in labels:
                basic_counts = Counter()
                for lbl in lbls:
                    emo_name = GO_EMOTIONS_LABELS[lbl] if lbl < len(GO_EMOTIONS_LABELS) else 'neutral'
                    basic = BASIC_EMOTION_MAP.get(emo_name, 'neutral')
                    basic_counts[basic] += 1

                total_votes = sum(basic_counts.values())
                # 软标签: 票数比例 + 均匀平滑
                label = np.ones(5, dtype=np.float32) * SMOOTH / 5
                for emo, count in basic_counts.items():
                    idx = BASIC_EMOTIONS.index(emo)
                    label[idx] += (1 - SMOOTH) * count / total_votes
                emotion_targets.append(label)

                # 加权效价/唤醒度
                val = sum(valence_map[BASIC_EMOTIONS[i]] * label[i] for i in range(5))
                aro = sum(arousal_map[BASIC_EMOTIONS[i]] * label[i] for i in range(5))
                valence_targets.append([val])
                arousal_targets.append([aro])

            self._emotion_targets = np.array(emotion_targets)
            self._valence_targets = np.array(valence_targets)
            self._arousal_targets = np.array(arousal_targets)

        return (embeddings, self._emotion_targets, self._valence_targets, self._arousal_targets)

    def train_emotion(self, epoch):
        """训练情绪识别网络 (GoEmotions + SBERT)"""
        data = self.prepare_emotion_data()
        if data is None:
            return 0.0, 0.0

        embeddings, emo_targets, val_targets, aro_targets = data
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        n = len(embeddings)

        # 随机采样
        n_samples = min(self.args.batch_size * 10, n)
        indices = np.random.permutation(n)[:n_samples]

        # SBERT 384维 → projector → 64维 → emotion_net
        batch_emb = torch.FloatTensor(embeddings[indices]).to(self.device)
        batch_emo = torch.FloatTensor(emo_targets[indices]).to(self.device)
        batch_val = torch.FloatTensor(val_targets[indices]).to(self.device)
        batch_aro = torch.FloatTensor(aro_targets[indices]).to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0

        # Mini-batch
        for i in range(0, n_samples, self.args.batch_size):
            emb = batch_emb[i:i+self.args.batch_size]
            e = batch_emo[i:i+self.args.batch_size]
            v = batch_val[i:i+self.args.batch_size]
            a = batch_aro[i:i+self.args.batch_size]

            if len(emb) < 2:
                continue

            # SBERT 384维 → projector → 64维
            state = self.text_encoder.forward_tensor(emb)  # [B, 64]

            # 64维 → emotion/valence/arousal
            emo_pred = self.emotion_net.emotion_net(state)
            val_pred = self.emotion_net.valence_net(state)
            aro_pred = self.emotion_net.arousal_net(state)

            emo_loss = F.mse_loss(emo_pred, e)
            val_loss = F.mse_loss(val_pred.squeeze(), v.squeeze())
            aro_loss = F.mse_loss(aro_pred.squeeze(), a.squeeze())

            loss = emo_loss + 0.5 * val_loss + 0.5 * aro_loss

            # 同时更新 projector 和 emotion_net
            self.emotion_optimizer.zero_grad()
            self.text_encoder_optimizer.zero_grad()
            loss.backward()
            self.emotion_optimizer.step()
            self.text_encoder_optimizer.step()

            total_loss += loss.item()

            pred_idx = emo_pred.argmax(dim=-1)
            true_idx = e.argmax(dim=-1)
            correct += (pred_idx == true_idx).sum().item()
            total += len(emb)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ─── 语言皮层训练 ──────────────────────────────────────────────

    def train_language_from_sharegpt(self, epoch):
        """用ShareGPT对话训练语言皮层"""
        if self.sharegpt is None:
            return 0.0

        n_samples = min(self.args.batch_size * 5, len(self.sharegpt))
        indices = np.random.permutation(len(self.sharegpt))[:n_samples]

        total_loss = 0.0
        count = 0

        for idx in indices:
            conv = self.sharegpt[int(idx)]['conversations']
            if len(conv) < 2:
                continue

            for i in range(len(conv) - 1):
                cur_text = conv[i]['value'][:200]
                next_text = conv[i+1]['value'][:200]

                try:
                    # tokens → LanguageCortex
                    cur_tokens = self.language_cortex._text_to_tokens(cur_text).to(self.device)
                    lc_out = self.language_cortex(cur_tokens)
                    lc_features = lc_out['features']  # [1, 256]

                    # 目标: 下一轮的state
                    next_state = torch.FloatTensor(
                        text_to_state(next_text)
                    ).to(self.device)

                    proj = lc_features[:, :64]
                    loss = F.mse_loss(proj.squeeze(0), next_state)

                    self.language_optimizer.zero_grad()
                    loss.backward()
                    self.language_optimizer.step()

                    total_loss += loss.item()
                    count += 1
                except Exception:
                    continue

        return total_loss / max(count, 1)

    # ─── ShareGPT 信息增益训练 ──────────────────────────────────────

    def train_info_gain_from_sharegpt(self, epoch):
        """用ShareGPT对话训练信息增益世界模型"""
        if self.sharegpt is None:
            return 0.0

        # 采样对话
        n_samples = min(self.args.batch_size * 10, len(self.sharegpt))
        indices = np.random.permutation(len(self.sharegpt))[:n_samples]

        total_loss = 0.0
        count = 0

        for idx in indices:
            conv = self.sharegpt[int(idx)]['conversations']
            if len(conv) < 2:
                continue

            # 将对话轮次转化为 (state, action, reward, next_state)
            for i in range(len(conv) - 1):
                state = text_to_state(conv[i]['value'])
                next_state = text_to_state(conv[i+1]['value'])
                action = hash(conv[i]['from']) % 16
                reward = 1.0 if conv[i+1]['from'] == 'gpt' else 0.5

                self.info_gain.add_experience(state, action, reward, next_state)
                count += 1

        # 从buffer训练
        if len(self.info_gain.buffer) >= self.info_gain.batch_size:
            for _ in range(self.args.steps_per_epoch):
                result = self.info_gain.train_step()
                total_loss += result.get('loss', 0)

        return total_loss / max(self.args.steps_per_epoch, 1)

    # ─── ShareGPT 好奇心训练 ──────────────────────────────────────

    def train_curiosity_from_sharegpt(self, epoch):
        """用ShareGPT对话训练好奇心引擎"""
        if self.sharegpt is None:
            return 0.0

        n_samples = min(50, len(self.sharegpt))
        indices = np.random.permutation(len(self.sharegpt))[:n_samples]

        goal_pairs = []
        for idx in indices:
            conv = self.sharegpt[int(idx)]['conversations']
            if len(conv) < 2:
                continue

            # 相邻轮次构成 (prev_goal, next_goal) 对
            for i in range(len(conv) - 1):
                prev_emb = text_to_state(conv[i]['value'])
                next_emb = text_to_state(conv[i+1]['value'])
                goal_pairs.append((prev_emb, next_emb))

        if len(goal_pairs) < 2:
            return 0.0

        # 限制数量避免OOM
        goal_pairs = goal_pairs[:100]

        result = self.curiosity.train_step(goal_pairs)
        return result.get('loss', 0)

    # ─── ShareGPT 元学习训练 ──────────────────────────────────────

    def train_meta_from_sharegpt(self, epoch):
        """用ShareGPT对话构造元学习任务"""
        if not hasattr(self, 'maml') or self.sharegpt is None:
            return 0.0

        # 每个对话作为一个任务
        n_tasks = min(5, len(self.sharegpt))
        indices = np.random.permutation(len(self.sharegpt))[:n_tasks]

        tasks = []
        for idx in indices:
            conv = self.sharegpt[int(idx)]['conversations']
            if len(conv) < 4:
                continue

            # 前半作为support, 后半作为query
            mid = len(conv) // 2
            support_states = [text_to_state(c['value']) for c in conv[:mid]]
            query_states = [text_to_state(c['value']) for c in conv[mid:]]

            if not support_states or not query_states:
                continue

            support_x = torch.FloatTensor(np.array(support_states)).to(self.device)
            # 目标: 预测下一轮的state (简化为预测自身类别)
            support_y = torch.zeros(len(support_states), 5).to(self.device)
            for i, c in enumerate(conv[:mid]):
                label = 0 if c['from'] == 'human' else 1
                support_y[i, label % 5] = 1.0

            query_x = torch.FloatTensor(np.array(query_states)).to(self.device)
            query_y = torch.zeros(len(query_states), 5).to(self.device)
            for i, c in enumerate(conv[mid:]):
                label = 0 if c['from'] == 'human' else 1
                query_y[i, label % 5] = 1.0

            tasks.append(Task(
                name=f"conv_{idx}",
                support_x=support_x, support_y=support_y,
                query_x=query_x, query_y=query_y,
            ))

        if not tasks:
            return 0.0

        result = self.maml.meta_train_step(tasks)
        return result.get('meta_loss', 0)

    # ─── 离线学习 (记忆回放) ──────────────────────────────────────

    def train_offline_from_data(self, epoch):
        """用数据集填充记忆回放并训练离线模型"""
        # 从GoEmotions填充
        if self.go_emotions is not None:
            n = min(self.args.batch_size * 5, len(self.go_emotions))
            indices = np.random.permutation(len(self.go_emotions))[:n]
            for idx in indices:
                sample = self.go_emotions[int(idx)]
                state = text_to_state(sample['text'])
                reward = 1.0 if sample['labels'] else 0.0
                next_state = text_to_state(sample['text'][:50])  # 简化
                self.memory_replayer.add_experience(state, "read", reward, next_state)

        # 训练
        if len(self.memory_replayer.replay_buffer) >= self.args.batch_size:
            result = self.memory_replayer.replay_and_learn(
                self.offline_model, self.offline_optimizer
            )
            return result.get('loss', 0)

        return 0.0

    # ─── 验证 ──────────────────────────────────────────────────────

    def validate_emotion(self):
        """验证情绪识别 (独立验证集 + 预计算SBERT + 软标签)"""
        if self.go_emotions_val is None:
            return 0.0, 0.0

        # 确保预计算完成
        if not hasattr(self, 'go_val_embeddings'):
            self._precompute_sbert_embeddings()

        self.emotion_net.eval()
        self.text_encoder.eval()
        correct = 0
        total = 0

        n = min(500, len(self.go_emotions_val))
        indices = np.random.permutation(len(self.go_emotions_val))[:n]

        with torch.no_grad():
            for idx in indices:
                i = int(idx)
                sample = self.go_emotions_val[i]

                # 使用预计算的embedding
                emb = torch.FloatTensor(self.go_val_embeddings[i]).unsqueeze(0).to(self.device)
                state = self.text_encoder.forward_tensor(emb)  # [1, 64]

                probs = self.emotion_net.emotion_net(state)
                pred_idx = probs.argmax(dim=-1).item()

                # 软标签: 取最高票的情绪
                lbls = sample['labels']
                basic_counts = Counter()
                for lbl in lbls:
                    emo_name = GO_EMOTIONS_LABELS[lbl] if lbl < len(GO_EMOTIONS_LABELS) else 'neutral'
                    basic = BASIC_EMOTION_MAP.get(emo_name, 'neutral')
                    basic_counts[basic] += 1
                true_emo = basic_counts.most_common(1)[0][0] if basic_counts else 'neutral'
                true_idx = BASIC_EMOTIONS.index(true_emo)

                if pred_idx == true_idx:
                    correct += 1
                total += 1

        self.emotion_net.train()
        self.text_encoder.train()
        accuracy = correct / max(total, 1)
        return accuracy, 0.0

    def validate_info_gain(self):
        """验证信息增益"""
        if self.sharegpt is None:
            return 0.0

        indices = np.random.permutation(len(self.sharegpt))[:50]
        total_ig = 0.0
        count = 0

        for idx in indices:
            conv = self.sharegpt[int(idx)]['conversations']
            if len(conv) < 2:
                continue

            state = text_to_state(conv[0]['value'])
            next_state = text_to_state(conv[1]['value'])
            action = hash(conv[0]['from']) % 16

            reward_obj = self.info_gain.compute_reward(
                state, action, 1.0, next_state, use_intrinsic=True
            )
            total_ig += reward_obj.information_gain
            count += 1

        return total_ig / max(count, 1)

    # ─── 训练循环 ──────────────────────────────────────────────────

    def train_epoch(self, epoch):
        """训练一个epoch"""
        epoch_start = time.time()
        losses = {}

        # 1. 信息增益 (ShareGPT)
        if self.mode in ['full', 'info_gain']:
            ig_loss = self.train_info_gain_from_sharegpt(epoch)
            losses['info_gain_loss'] = ig_loss

        # 2. 好奇心 (ShareGPT)
        if self.mode in ['full', 'curiosity']:
            c_loss = self.train_curiosity_from_sharegpt(epoch)
            losses['curiosity_loss'] = c_loss

        # 3. 情绪识别 (GoEmotions)
        if self.mode in ['full', 'emotion']:
            e_loss, e_acc = self.train_emotion(epoch)
            losses['emotion_loss'] = e_loss
            losses['emotion_acc'] = e_acc

        # 4. 元学习 (ShareGPT)
        if self.mode in ['full', 'meta']:
            m_loss = self.train_meta_from_sharegpt(epoch)
            losses['meta_loss'] = m_loss

        # 5. 离线学习 (GoEmotions + 记忆回放)
        if self.mode in ['full', 'offline']:
            o_loss = self.train_offline_from_data(epoch)
            losses['offline_loss'] = o_loss

        # 6. 语言皮层 (ShareGPT)
        if self.mode in ['full', 'language']:
            l_loss = self.train_language_from_sharegpt(epoch)
            losses['language_loss'] = l_loss

        # GPU状态
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            self.history['gpu_memory'].append(gpu_mem)
            torch.cuda.reset_peak_memory_stats()

        epoch_time = time.time() - epoch_start
        self.history['epoch_time'].append(epoch_time)

        # 记录训练loss
        avg_loss = sum(losses.values()) / max(len(losses), 1)
        self.history['loss'].append(avg_loss)

        # 记录language loss
        if 'language_loss' in losses:
            self.history['language_loss'].append(losses['language_loss'])

        return losses

    def validate(self, epoch):
        """验证"""
        results = {}

        if self.mode in ['full', 'emotion']:
            acc, _ = self.validate_emotion()
            results['emotion_acc'] = acc
            self.history['val_emotion_acc'].append(acc)

        if self.mode in ['full', 'info_gain']:
            ig = self.validate_info_gain()
            results['info_gain'] = ig
            self.history['info_gain'].append(ig)

        return results

    # ─── Checkpoint ────────────────────────────────────────────────

    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'args': vars(self.args),
            'info_gain_state': self.info_gain.world_model.state_dict(),
            'info_gain_optimizer': self.info_gain.optimizer.state_dict(),
            'curiosity_state': self.curiosity.state_dict(),
            'emotion_state': self.emotion_net.state_dict(),
            'language_state': self.language_cortex.state_dict(),
            'offline_model_state': self.offline_model.state_dict(),
            'history': self.history,
        }

        if hasattr(self, 'maml'):
            checkpoint['maml_state'] = self.maml.state_dict()
            checkpoint['maml_optimizer'] = self.maml.meta_optimizer.state_dict()

        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()

        latest_path = self.output_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  [Checkpoint] Best model saved at epoch {epoch}")

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
        self.emotion_net.load_state_dict(checkpoint['emotion_state'])
        self.offline_model.load_state_dict(checkpoint['offline_model_state'])

        if 'maml_state' in checkpoint and hasattr(self, 'maml'):
            self.maml.load_state_dict(checkpoint['maml_state'])
            self.maml.meta_optimizer.load_state_dict(checkpoint['maml_optimizer'])

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
    parser = argparse.ArgumentParser(description='Civis Lucri-Faber Real Data Training')

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'info_gain', 'curiosity', 'emotion', 'offline', 'meta', 'language'],
                        help='Training mode')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
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

    print("\n" + "=" * 60)
    print("Civis Lucri-Faber Real Data Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Output: {args.output_dir}")

    trainer = RealDataTrainer(args)

    # Resume
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        loaded_epoch = trainer.load_checkpoint(resume_path)
        if loaded_epoch > 0:
            start_epoch = loaded_epoch + 1

    # 训练循环
    best_metric = 0.0
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        # 训练
        losses = trainer.train_epoch(epoch)
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        print(f"  [Losses] {loss_str}")

        # 验证
        if epoch % args.log_interval == 0:
            val_results = trainer.validate(epoch)
            if val_results:
                val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_results.items()])
                print(f"  [Val] {val_str}")

                # 用情绪准确率作为主指标
                metric = val_results.get('emotion_acc', val_results.get('info_gain', 0))
                is_best = metric > best_metric
                if is_best:
                    best_metric = metric

                trainer.save_checkpoint(epoch, is_best=is_best)

        # GPU状态
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / 1e9
            print(f"  [GPU] Memory: {current_mem:.2f} GB")

        epoch_time = trainer.history['epoch_time'][-1]
        print(f"  [Time] {epoch_time:.2f}s")

    # 完成
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("Training Completed!")
    print(f"{'=' * 60}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best Metric: {best_metric:.4f}")

    trainer.save_history()
    trainer.save_checkpoint(args.epochs)


if __name__ == '__main__':
    main()

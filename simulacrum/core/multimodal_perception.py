"""
多模态感知系统

整合视觉、听觉、语言感知 -> 情感生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 听觉系统 ============

class AuditoryCortex(nn.Module):
    """
    听觉皮层 - 仿生架构

    整合:
    1. 外周/Subcortical: 耳蜗→时频分析
    2. A1: 初级听觉皮层
    3. 腹侧流: 识别 (A1→STG→颞叶)
    4. 背侧流: 定位+运动 (A1→顶叶→额叶)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
    ):
        super().__init__()

        # 直接创建听觉系统，避免导入问题
        from core.auditory_cortex import create_auditory_cortex
        self.auditory = create_auditory_cortex(
            sample_rate=sample_rate,
            n_filters=n_mels,
        )

    def process_audio(
        self,
        audio: torch.Tensor,
    ) -> dict:
        """处理原始音频"""
        result = self.auditory(audio)

        return {
            'features': result['features'],
            'valence': result['valence'],
            'arousal': result['arousal'],
            'dominance': result['dominance'],
            'pleasantness': result['pleasantness'],
            # 腹侧流
            'what': result['what'],
            'lexical': result['lexical'],
            # 背侧流
            'where': result['where'],
            'how': result['how'],
            'spatial': result['spatial'],
            'motor': result['motor'],
        }


# ============ 语言系统 ============

class TextEncoder(nn.Module):
    """
    文本编码器

    使用简单embedding + Transformer
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)

        # 简单Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 情感头
        self.emotion_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # VAD模型
        )

        self.embed_dim = embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> dict:
        """
        Args:
            tokens: [B, T] token ids
        Returns:
            features: [B, embed_dim]
            emotion: dict
        """
        B, T = tokens.shape

        # Embedding
        x = self.embedding(tokens)

        # 位置编码
        if self.pos_embedding.shape[1] >= T:
            x = x + self.pos_embedding[:, :T, :]
        else:
            x = x + self.pos_embedding[:, :1, :].expand(-1, T, -1)

        # Transformer
        x = self.transformer(x)

        # CLS token (取第一个)
        features = x[:, 0]

        # 情感
        emotion = self.emotion_head(features)

        return {
            'features': features,
            'tokens': tokens,
            'valence': torch.tanh(emotion[:, 0]),
            'arousal': torch.sigmoid(emotion[:, 1]),
            'dominance': torch.sigmoid(emotion[:, 2]),
            'pleasantness': torch.sigmoid(emotion[:, 3]),
        }


class LanguageCortex(nn.Module):
    """
    语言皮层

    整合文本理解
    """

    def __init__(
        self,
        vocab_size: int = 10000,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size=vocab_size)

    def process_text(
        self,
        tokens: torch.Tensor,
    ) -> dict:
        """处理文本"""
        return self.text_encoder(tokens)


# ============ 多模态融合系统 ============

class MultimodalFusion(nn.Module):
    """
    多模态融合

    整合视觉、听觉、语言 -> 统一表示
    """

    def __init__(
        self,
        visual_dim: int = 64,
        audio_dim: int = 64,
        text_dim: int = 128,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim

        # 各模态投影到统一空间
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, 64),
            nn.ReLU(),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
        )

        # 融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # 注意力权重 (可学习)
        self.attn_weights = nn.Parameter(torch.ones(3))

    def forward(
        self,
        visual_features: torch.Tensor = None,
        audio_features: torch.Tensor = None,
        text_features: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            visual_features: [B, visual_dim]
            audio_features: [B, audio_dim]
            text_features: [B, text_dim]
        """
        features = []
        weight_list = []

        if visual_features is not None:
            v = self.visual_proj(visual_features)
            features.append(v)
            weight_list.append(self.attn_weights[0])

        if audio_features is not None:
            a = self.audio_proj(audio_features)
            features.append(a)
            weight_list.append(self.attn_weights[1])

        if text_features is not None:
            t = self.text_proj(text_features)
            features.append(t)
            weight_list.append(self.attn_weights[2])

        if not features:
            return {
                'fused': None,
                'salience': 0.0,
            }

        # 注意力加权
        weights_tensor = torch.stack(weight_list)
        weights = F.softmax(weights_tensor, dim=0)

        # 加权融合
        fused = sum(f * w for f, w in zip(features, weights))

        # 进一步融合
        if len(features) > 1:
            fused = self.fusion_layer(
                torch.cat(features, dim=-1)
            )

        salience = torch.sigmoid(fused.mean())

        return {
            'fused': fused,
            'salience': salience.item(),
            'weights': {
                'visual': weights[0].item(),
                'audio': weights[1].item(),
                'text': weights[2].item(),
            } if len(weights) >= 3 else {},
        }


# ============ 情感生成系统 ============

class EmotionalEncoder(nn.Module):
    """
    情感编码器

    从多模态感知生成情感
    """

    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        # 情绪分类
        self.emotion_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # joy, sadness, anger, fear, neutral
            nn.Softmax(dim=-1)
        )

        # VAD模型
        self.vad_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # valence, arousal, dominance
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> dict:
        """
        Args:
            features: [B, input_dim]
        """
        # 情绪分类
        emotion_probs = self.emotion_net(features)
        emotions = ["joy", "sadness", "anger", "fear", "neutral"]
        emotion = emotions[emotion_probs.argmax(dim=-1).item()]

        # VAD
        vad = self.vad_net(features)
        valence = torch.tanh(vad[:, 0])
        arousal = torch.sigmoid(vad[:, 1])
        dominance = torch.sigmoid(vad[:, 2])

        # 强度
        intensity = arousal * torch.abs(valence)

        return {
            'emotion': emotion,
            'emotion_probs': emotion_probs,
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'intensity': intensity,
        }


# ============完整感知系统 ============

class MultimodalPerception(nn.Module):
    """
    多模态感知系统

    整合:
    - 视觉 (Censor双通路)
    - 听觉 (Audio)
    - 语言 (Text)
    输出情感
    """

    def __init__(
        self,
        vocab_size: int = 10000,
    ):
        super().__init__()

        # 视觉 - 使用Censor桥接
        from censor_bridge import create_censor_vision
        self.vision = create_censor_vision('dual')

        # 听觉
        self.auditory = AuditoryCortex()

        # 语言
        self.language = LanguageCortex(vocab_size=vocab_size)

        # 多模态融合
        self.fusion = MultimodalFusion(
            visual_dim=64,
            audio_dim=64,
            text_dim=128,
        )

        # 情感生成
        self.emotion_encoder = EmotionalEncoder(input_dim=64)

    def forward(
        self,
        optical_flow: torch.Tensor = None,  # 视觉光流
        rgb_ppg: torch.Tensor = None,         # 视觉RGB
        audio: torch.Tensor = None,           # 听觉
        text_tokens: torch.Tensor = None,      # 语言
    ) -> dict:
        """
        多模态感知前向传播

        所有输入都是可选的，缺省的模态不参与计算
        """
        visual_result = None
        audio_result = None
        text_result = None
        modality_active = {}

        # 视觉
        if optical_flow is not None or rgb_ppg is not None:
            visual_result = self.vision(optical_flow, rgb_ppg)
            if visual_result.get('embedding') is not None:
                visual_result = {'features': visual_result['embedding']}
            modality_active['visual'] = visual_result is not None

        # 听觉
        if audio is not None:
            audio_result = self.auditory.process_audio(audio)
            modality_active['audio'] = True

        # 语言
        if text_tokens is not None:
            text_result = self.language.process_text(text_tokens)
            modality_active['text'] = True

        # 融合
        v_feat = visual_result.get('features') if visual_result else None
        a_feat = audio_result.get('features') if audio_result else None
        t_feat = text_result.get('features') if text_result else None

        fusion_result = self.fusion(v_feat, a_feat, t_feat)

        # 情感生成
        emotion = None
        if fusion_result.get('fused') is not None:
            emotion = self.emotion_encoder(fusion_result['fused'])

        return {
            'vision': visual_result,
            'audio': audio_result,
            'text': text_result,
            'fused': fusion_result.get('fused'),
            'salience': fusion_result.get('salience', 0.5),
            'emotion': emotion,
            'modality_active': modality_active,
        }


# ============ 便捷函数 ============

def create_multimodal_perception(
    vocab_size: int = 10000,
) -> MultimodalPerception:
    """创建多模态感知系统"""
    return MultimodalPerception(vocab_size=vocab_size)


__all__ = [
    'MelSpectrogram',
    'AudioEncoder',
    'AuditoryCortex',
    'TextEncoder',
    'LanguageCortex',
    'MultimodalFusion',
    'EmotionalEncoder',
    'MultimodalPerception',
    'create_multimodal_perception',
]

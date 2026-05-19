"""
视觉皮层系统 - 参考Censor项目

基于Vision Transformer (ViT)的视觉处理：
- 图像分块 + 线性嵌入
- Transformer编码器
- 分类头/特征输出

参考：D:\censor\data\iMER\backbone\vision_transformer_dual_prompt.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class ImagePatchEmbedding(nn.Module):
    """图像分块嵌入"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 线性嵌入层 (类似ResNet的卷积)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, C, H, W = x.shape
        # 展平空间维度
        x = self.proj(x)  # B, embed_dim, H//patch, W//patch
        x = x.flatten(2).transpose(1, 2)  # B, n_patches, embed_dim
        return x, self.n_patches


class ViTAttention(nn.Module):
    """
    ViT自注意力 - 参考censor项目
    简化版
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """MLP块"""
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)

        # Attention
        self.attn = ViTAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # LayerNorm
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer 视觉皮层

    完整的ViT架构
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 图像嵌入
        self.patch_embed = ImagePatchEmbedding(
            img_size, patch_size, in_chans, embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # Class token + 位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """前向传播"""
        # 嵌入
        x, n_patches = self.patch_embed(x)

        # Class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer块
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 特征或分类
        if return_features:
            return x  # 返回所有token的特征

        # 只取cls token
        x = x[:, 0]

        return self.head(x)


class ResNetBackbone(nn.Module):
    """
    ResNet骨干 - 参考censor项目

    替代ViT的卷积 backbone
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 64,
    ):
        super().__init__()

        # 基础卷积
        self.conv1 = nn.Conv2d(in_channels, base_dim, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块
        self.layer1 = self._make_layer(base_dim, base_dim, 2)
        self.layer2 = self._make_layer(base_dim, base_dim * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_dim * 2, base_dim * 4, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ):
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class VisualCortex(nn.Module):
    """
    完整视觉皮层

    整合ViT + ResNet
    """

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 768,
        use_vit: bool = True,
    ):
        super().__init__()

        self.use_vit = use_vit
        self.embed_dim = embed_dim

        if use_vit:
            self.encoder = VisionTransformer(
                img_size=224,
                patch_size=16,
                in_chans=input_channels,
                num_classes=1000,
                embed_dim=embed_dim,
                depth=12,
                num_heads=8,
            )
        else:
            self.encoder = ResNetBackbone(
                in_channels=input_channels,
                base_dim=embed_dim // 4,
            )

        # 特征投影
        self.feature_proj = nn.Linear(embed_dim, 64)

    def forward(
        self,
        visual_input: torch.Tensor,
        return_features: bool = False,
    ) -> Dict:
        """
        处理视觉输入

        Args:
            visual_input: [B, C, H, W] 或 [B, C]
        """
        # 调整维度
        if visual_input.dim() == 2:
            # 1D特征，视为已经处理过
            return {
                'features': visual_input,
                'salience': 0.5,
            }

        if visual_input.dim() == 3 and visual_input.shape[1] < 50:
            # 已经分块的序列
            features = visual_input
        else:
            # 2D/3D图像
            if self.use_vit:
                # 确保图像尺寸
                if visual_input.shape[-1] != 224:
                    visual_input = F.interpolate(
                        visual_input,
                        size=224,
                        mode='bilinear',
                        align_corners=False,
                    )
                features = self.encoder(
                    visual_input,
                    return_features=True,
                )
            else:
                # ResNet
                features = self.encoder(visual_input)
                features = features.unsqueeze(1)  # [B, 1, dim]

        # 特征投影
        features_64 = self.feature_proj(features[:, -1, :] if features.dim() == 3 else features)

        # 计算显著度
        salience = torch.sigmoid(features_64.mean())

        return {
            'features': features,
            'embedding': features_64,
            'salience': salience.item(),
        }


# ============ 便捷函数 ============

def create_visual_cortex(
    input_channels: int = 3,
    embed_dim: int = 768,
    use_vit: bool = False,  # ResNet更快
) -> VisualCortex:
    return VisualCortex(input_channels, embed_dim, use_vit)


__all__ = [
    'VisionTransformer',
    'ResNetBackbone',
    'VisualCortex',
    'create_visual_cortex',
    # 新增模块
    'AdaptiveVisualAttention',
    'SaliencyDetectorE2E',
    'StandardMoE',
]


# ============ AdaptiveVisualAttention - 两阶段视觉注意力 ============
# 对应Censor的AdaptiveOpticalFlow，但用于视觉注意力


class AdaptiveVisualAttention(nn.Module):
    """
    两阶段视觉注意力：
    - Stage 1: 快速粗筛选 (saliency screening)
    - Stage 2: 精细注意力 (fine attention) 仅当motion detected

    对应Censor的AdaptiveOpticalFlow思想
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.threshold = threshold

        # Saliency检测器
        self.saliency_scorer = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 精细注意力
        self.fine_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_stage: bool = False,
    ) -> Dict:
        """
        Args:
            x: [B, T, D] 时序特征
        Returns:
            output: [B, T, D] 注意力后的特征
            stage: 'fast' or 'fine'
        """
        B, T, D = x.shape

        # Stage 1: 快速筛选
        saliency_scores = self.saliency_scorer(x).squeeze(-1)  # [B, T]
        motion_magnitude = saliency_scores.mean()

        if motion_magnitude > self.threshold:
            # Stage 2: 精细注意力
            attn_out, _ = self.fine_attn(x, x, x)
            stage = 'fine'
        else:
            # 快速路径：直接使用原始特征
            attn_out = x
            stage = 'fast'

        output = self.output_proj(attn_out)

        if return_stage:
            return {'output': output, 'stage': stage, 'saliency': saliency_scores}
        return {'output': output, 'stage': stage}


# ============ SaliencyDetectorE2E - 全端到端显著性检测 ============
# 对应Censor的SaliencyDetectorE2E


class SaliencyDetectorE2E(nn.Module):
    """
    全端到端显著性检测器

    对应Censor的SaliencyDetectorE2E：
    1. 所有参数可学习
    2. 分辨率自适应sigma
    """

    def __init__(
        self,
        embed_dim: int = 768,
        pyramid_levels: int = 4,
        sigma_ratio: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pyramid_levels = pyramid_levels

        # 可学习参数
        self.sigma_ratio = nn.Parameter(torch.tensor(sigma_ratio))
        self.center_bias = nn.Parameter(torch.tensor(0.5))
        self.fusion_weights = nn.Parameter(torch.ones(pyramid_levels) / pyramid_levels)

        # 多层特征提取
        self.pyramid_convs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(pyramid_levels)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict:
        """
        Args:
            x: [B, T, D] 特征序列
        Returns:
            saliency: [B, T] 显著性分数
        """
        B, T, D = x.shape

        # 多层融合
        weights = F.softmax(self.fusion_weights, dim=0)

        pyramid_feats = []
        for i, conv in enumerate(self.pyramid_convs):
            if i == 0:
                pyramid_feats.append(x)
            else:
                # 下采样
                pyramid_feats.append(F.avg_pool1d(pyramid_feats[-1].transpose(1, 2), 2).transpose(1, 2))

        # 加权融合
        fused = sum(w * feat for w, feat in zip(weights, pyramid_feats))

        # 显著性评分
        scores = torch.sum(fused * x, dim=-1)  # [B, T]
        scores = scores * self.center_bias

        return {
            'saliency': scores,
            'fused_features': fused,
        }


# ============ StandardMoE - 标准MoE对比 ============
# 对应Censor的StandardMoE（更客观的替代方案）


class StandardMoE(nn.Module):
    """
    标准MoE - 对比Censor的BioMoE

    对比BioMoE：
    - 简单的gating（无生物学先验）
    - 无membrane potential
    - 无emotional state
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 7,
        num_experts: int = 3,
        expert_hidden: int = 2048,
        k: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k

        # 专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, output_dim),
            )
            for _ in range(num_experts)
        ])

        # 简单门控
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
        )

        # Load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict:
        """
        Args:
            x: [B, D]
        Returns:
            output: [B, output_dim]
            gate_weights: [B, k]
        """
        B = x.shape[0]

        # 门控
        gate_logits = self.gate(x)  # [B, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Top-k
        top_k_weights, top_k_idx = torch.topk(gate_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 专家输出
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, num_experts, output_dim]

        # 加权
        output = sum(w * o for w, o in zip(gate_weights.unbind(), outputs.unbind()))
        output = output.squeeze(1)

        # 更新使用统计
        self.expert_usage += (gate_weights > 0.5).float().sum(dim=0)

        return {
            'output': output,
            'gate_weights': gate_weights,
            'expert_usage': self.expert_usage,
        }
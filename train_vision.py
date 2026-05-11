"""
视觉Censor训练 - MAE重建版
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mask_random_patches(x, mask_ratio=0.75):
    """随机遮罩 patches (类似MAE)"""
    B, C, T, H, W = x.shape
    # 简化: 随机遮罩时间/空间
    mask = torch.rand(B, T, H, W).float() > mask_ratio
    masked = x.clone()
    masked = masked * mask.unsqueeze(1).float()  # 应用遮罩
    return masked, mask


def train():
    print("=== 视觉Censor训练 (MAE重建) ===")

    base = os.path.dirname(__file__)
    censor = load(os.path.join(base, 'censor_bridge.py'), 'censor_bridge')

    # 模型
    model = censor.create_censor_vision('dual')
    print(f"参数: {sum(p.numel() for p in model.parameters())}")

    # 解码头 (重建原始patch)
    decoder = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64)  # 64维特征重建
    )

    opt = torch.optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)

    # 训练
    model.train()
    for epoch in range(5):
        losses = []
        for step in range(20):
            # 随机视频
            flow = torch.randn(2, 2, 8, 32, 32)
            rgb = torch.randn(2, 6, 8, 32, 32)

            # 遮罩
            masked_rgb, mask = mask_random_patches(rgb, mask_ratio=0.75)

            # 编码 ( видий通路)
            with torch.no_grad():
                result = model(flow, masked_rgb)
            embedding = result.get('embedding')

            # 解码重建
            reconstructed = decoder(embedding)

            # 重建损失: 对未遮罩区域重建
            full_result = model(flow, rgb)
            target = full_result.get('embedding')

            # 对比重建损失
            loss = F.mse_loss(reconstructed, target.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}: loss={sum(losses)/len(losses):.4f}")

    # 保存
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vision.pt')
    print("保存 checkpoints/vision.pt")
    print("训练完成!")


if __name__ == "__main__":
    train()
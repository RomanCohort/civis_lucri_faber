"""Debug GRU shape"""
import torch
gru = torch.nn.GRU(64, 64, batch_first=True, bidirectional=True)
x = torch.randn(1, 8, 64)
output, hidden = gru(x)
print(f"input: {x.shape}")
print(f"output: {output.shape}")  # [B,T,hidden]
print(f"hidden: {hidden.shape}")  # [dirs, B, hidden]
# 取最后一个
combined = torch.cat([output[:, -1, :], output[:, 0, :]], dim=-1)
print(f"combined: {combined.shape}")
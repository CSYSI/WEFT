import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareCrossTokenRefine(nn.Module):
    def __init__(self, residual=True, use_gate=True, num_heads=4, temperature=1.0, edge_weight=0.5):
        super().__init__()
        self.residual = residual
        self.use_gate = use_gate
        self.num_heads = num_heads
        self.temperature = temperature
        self.edge_weight = edge_weight

        if use_gate:
            self.gate_layer = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            )

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        x_norm = F.normalize(x, dim=-1)  # [B, N, C]

        if self.num_heads > 1:
            assert C % self.num_heads == 0
            head_dim = C // self.num_heads
            x_heads = x_norm.view(B, N, self.num_heads, head_dim).transpose(1, 2)
            sim = torch.matmul(x_heads, x_heads.transpose(-1, -2)) / (head_dim ** 0.5 * self.temperature)
            attn = F.softmax(sim, dim=-1)
            x_ = x.view(B, N, self.num_heads, head_dim).transpose(1, 2)
            x_refined = torch.matmul(attn, x_)
            x_refined = x_refined.transpose(1, 2).reshape(B, N, C)
        else:
            sim = torch.bmm(x_norm, x_norm.transpose(1, 2)) / (C ** 0.5 * self.temperature)
            attn = F.softmax(sim, dim=-1)
            x_refined = torch.bmm(attn, x)

        with torch.no_grad():
            diff = torch.var(x, dim=-1, keepdim=True)  # [B, N, 1]
            edge_mask = torch.sigmoid((diff - diff.mean()) / (diff.std() + 1e-6))

        x_refined = (1 + self.edge_weight * edge_mask) * x_refined

        if self.use_gate:
            gate_input = x.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)  # [B, 1, 1]
            gate = self.gate_layer(gate_input)
            x_refined = gate * x_refined

        if self.residual:
            x_refined = x + x_refined

        return x_refined



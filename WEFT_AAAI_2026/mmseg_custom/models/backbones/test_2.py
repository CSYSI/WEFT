import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTokenMixer(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim  # Depthwise
        )
        self.pwconv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1  # Pointwise
        )
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # back to (B, L, C)
        return x

class DepthPointwiseConv1D(nn.Module):
    def __init__(self, dim, outdim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim // 4

        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            groups=dim,
            bias=False
        )


        self.pointwise_down = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)


        self.pointwise_up = nn.Conv1d(hidden_dim, outdim, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: Tensor of shape (B, L, D)
        """
        x = x.transpose(1, 2)  # → (B, D, L)

        # Depth-wise convolution
        x = self.depthwise(x)  # (B, D, L)

        # Point-wise projection down
        x = self.pointwise_down(x)  # (B, hidden_dim, L)
        x = self.act(x)
        x = self.dropout(x)

        # Point-wise projection up
        x = self.pointwise_up(x)  # (B, D, L)
        x = self.dropout(x)

        x = x.transpose(1, 2).contiguous()  # → (B, L, D)
        return x


class DepthPointwiseConv1D_1(nn.Module):
    def __init__(self, dim, outdim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim // 4


        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            groups=dim,
            bias=False
        )


        self.pointwise_down = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)


        self.pointwise_up = nn.Conv1d(hidden_dim, outdim, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: Tensor of shape (B, L, D)
        """
        x = x.transpose(1, 2)  # → (B, D, L)


        x = self.depthwise(x)  # (B, D, L)


        x = self.pointwise_down(x)  # (B, hidden_dim, L)
        x = self.act(x)
        x = self.dropout(x)

        # Point-wise projection up
        x = self.pointwise_up(x)  # (B, D, L)
        x = self.dropout(x)

        x = x.transpose(1, 2) # → (B, L, D)
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim,dim)
        self.k_proj = nn.Linear(dim,dim)
        self.v_proj = nn.Linear(dim,dim)
        self.out_proj = nn.Linear(dim,dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_q, token_kv):
        """
        token_q: Tensor, shape (B, L_q, D)  # e.g. (2, 128, 32)
        token_kv: Tensor, shape (B, L_kv, D)  # e.g. (2, 45, 32)
        """
        B, L_q, D = token_q.shape
        L_kv = token_kv.shape[1]


        Q = self.q_proj(token_q)     # (B, L_q, D)
        K = self.k_proj(token_kv)    # (B, L_kv, D)
        V = self.v_proj(token_kv)    # (B, L_kv, D)


        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, L_q, L_kv)
        attn_weights = F.softmax(attn_scores, dim=-1)                    # (B, L_q, L_kv)
        attn_weights = self.dropout(attn_weights)


        out = torch.matmul(attn_weights, V)
        out = self.out_proj(out)

        return out  # shape: (B, L_q, D)



class DepthPointwiseLinear(nn.Module):
    def __init__(self, dim, outdim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim // 4


        self.depthwise = nn.Linear(dim, hidden_dim, bias=False)


        self.pointwise_down = nn.Linear(hidden_dim, hidden_dim, bias=False)


        self.pointwise_up = nn.Linear(hidden_dim, outdim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: Tensor of shape (B, L, D)
        """

        x = self.depthwise(x)  # (B, L, D)

        # Point-wise down
        x = self.pointwise_down(x)  # (B, L, hidden_dim)
        x = self.act(x)
        x = self.dropout(x)

        # Point-wise up
        x = self.pointwise_up(x)  # (B, L, D)
        x = self.dropout(x)

        return x





def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




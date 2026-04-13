import torch
import torch.nn as nn
import torch.nn.functional as F


class LoGConv2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        log_kernel = torch.tensor([[1, -2, 1], [1, -2, 1], [1, -2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.weight = nn.Parameter(log_kernel.repeat(in_channels, 1, 1, 1))
        self.pad = 1

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return F.conv2d(x, self.weight, padding=0, groups=x.shape[1])

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels
        )
    def forward(self, x):
        return self.depthwise(x)


class DConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super(DConv, self).__init__()

        self.DConv = nn.Sequential(*[
            nn.Conv2d(channels//4, channels//4, kernel_size, padding=kernel_size // 2, groups=channels//4),
            nn.SyncBatchNorm(channels//4),
            nn.ReLU(inplace=True),
        ])

        self.Conv1 = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
        ])

    def forward(self, x):
        x_res = x
        split0,split1,split2,split3 = torch.chunk(x,chunks=4,dim=1)
        x0 = self.DConv(split0)
        x1 = self.DConv(x0+split1)
        x2 = self.DConv(x1+split2)
        x3 = self.DConv(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+ x_res)
        return output

class TokenEnhancerWeighted(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_branch = LoGConv2D(channels)
        self.pool_branch = nn.AdaptiveMaxPool2d(output_size=(None,None))

        self.ms_conv3 = DConv(channels, kernel_size=3)
        self.ms_conv5 = DConv(channels, kernel_size=5)
        self.ms_conv7 = DConv(channels, kernel_size=7)

        self.weight_log_x1 = nn.Parameter(torch.ones(1))
        self.weight_pool_x1 = nn.Parameter(torch.ones(1))
        self.weight_ms_x1 = nn.Parameter(torch.ones(1))

        self.weight_log_x2 = nn.Parameter(torch.ones(1))
        self.weight_pool_x2 = nn.Parameter(torch.ones(1))
        self.weight_ms_x2 = nn.Parameter(torch.ones(1))

        self.weight_log_x3 = nn.Parameter(torch.ones(1))
        self.weight_pool_x3 = nn.Parameter(torch.ones(1))
        self.weight_ms_x3 = nn.Parameter(torch.ones(1))

        self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21

        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()

        ori_x1=x1
        ori_x2 = x2
        ori_x3 = x3

        log_feat_x1 = self.log_branch(x1)
        log_feat_x2 = self.log_branch(x2)
        log_feat_x3 = self.log_branch(x3)

        pooled_x1 = self.pool_branch(x1)
        pooled_x2 = self.pool_branch(x2)
        pooled_x3 = self.pool_branch(x3)

        ms_feat1_x1 = self.ms_conv3(x1) + self.ms_conv5(x1) + self.ms_conv7(x1)
        ms_feat1_x2 = self.ms_conv3(x2) + self.ms_conv5(x2) + self.ms_conv7(x2)
        ms_feat1_x3 = self.ms_conv3(x3) + self.ms_conv5(x3) + self.ms_conv7(x3)


        weights_x1 = F.softmax(torch.stack([self.weight_log_x1, self.weight_pool_x1, self.weight_ms_x1]), dim=0)
        w1_x1, w2_x1, w3_x1 = weights_x1[0], weights_x1[1], weights_x1[2]

        weights_x2 = F.softmax(torch.stack([self.weight_log_x2, self.weight_pool_x2, self.weight_ms_x2]), dim=0)
        w1_x2, w2_x2, w3_x2 = weights_x2[0], weights_x2[1], weights_x2[2]

        weights_x3 = F.softmax(torch.stack([self.weight_log_x3, self.weight_pool_x3, self.weight_ms_x3]), dim=0)
        w1_x3, w2_x3, w3_x3 = weights_x3[0], weights_x3[1], weights_x3[2]

        fused_x1 = w1_x1 * log_feat_x1 + w2_x1 * pooled_x1 + w3_x1 * ms_feat1_x1
        fused_x2 = w1_x2 * log_feat_x2 + w2_x2 * pooled_x2 + w3_x2 * ms_feat1_x2
        fused_x3 = w1_x3 * log_feat_x3 + w2_x3 * pooled_x3 + w3_x3 * ms_feat1_x3

        x1 = self.act(ori_x1 + fused_x1)
        x2 = self.act(ori_x2 + fused_x2)
        x3 = self.act(ori_x3 + fused_x3)

        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = x3.flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)
        return x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧮 Total trainable parameters: {total:,}\n")
    print("📊 Breakdown by submodule:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<40} {param.numel():>10,} params")
    return total

if __name__ == "__main__":
    channels = 1024
    model = TokenEnhancerWeighted(channels=channels)
    test_input = torch.randn(2, 21504, 1024)
    out = model(test_input,H=64,W=64)

    print(f"✅ Output shape: {out.shape}")
    count_parameters(model)



#
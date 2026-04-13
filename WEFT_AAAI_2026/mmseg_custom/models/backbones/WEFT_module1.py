from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath

import torch.nn.functional as F
import pywt
import pywt.data

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class Expert_WConv1(nn.Module):
    def __init__(self, channels):
        super(Expert_WConv1, self).__init__()

        self.WConv1 = nn.Sequential(*[
            WTConv2d(channels//4, channels//4, kernel_size=1, stride=1, bias=True),
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
        x0 = self.WConv1(split0)
        x1 = self.WConv1(x0+split1)
        x2 = self.WConv1(x1+split2)
        x3 = self.WConv1(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+x_res)
        return output

class Expert_WConv3(nn.Module):
    def __init__(self, channels):
        super(Expert_WConv3, self).__init__()

        self.WConv3 = nn.Sequential(*[
            WTConv2d(channels//4, channels//4, kernel_size=3, stride=1, bias=True),
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
        x0 = self.WConv3(split0)
        x1 = self.WConv3(x0+split1)
        x2 = self.WConv3(x1+split2)
        x3 = self.WConv3(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+x_res)
        return output


class Expert_WConv5(nn.Module):
    def __init__(self, channels):
        super(Expert_WConv5, self).__init__()

        self.WConv5 = nn.Sequential(*[
            WTConv2d(channels//4, channels//4, kernel_size=5, stride=1, bias=True),
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
        x0 = self.WConv5(split0)
        x1 = self.WConv5(x0+split1)
        x2 = self.WConv5(x1+split2)
        x3 = self.WConv5(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+x_res)
        return output

class Expert_WConv7(nn.Module):
    def __init__(self, channels):
        super(Expert_WConv7, self).__init__()

        self.WConv7 = nn.Sequential(*[
            WTConv2d(channels//4, channels//4, kernel_size=7, stride=1, bias=True),
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
        x0 = self.WConv7(split0)
        x1 = self.WConv7(x0+split1)
        x2 = self.WConv7(x1+split2)
        x3 = self.WConv7(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+x_res)
        return output

class Expert_WConv9(nn.Module):
    def __init__(self, channels):
        super(Expert_WConv9, self).__init__()

        self.WConv9 = nn.Sequential(*[
            WTConv2d(channels//4, channels//4, kernel_size=9, stride=1, bias=True),
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
        x0 = self.WConv9(split0)
        x1 = self.WConv9(x0+split1)
        x2 = self.WConv9(x1+split2)
        x3 = self.WConv9(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+x_res)
        return output

class Expert_WConv11(nn.Module):
    def __init__(self, channels):
        super(Expert_WConv11, self).__init__()

        self.WConv11 = nn.Sequential(*[
            WTConv2d(channels//4, channels//4, kernel_size=11, stride=1, bias=True),
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
        x0 = self.WConv11(split0)
        x1 = self.WConv11(x0+split1)
        x2 = self.WConv11(x1+split2)
        x3 = self.WConv11(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+x_res)
        return output

class Expert_WConv13(nn.Module):
    def __init__(self, channels):
        super(Expert_WConv13, self).__init__()

        self.WConv13 = nn.Sequential(*[
            WTConv2d(channels//4, channels//4, kernel_size=13, stride=1, bias=True),
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
        x0 = self.WConv13(split0)
        x1 = self.WConv13(x0+split1)
        x2 = self.WConv13(x1+split2)
        x3 = self.WConv13(x2+split3)
        output = self.Conv1(torch.cat((x0,x1,x2,x3),1)+x_res)
        return output

class GatingNetwork(nn.Module):
    def __init__(self, in_features, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(in_features, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

class WTConv_Expert(nn.Module):
    def __init__(self, channels):
        super(WTConv_Expert, self).__init__()


        self.experts = nn.ModuleList([
            Expert_WConv1(channels),
            Expert_WConv3(channels),
            Expert_WConv5(channels),
            Expert_WConv7(channels),
            Expert_WConv9(channels),
            Expert_WConv11(channels),
            Expert_WConv13(channels),
        ])

        self.gating_network = GatingNetwork(in_features=channels, num_experts=7)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        x_flattened = x.view(batch_size, channels, -1)
        gate_weights = self.gating_network(x_flattened.mean(dim=-1))

        top4_weights, top4_indices = torch.topk(gate_weights, k=4, dim=1)

        top4_weights = torch.softmax(top4_weights, dim=1) + 1e-8
        top4_weights /= top4_weights.sum(dim=1, keepdim=True)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, 10, C, H, W)

        top4_indices = top4_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *expert_outputs.shape[2:])
        selected_experts = torch.gather(expert_outputs, 1, top4_indices)  # (batch_size, 4, C, H, W)

        final_output = torch.sum(selected_experts * top4_weights.view(batch_size, 4, 1, 1, 1), dim=1)

        return final_output


class Module1(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.down_sampling = nn.Sequential(*[
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
        ])

        self.WTConv_EH = WTConv_Expert(inplanes)

        self.up_embeding = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c1 = c1 + self.WTConv_EH(c1)
            c2 = self.down_sampling(c1)
            c2 = c2 + self.WTConv_EH(c2)
            c3 = self.down_sampling(c2)
            c3 = c3 + self.WTConv_EH(c3)
            c4 = self.down_sampling(c3)
            c4 = c4 + self.WTConv_EH(c4)

            c1 = self.up_embeding(c1)
            c2 = self.up_embeding(c2)
            c3 = self.up_embeding(c3)
            c4 = self.up_embeding(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
import torch
import torch.nn as nn
from torch.nn import *
import math
from torch import Tensor
from torch.utils.flop_counter import FlopCounterMode

class MulSE(nn.Module):

    def __init__(
            self,
            in_ch=12,
            out_ch=64,
            dim_squeeze=8,
            num_freqs = 257,
            num_layers: int = 5,
    ):
        super().__init__()

        self.encoder = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 5), stride=1, padding=(0,2))
        self.decoder = nn.ConvTranspose2d(out_ch, 2, kernel_size=(1, 5), stride=1, padding=(0,2))

        mattn_layers = []
        for n in range(num_layers):
            mattn_layer = MulSE_layer(
                embed_dim=out_ch,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                bidirectional=True
            )
            mattn_layers.append(mattn_layer)
        self.mattn_layers = nn.ModuleList(mattn_layers)

    def forward(self, noisy_input: Tensor) -> tuple:  # noisy_input------> (B,F,T,C)
        B, F, T, C = noisy_input.shape
        noisy_com = torch.view_as_real(noisy_input)   # (B,F,T,C,2)
        noisy_ipt = noisy_com.reshape(B, F, T, C * 2).permute(0, -1, 1, 2)  # (B,C*2,F,T)
        # --------------------------------------------------------------------------------------------------------------
        # 1. Apply Encoder
        x = self.encoder(noisy_ipt) # (B,C,F,T)
        # --------------------------------------------------------------------------------------------------------------
        # 2. Apply bottleneck Block
        for m in self.mattn_layers:
            x = m(x)  # (B,C,F,T)
        # --------------------------------------------------------------------------------------------------------------
        # 3. Apply Decoder
        x = self.decoder(x).permute(0, 2, 3, 1) # (B,F,T,2)

        return x


class GAB(nn.Module):
    """global time-frequency-channel attention block"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.t_gru = nn.GRU(channels, channels * 2, 1, batch_first=True, bidirectional=True)
        self.t_fc = nn.Linear(channels * 4, channels)
        self.f_gru = nn.GRU(channels, channels * 2, 1, batch_first=True, bidirectional=True)
        self.f_fc = nn.Linear(channels * 4, channels)
        self.eca = ECA_layer()

    def forward(self, x):
        """x: (B,C,F,T)"""
        zt = torch.mean(x.pow(2), dim=-2)  # (B,C,T)
        at = self.t_gru(zt.transpose(1, 2))[0]
        at = self.t_fc(at).transpose(1, 2)  # (B,C,T)
        at = torch.sigmoid(at)

        zf = torch.mean(x.pow(2), dim=-1)  # (B,C,F)
        af = self.f_gru(zf.transpose(1, 2))[0]
        af = self.f_fc(af).transpose(1, 2)  # (B,C,F)
        af = torch.sigmoid(af)

        ac = self.eca(x)  # (B,C,F,T)

        att = ac * af.unsqueeze(-1) * at.unsqueeze(2)  # (B,C,F,T)

        return x * att


class ECA_layer(nn.Module):
    """Constructs a ECA module.
    """
    def __init__(self, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y.expand_as(x)

class cross_block(nn.Module):

    def __init__(self,
                 embed_dim=96,
                 dim_squeeze=8,
                 num_freqs=257,
                 bidirectional=False,):
        super().__init__()
        self.norm1 = LayerNorm(normalized_shape=embed_dim, seq_last=True)
        self.rnn = nn.LSTM(embed_dim, embed_dim * 2, 1, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.dense = nn.Linear(embed_dim * 4, embed_dim)
        else:
            self.dense = nn.Linear(embed_dim * 2, embed_dim)
        self.norm2 = LayerNorm(normalized_shape=embed_dim, seq_last=True)
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=embed_dim, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze)
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=embed_dim, kernel_size=1), nn.SiLU())

    def forward(self, x):
        """x: (B,C,F,T)"""
        B, C, F, T = x.size()
        x_resi1 = x
        x = self.norm1(x)  #(B,C,F,T)
        x = x.permute(0, 3, 2, 1)  # (B,T,F,C)
        x = x.reshape(B * T, F, C)  # (B*T,F,C)
        x, _ = self.rnn(x)
        x = self.dense(x)  # (B*T,F,C)
        x = x.reshape(B, T, F, C).permute(0, 3, 2, 1)  #(B,C,F,T)
        x = x + x_resi1  #(B,C,F,T)

        x_resi2 = x
        x = self.norm2(x)  #(B,C,F,T)
        x = x.permute(0, 3, 1, 2)  # [B,T,C,F]
        x = x.reshape(B * T, C, F)
        x = self.squeeze(x)  # [B*T,C',F]
        x = self.full(x)  # [B*T,C',F]
        x = self.unsqueeze(x)  # [B*T,C,F]
        x = x.reshape(B, T, C, F)
        x = x.permute(0, 2, 3, 1)  # [B,C,F,T]
        x = x + x_resi2  # [B,C,F,T]

        return x

class LinearGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"

class LayerNorm(nn.LayerNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        """
        Arg s:
            seq_last (bool): whether the sequence dim is the last dim
        """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o

class narrow_block(nn.Module):
    def __init__(self,
                 embed_dim=96,):
        super().__init__()
        self.conv_att = AttentionGroup(embed_dim)

    def forward(self, x):
        """x: (B,C,F,T)"""
        B, C, F, T = x.size()
        x = x.permute(0, 2, 1, 3) # (B,F,C,T)
        x = x.reshape(B * F, C, T)  # (B*F,C,T)
        x = self.conv_att(x)  # (B*F,C,T)
        x = x.view(B, F, C, T).permute(0, 2, 1, 3).contiguous()  # (B,C,F,T)

        return x


class DGC(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        hidden_dim = emb_dim * 2
        self.fc1 = nn.Sequential(
            nn.InstanceNorm1d(emb_dim),
            nn.Conv1d(emb_dim, hidden_dim, 1),
            nn.SiLU()
        )
        self.left_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, 1, padding=1, groups=hidden_dim)

        self.right_conv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, padding=1, groups=hidden_dim),
                                        nn.Sigmoid()
        )
        self.fc2 = nn.Conv1d(hidden_dim, emb_dim, 1)

    def forward(self, x):
        # x:(BF,C,T)
        x = self.fc1(x)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.fc2(x)
        return x


class AttentionGroup(nn.Module):
    def __init__(self, d_model: int):
        super(AttentionGroup, self).__init__()
        self.dilation_list = [32, 16, 8, 4, 2, 1]
        self.att_list = nn.ModuleList([Attention(d_model, i) for i in self.dilation_list])
        self.dgc = DGC(d_model)

    def forward(self, x):
        x_resi = x
        for i in range(len(self.dilation_list)):
            x = self.att_list[i](x)
        x = x + x_resi
        x = x + self.dgc(x)
        return x

class Attention(nn.Module):
    def __init__(self, d_model, dilation):
        super().__init__()
        self.proj_1 = nn.Conv1d(d_model, d_model//2, 1)
        self.activation = nn.SiLU()
        self.spatial_gating_unit = LKA(d_model//2, dilation)
        self.proj_2 = nn.Conv1d(d_model//2, d_model, 1)
        self.norm = nn.InstanceNorm1d(d_model)  

    def forward(self, x):
        shorcut = x
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class LKA(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        pad = 3 * dilation
        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, 7, stride=1, padding=pad, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv1d(dim, dim, 1)

    def forward(self, x): #(B,C,N)
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class MulSE_layer(nn.Module):
    def __init__(self,
                 embed_dim,
                 dim_squeeze,
                 num_freqs,
                 bidirectional,
                 ):
        super().__init__()
        self.cross_block = cross_block(embed_dim=embed_dim, dim_squeeze=dim_squeeze, num_freqs=num_freqs, bidirectional=bidirectional)
        self.narrow_block = narrow_block(embed_dim=embed_dim)
        self.gab = GAB(channels=embed_dim)
        self.norm = LayerNorm(normalized_shape=embed_dim, seq_last=True)

    def forward(self, x: Tensor) -> tuple:

        x = self.gab(self.norm(x))
        x = self.cross_block(x)
        x = self.narrow_block(x)

        return x
#
if __name__ == '__main__':
    dur = 4
    fs = 16000
    hop_ms = 16
    hop = int(fs * hop_ms / 1000)  # 16 ms -> 256 samples
    T = int(dur * fs / hop) + 1
    F = 257
    M = 6

    x_real = torch.randn((1, F, T, M))  # 251 = 4 second
    x_imag = torch.randn((1, F, T, M))
    x = torch.complex(x_real, x_imag).cuda()

    model = MulSE(in_ch=12,
                  out_ch=64,
                  dim_squeeze=8,
                  num_freqs=257,
                  num_layers=5
                  ).cuda()

    with FlopCounterMode(display=False) as fcm: 
        y = model(x)
        print(y.shape)
        flops_forward_eval = fcm.get_total_flops()  
    params_eval = sum(param.numel() for param in model.parameters())
    print(f"flops_forward={flops_forward_eval / 1e9:.3f}G")
    print(f"Avg_FLOPs={flops_forward_eval / 1e9/dur:.3f}G/s, params={params_eval / 1e6:.3f} M")





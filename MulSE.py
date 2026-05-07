import torch
import torch.nn as nn
from torch import Tensor
from modules.channel_attention import ECA_layer
from modules.spatial_attention import Block
from modules.Attengroup import AttentionGroup
from modules.linear_group import LinearGroup
from modules.norm import NormSwitch, LayerNorm
from torch.utils.flop_counter import FlopCounterMode
import torch.nn.functional as Func

class MulSE(nn.Module):

    def __init__(
            self,
            in_ch=12,
            out_ch=64,
            dim_squeeze=8,
            num_freqs = 257,
            num_layers: int = 3,
    ):
        super().__init__()

        self.encoder = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 5), stride=1, padding=(0,2))
        self.decoder = nn.ConvTranspose2d(out_ch, 2, kernel_size=(1, 5), stride=1, padding=(0,2))

        mattn_layers = []
        for n in range(num_layers):
            mattn_layer = Mattn(
                embed_dim=out_ch,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                bidirectional=True
            )
            mattn_layers.append(mattn_layer)
        self.mattn_layers = nn.ModuleList(mattn_layers)

    def forward(self, noisy_input: Tensor) -> tuple:  # noisy_input------> (B,F,T,C)

        B, F, T, C = noisy_input.shape
        noisy_ri = torch.view_as_real(noisy_input)  # (B,F,T,C,2)
        noisy_ipt = noisy_ri.reshape(B, F, T, C * 2).permute(0, -1, 1, 2) # (B,C*2,F,T)
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

    def beamform_sum(self, bf, noisy): #bf--->(B,F,T,C,2)  noisy--->(B,F,T,C,2)
        bf_r, bf_i = bf[..., 0], bf[..., 1]  #(B,F,T,C)
        noisy_r, noisy_i = noisy[..., 0], noisy[..., 1]  # (B,F,T,C)
        est_r, est_i = (bf_r * noisy_r - bf_i * noisy_i).sum(dim=-1), \
               (bf_r * noisy_i + bf_i * noisy_r).sum(dim=-1)

        out_spec = torch.stack((est_r, est_i), dim=-1)  # (B,F,T,2)
        return out_spec


class TFCA(nn.Module):
    """time-frequency-channel attention"""

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

class narrow_block(nn.Module):

    def __init__(self,
                 embed_dim=96,):
        super().__init__()
        self.conv_att = AttentionGroup(embed_dim, "1D")

    def forward(self, x):
        """x: (B,C,F,T)"""
        B, C, F, T = x.size()
        x = x.permute(0, 2, 1, 3) # (B,F,C,T)
        x = x.reshape(B * F, C, T)  # (B*F,C,T)
        x = self.conv_att(x)  # (B*F,C,T)
        x = x.view(B, F, C, T).permute(0, 2, 1, 3).contiguous()  # (B,C,F,T)

        return x


class Mattn(nn.Module):
    def __init__(self,
                 embed_dim,
                 dim_squeeze,
                 num_freqs,
                 bidirectional,
                 ):
        super().__init__()
        self.cross_block = cross_block(embed_dim=embed_dim, dim_squeeze=dim_squeeze, num_freqs=num_freqs, bidirectional=bidirectional)
        self.narrow_block = narrow_block(embed_dim=embed_dim)
        self.att_tf = TFCA(channels=embed_dim)
        self.norm = LayerNorm(normalized_shape=embed_dim, seq_last=True)

    def forward(self, x: Tensor) -> tuple:

        x = self.att_tf(self.norm(x))
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

    with FlopCounterMode(display=False) as fcm:  # display=False 表示不直接显示 FLOPs 的中间结果
        y = model(x)
        print(y.shape)
        flops_forward_eval = fcm.get_total_flops()  # 获取前向传播完成后累计的 FLOPs
    params_eval = sum(param.numel() for param in model.parameters())
    print(f"flops_forward={flops_forward_eval / 1e9:.3f}G")
    print(f"Avg_FLOPs={flops_forward_eval / 1e9/dur:.3f}G/s, params={params_eval / 1e6:.3f} M")

    """complexity count"""
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, (257, 63, 6), as_strings=True,
    #                                           print_per_layer_stat=True, verbose=True)
    # print(flops, params)




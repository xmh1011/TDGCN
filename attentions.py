import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout, Linear
import math

class AttentionBlock(nn.Module):
    def __init__(self, attention_model, in_channels, ratio=8, residual=False, apply_to_input=True, key_dim=8, num_heads=2, dropout=0.5):
        super().__init__()
        self.attention_model = attention_model
        self.residual = residual
        self.apply_to_input = apply_to_input
        self.expanded_axis = 2  # default = 2
        if attention_model == 'mha':
            self.attention_layer = MHA_Block(in_channels, key_dim, num_heads, dropout)
        elif attention_model == 'mhla':
            self.attention_layer = MHA_Block(in_channels, key_dim, num_heads, dropout, vanilla=False)
        elif attention_model == 'se':
            self.attention_layer = SE_Block(in_channels, ratio)
        elif attention_model == 'cbam':
            self.attention_layer = CBAM_Block(in_channels, ratio)
        else:
            raise ValueError(f"'{attention_model}' is not a supported attention module!")

    def forward(self, x):
        return self.attention_layer(x)


class MHA_Block(nn.Module):
    def __init__(self, in_channels, key_dim, num_heads, dropout=0.5, vanilla=True):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.dropout = Dropout(dropout)
        self.vanilla = vanilla
        if vanilla:
            self.mha = MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout)
        else:
            self.mha = MultiHeadAttention_LSA(embed_dim=in_channels, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        if self.vanilla:
            attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        else:
            attn_output = self.mha(x_norm)
        attn_output = self.dropout(attn_output)
        return x + attn_output


class SE_Block(nn.Module):
    def __init__(self, channel, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM_Block(nn.Module):
    def __init__(self, channel, ratio=8):
        super().__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())
        max_out = self.fc(self.max_pool(x).squeeze())
        out = avg_out + max_out
        return x * out.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return

class MultiHeadAttention_LSA(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim)
        self.tau = nn.Parameter(torch.sqrt(torch.tensor(float(embed_dim))))

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # Scale query for local self-attention
        query = query / self.tau

        # Call the parent class's forward method
        attn_output, attn_output_weights = super().forward(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)

        return attn_output

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0)."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
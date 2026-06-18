import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class VLTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class VLTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output

class VLTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()

        encoder_layer = VLTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="relu", normalize_before=normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = VLTransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        return self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)


def build_vl_transformer(config):
    return VLTransformer(
        d_model=config.vl_hidden_dim,
        nhead=config.vl_nheads,
        num_encoder_layers=config.vl_enc_layers,
        dim_feedforward=config.vl_dim_feedforward,
        dropout=config.vl_dropout,
        normalize_before=False,
    )


# # test
# if __name__ == "__main__":
#     print("=== Test VL Transformer ===\n")

#     model = VLTransformer(d_model=256, nhead=8, num_encoder_layers=6)

#     total = sum(p.numel() for p in model.parameters())
#     print(f"Total params: {total:,}")

#     B = 2
#     L = 418
#     src = torch.randn(L, B, 256)
#     mask = torch.zeros(B, L, dtype=torch.bool)

#     mask[0, 7:18] = True
#     mask[1, 5:18] = True
#     pos = torch.randn(L, B, 256)

#     output = model(src, mask, pos)
#     print(f"Input shape:  {src.shape}")     
#     print(f"Output shape: {output.shape}")  

#     reg_output = output[0] 
#     print(f"[REG] output: {reg_output.shape}")

#     print("VL Transformer test passed")

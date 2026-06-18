import copy
import math
from collections import OrderedDict
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from utils.misc import NestedTensor


class FrozenBatchNorm2d(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'num_batches_tracked'
        if key in state_dict:
            del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class Backbone(nn.Module):
    def __init__(self, name='resnet50', dilation=False):
        super().__init__()

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights='IMAGENET1K_V1',
            norm_layer=FrozenBatchNorm2d
        )

        for name_param, parameter in backbone.named_parameters():
            if 'layer2' not in name_param and 'layer3' not in name_param and 'layer4' not in name_param:
                parameter.requires_grad_(False)

        return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = 2048

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)  
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  
        pos_y = y_embed[:, :, :, None] / dim_t  
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Detr

class TransformerEncoderLayer(nn.Module):
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


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


# Visual Encoder

class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = Backbone(name=config.backbone, dilation=config.dilation)

        N_steps = config.hidden_dim // 2
        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

        if config.detr_enc_num > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.nheads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                normalize_before=config.pre_norm
            )
            encoder_norm = nn.LayerNorm(config.hidden_dim) if config.pre_norm else None
            self.transformer = TransformerEncoder(
                encoder_layer, config.detr_enc_num, encoder_norm
            )
            self.input_proj = nn.Conv2d(
                self.backbone.num_channels, config.hidden_dim, kernel_size=1
            )
        else:
            self.transformer = None

        if self.transformer is not None:
            self.num_channels = config.hidden_dim  
        else:
            self.num_channels = self.backbone.num_channels

        self._reset_transformer_parameters()

    def _reset_transformer_parameters(self):
        if self.transformer is not None:
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, img_data: NestedTensor):

        features = self.backbone(img_data)
        src_nested = list(features.values())[-1] 
        src, mask = src_nested.decompose()         
        pos = self.position_embedding(src_nested)  

        assert mask is not None

        if self.transformer is not None:
            src = self.input_proj(src)  

            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)    
            pos = pos.flatten(2).permute(2, 0, 1)    
            mask = mask.flatten(1)                   

            memory = self.transformer(src, src_key_padding_mask=mask, pos=pos)

            return mask, memory
        else:
            mask = mask.flatten(1)                     
            src = src.flatten(2).permute(2, 0, 1)     
            return mask, src


def build_visual_encoder(config):
    return VisualEncoder(config)


# # Test
# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, '.')
#     from config import Config

#     print("Test Visual Encoder")

#     # Build model
#     model = VisualEncoder(Config)
#     print(f"Output channels: {model.num_channels}")

#     # Count parameters
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     frozen = total - trainable
#     print(f"Total params:{total:,}")
#     print(f"Trainable params:{trainable:,}")
#     print(f"Frozen params:{frozen:,}")

#     # Test forward
#     B = 2
#     img = torch.randn(B, 3, 640, 640)
#     mask = torch.zeros(B, 640, 640, dtype=torch.bool)
#     # Giả lập: ảnh thứ 2 có padding bên phải
#     mask[1, :, 500:] = True

#     img_data = NestedTensor(img, mask)
#     visu_mask, visu_src = model(img_data)

#     print(f"\nvisu_mask shape: {visu_mask.shape}")  
#     print(f"visu_src shape:  {visu_src.shape}")     
#     print(f"visu_mask dtype: {visu_mask.dtype}")     

#     print("Visual Encoder test passed")

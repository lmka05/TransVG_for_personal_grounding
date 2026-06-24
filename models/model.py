import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import NestedTensor

from .backbone import build_visual_encoder
from .language import build_bert_encoder
from .vl_transformer import build_vl_transformer


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransVG(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.vl_hidden_dim  
        divisor = 16 if config.dilation else 32
        self.num_visu_token = int((config.imsize / divisor) ** 2)  
        self.num_text_token = config.max_query_len + 2             

        # Visual Encoder (ResNet + DETR)
        self.visumodel = build_visual_encoder(config)

        # BERT
        self.textmodel = build_bert_encoder(config)

        # Projection layers
        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)  # 256→256
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)  # 768→256

        # Special tokens
        num_total = 1 + self.num_text_token + self.num_visu_token  
        self.reg_token = nn.Embedding(1, hidden_dim)               
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)   

        # VL Transformer
        self.vl_transformer = build_vl_transformer(config)

        # MLP Head -> bbox
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, img_tensors, img_mask, word_ids, word_mask):
        bs = img_tensors.shape[0]
        img_data = NestedTensor(img_tensors, img_mask)
        text_data = NestedTensor(word_ids, word_mask)
        visu_mask, visu_src = self.visumodel(img_data)
        visu_src = self.visu_proj(visu_src)  # [400, B, 256] (project nếu cần)


        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()

        text_src = self.text_proj(text_src)   
        text_src = text_src.permute(1, 0, 2)  
        text_mask = text_mask.flatten(1)      

        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)  
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)  

        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)   
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)  
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)  

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)
        vg_hs = vg_hs[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box



# # test
# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, '.')
#     from config import Config
#     from utils.misc import NestedTensor

#     print("Test TransVG Full Model")

#     model = TransVG(Config)

#     # Count parameters
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     frozen = total - trainable
#     print(f"Total params:{total:,}")
#     print(f"Trainable params:{trainable:,}")
#     print(f"Frozen params:{frozen:,}")
#     print(f"num_visu_token:{model.num_visu_token}")
#     print(f"num_text_token:{model.num_text_token}")

#     # Test forward
#     B = 2
#     img = torch.randn(B, 3, 640, 640)
#     img_mask = torch.zeros(B, 640, 640, dtype=torch.bool)
#     img_data = NestedTensor(img, img_mask)

#     word_ids = torch.randint(100, 30000, (B, 17))
#     word_mask = torch.ones(B, 17, dtype=torch.long)
#     word_mask[0, 8:] = 0  # Câu 1: 8 tokens thật
#     word_mask[1, 5:] = 0  # Câu 2: 5 tokens thật
#     text_data = NestedTensor(word_ids, word_mask)

#     model.eval()
#     with torch.no_grad():
#         pred_box = model(img_data, text_data)

#     print(f"\npred_box shape: {pred_box.shape}")  # [2, 4]
#     print(f"pred_box:       {pred_box}")           # values ∈ [0, 1]
#     print(f"pred_box range: [{pred_box.min():.4f}, {pred_box.max():.4f}]")

#     print("TransVG full model test passed")

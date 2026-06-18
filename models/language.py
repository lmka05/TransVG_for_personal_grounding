import torch
from torch import nn
from transformers import AutoModel
from utils.misc import NestedTensor


class BERTEncoder(nn.Module):
    def __init__(self, bert_model='vinai/phobert-base', train_bert=True, enc_num=12):
        super().__init__()

        if 'base' in bert_model:
            self.num_channels = 768
        else:
            self.num_channels = 1024

        self.enc_num = enc_num

        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(bert_model)

        # Freeze BERT nếu không fine-tune
        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, text_data: NestedTensor):
        input_ids = text_data.tensors      
        attention_mask = text_data.mask      

        if self.enc_num > 0:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
            xs = outputs.hidden_states[self.enc_num]
        else:
            xs = self.bert.embeddings.word_embeddings(input_ids)

        mask = attention_mask.to(torch.bool)
        mask = ~mask

        return NestedTensor(xs, mask)


def build_bert_encoder(config):
    train_bert = config.lr_bert > 0
    return BERTEncoder(
        bert_model=config.bert_model,
        train_bert=train_bert,
        enc_num=config.bert_enc_num
    )


# # test
# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, '.')
#     from config import Config

#     print("=== Test BERT Encoder ===\n")

#     model = BERTEncoder(
#         bert_model='bert-base-uncased',
#         train_bert=True,
#         enc_num=12
#     )
#     print(f"num_channels: {model.num_channels}")

#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total params: {total:,}")
#     print(f"Trainable params: {trainable:,}")

#     # Test forward
#     B = 2
#     word_ids = torch.tensor([
#         [101, 1996, 2158, 1999, 2417, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [101, 2187, 3899, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])
#     word_mask = torch.tensor([
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])

#     text_data = NestedTensor(word_ids, word_mask)
#     output = model(text_data)

#     text_src, text_mask = output.decompose()
#     print(f"\ntext_src shape:  {text_src.shape}")   
#     print(f"text_mask shape: {text_mask.shape}")     
#     print(f"text_mask[0]:    {text_mask[0]}")         
#     print(f"text_mask dtype: {text_mask.dtype}")      

#     print("BERT Encoder test passed")

class Config:
    img_dir = ""
    ann_train = "/path/to/train.json"
    ann_dev = "/path/to/dev.json"
    ann_test = "/path/to/test.json"
    detr_model = ""
    # Kích thước đầu vào
    imsize = 640            
    max_query_len = 25     

    # Visual backbone
    backbone = "resnet50"
    dilation = False         # False -> stride=32, feature map 20×20
    hidden_dim = 256         # Dimension sau Conv1x1 (2048 -> 256)
    nheads = 8               # Số attention heads trong DETR Encoder
    dim_feedforward = 2048   # FFN hidden dim trong DETR Encoder
    dropout = 0.1
    pre_norm = False         # Post-norm (mặc định)
    detr_enc_num = 6         # Số encoder layers trong DETR
    position_embedding = "sine"  # Loại positional encoding ("sine" = sin/cos, không cần train)

    # Language model (Monosyllabic / Space-separated Vietnamese)
    # Các lựa chọn khác: "xlm-roberta-base" hoặc "bert-base-multilingual-cased"
    bert_model = "uitnlp/CafeBERT"
    bert_enc_num = 12        # Dùng output layer thứ 12

    # VL Transformer
    vl_hidden_dim = 256
    vl_nheads = 8
    vl_enc_layers = 6
    vl_dim_feedforward = 2048
    vl_dropout = 0.1

    # 6. TRAINING
    optimizer = "adamw"      
    lr          = 1.25e-5    # LR cho VL Transformer + MLP  
    lr_bert     = 1.25e-6    # LR cho BERT                  
    lr_visu_cnn = 1.25e-6    # LR cho ResNet backbone        
    lr_visu_tra = 1.25e-6    # LR cho DETR Encoder           
    weight_decay = 1e-4
    batch_size = 8
    epochs = 30
    lr_scheduler = "step"    # "step", "cosine" 
    lr_drop = 60             
    clip_max_norm = 0.15     # Gradient clipping

    # 7. LOGGING & CHECKPOINT
    log_interval = 80        # In log mỗi N batches
    output_dir = "/kaggle/working/transvg_outputs"
    resume = ""              # Đường dẫn checkpoint để resume training

    # 8. MISC
    seed = 13
    num_workers = 0
    device = "cuda"

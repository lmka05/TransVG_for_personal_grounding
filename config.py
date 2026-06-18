class Config:
    img_dir = "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014"
    ann_train = "/path/to/train.json"
    ann_dev = "/path/to/dev.json"
    ann_test = "/path/to/test.json"
    detr_model = "/kaggle/input/datasets/minhkhoai/detr-pretrained/detr-r50-e632da11.pth"
    # Kích thước đầu vào
    imsize = 640            
    max_query_len = 15       # Số từ tối đa trong câu (không tính [CLS], [SEP])

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

    # bert
    bert_model = "vinai/phobert-base"
    bert_enc_num = 12        # Dùng output layer thứ 12

    # VL Transformer
    vl_hidden_dim = 256
    vl_nheads = 8
    vl_enc_layers = 6
    vl_dim_feedforward = 2048
    vl_dropout = 0.1

    # 6. TRAINING
    optimizer = "adamw"      # Optimizer (chỉ dùng AdamW)
    lr          = 1.25e-5    # LR cho VL Transformer + MLP  (1e-4  × 8/64)
    lr_bert     = 1.25e-6    # LR cho BERT                  (1e-5  × 8/64)
    lr_visu_cnn = 1.25e-6    # LR cho ResNet backbone        (1e-5  × 8/64)
    lr_visu_tra = 1.25e-6    # LR cho DETR Encoder           (1e-5  × 8/64)
    weight_decay = 1e-4
    batch_size = 8
    epochs = 30
    lr_scheduler = "step"    # "step", "cosine" 
    lr_drop = 60             # Epoch giảm lr (cho step scheduler)
    clip_max_norm = 0.15     # Gradient clipping

    # 7. LOGGING & CHECKPOINT
    log_interval = 80        # In log mỗi N batches
    output_dir = "/kaggle/working/transvg_outputs"
    resume = ""              # Đường dẫn checkpoint để resume training

    # 8. MISC
    seed = 13
    num_workers = 0
    device = "cuda"

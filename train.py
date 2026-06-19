import os
import torch.nn as nn
import sys
import math
import time
import json
import random
import gc

import numpy as np
import torch

from config import Config
from models import build_model
from torch.utils.data import DataLoader
from datasets.dataset import CustomGroundingDataset
from datasets.dataloader import build_dataloader
from utils.misc import collate_fn
from evaluate import trans_vg_loss, evaluate



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, best_accuracy, config):
    os.makedirs(config.output_dir, exist_ok=True)

    raw_model = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
        'best_accuracy': best_accuracy,
    }

    latest_path = os.path.join(config.output_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)

    if accuracy >= best_accuracy:
        best_path = os.path.join(config.output_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"  ★ New best model saved! Acc: {accuracy:.2f}%")



def train_one_epoch(model, dataloader, optimizer, device, epoch, config):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (img_data, text_data, target) in enumerate(dataloader):
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        pred_box = model(img_data.tensors, img_data.mask,
                         text_data.tensors, text_data.mask)  # [B, 4]

        losses = trans_vg_loss(pred_box, target)
        loss = losses['loss_bbox'] + losses['loss_giou']

        if loss.dim() > 0:
            loss = loss.mean()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
        optimizer.step()

        total_loss += loss_value
        num_batches += 1

        del img_data, text_data, target, pred_box, losses, loss
        if batch_idx % 100 == 0:
            gc.collect()
            
        if (batch_idx + 1) % config.log_interval == 0:
            avg = total_loss / num_batches
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    return avg_loss




def main():
    config = Config

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Building model")
    model = build_model(config)
    model.to(device)

    if os.path.exists(config.detr_model):
        print(f"Loading DETR pretrain from {config.detr_model}")
        checkpoint = torch.load(config.detr_model, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        new_state_dict = {}
        
        # Đổi tên key cho khớp với kiến trúc VisualEncoder
        for k, v in state_dict.items():
            if k.startswith('backbone.0.body'):
                new_key = k.replace('backbone.0.body', 'backbone.body')
            elif k.startswith('transformer.encoder.'):
                new_key = k.replace('transformer.encoder.', 'transformer.')
            else:
                new_key = k
            new_state_dict[new_key] = v

        missing_keys, unexpected_keys = model.visumodel.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print("Đã tải thành công tệp DETR pretrain cho ResNet-50 và DETR Encoder")
    else:
        print(f"Không tìm thấy file DETR pretrain tại: {config.detr_model}")
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPU")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU")

    print("\n" + "=" * 60)
    print("STEP 2: Setting up optimizer")
    print("=" * 60)

    # ResNet backbone -> fine-tune
    visu_cnn_param = [p for n, p in model.named_parameters()
                      if "visumodel" in n and "backbone" in n and p.requires_grad]
    # DETR Encoder -> fine-tune
    visu_tra_param = [p for n, p in model.named_parameters()
                      if "visumodel" in n and "backbone" not in n and p.requires_grad]
    # BERT -> fine-tune
    text_tra_param = [p for n, p in model.named_parameters()
                      if "textmodel" in n and p.requires_grad]
    # VL Transformer
    rest_param = [p for n, p in model.named_parameters()
                  if "visumodel" not in n and "textmodel" not in n and p.requires_grad]

    param_list = [
        {"params": rest_param,"lr": config.lr},
        {"params": visu_cnn_param,"lr": config.lr_visu_cnn},
        {"params": visu_tra_param,"lr": config.lr_visu_tra},
        {"params": text_tra_param,"lr": config.lr_bert},
    ]

    print(f"  rest (VL Trans+MLP):{sum(p.numel() for p in rest_param):>10,} params, lr={config.lr}")
    print(f"  visu_cnn (ResNet):{sum(p.numel() for p in visu_cnn_param):>10,} params, lr={config.lr_visu_cnn}")
    print(f"  visu_tra (DETR Enc):{sum(p.numel() for p in visu_tra_param):>10,} params, lr={config.lr_visu_tra}")
    print(f"  text_tra (BERT):{sum(p.numel() for p in text_tra_param):>10,} params, lr={config.lr_bert}")

    optimizer = torch.optim.AdamW(param_list, lr=config.lr, weight_decay=config.weight_decay)

    # LR Scheduler
    if config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    elif config.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / config.epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")

    # build dataloaders
    print("Creating datasets")
    train_dataset = CustomGroundingDataset(config.ann_train, config.img_dir, 'train', config)
    train_loader = build_dataloader(train_dataset, config.batch_size, True, config.num_workers)
    val_dataset = CustomGroundingDataset(config.ann_dev, config.img_dir, 'dev', config)
    val_loader = build_dataloader(val_dataset, config.batch_size, False, config.num_workers)

    # Resume from checkpoint
    start_epoch = 0
    best_accuracy = 0.0
    latest_ckpt = os.path.join(config.output_dir, 'latest.pth')

    if config.resume and os.path.isfile(config.resume):
        latest_ckpt = config.resume

    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_accuracy = ckpt.get('best_accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_accuracy:.2f}%")

    # Training loop
    print("Start training")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config)

        # Evaluate
        print(f"Evaluating epoch {epoch+1}")
        val_acc, val_iou = evaluate(model, val_loader, device, desc="val")

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_acc, best_accuracy, config)
        best_accuracy = max(best_accuracy, val_acc)

        # Step scheduler
        scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs} Summary:")
        print(f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Best: {best_accuracy:.2f}% | LR: {lr:.6f} | Time: {epoch_time:.0f}s")
        print(f"{'='*60}\n")

    print(f"Training finished/Best accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoints saved at: {config.output_dir}")


if __name__ == "__main__":
    main()

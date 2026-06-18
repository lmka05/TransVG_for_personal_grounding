import os
import sys
import argparse

import torch

from config import Config
from models import build_model
from torch.utils.data import DataLoader
from datasets.dataset import CustomGroundingDataset
from utils.misc import collate_fn
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='TransVG — Test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Đường dẫn tới file checkpoint (.pth)')
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'])
    args = parser.parse_args()

    config = Config
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Splits: {args.splits}")

    # Build model
    print("Building model")
    model = build_model(config)
    model.to(device)

    # Load checkpoint
    print("Loading checkpoint")
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', '?')
    best_acc = ckpt.get('best_accuracy', '?')
    print(f"Loaded checkpoint: epoch={epoch}, best_acc={best_acc}")

    # Evaluate từng split
    results = {}
    for split in args.splits:
        print(f"Evaluating on [{split}]")
        try:
            ann_map = {'dev': config.ann_dev, 'test': config.ann_test}
            if split not in ann_map:
                print(f"Split '{split}' không được hỗ trợ.")
                continue
            dataset = CustomGroundingDataset(ann_map[split], config.img_dir, split, config)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=False,
            )
            acc, avg_iou = evaluate(model, loader, device, desc=split)
            results[split] = {'accuracy': acc, 'avg_iou': avg_iou}
        except KeyError:
            print(f"Split '{split}' không tồn tại trong file annotations.")

    # Tổng kết 
    print("Kết quả")
    for split, res in results.items():
        print(f"{split:8s}: Acc@IoU>=0.5 = {res['accuracy']:.2f}% | "
              f"Avg IoU = {res['avg_iou']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

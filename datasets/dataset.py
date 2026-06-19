import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.image_transforms import make_transforms
from utils.text_transforms import TextTransform


class CustomGroundingDataset(Dataset):
    """
    Mỗi sample = 1 cặp (ảnh + bbox + câu mô tả).
    Một ảnh có thể có nhiều bbox → tạo ra nhiều samples.

    Returns (mỗi sample):
        img:       Tensor [3, imsize, imsize]  — ảnh đã augment + normalize + pad
        img_mask:  Tensor [imsize, imsize]     — True=padding, False=pixel thật
        word_id:   Tensor [max_query_len+2]    — token IDs (PhoBERT)
        word_mask: Tensor [max_query_len+2]    — 1=token thật, 0=padding
        bbox:      Tensor [4]                  — normalized [cx, cy, w, h] ∈ [0,1]
    """

    def __init__(self, ann_file, img_dir, split, config):
        """
        Args:
            ann_file (str): Đường dẫn tới file JSON (ví dụ: train.json)
            img_dir (str): Thư mục chứa ảnh
            split (str): 'train', 'dev', hoặc 'test'
            config: Config object chứa hyperparameters
        """
        super().__init__()

        self.img_dir = img_dir
        self.split = split

        # =====================================================================
        # Load annotations + Flatten thành danh sách samples
        # =====================================================================
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = []
        for img_id, img_info in data.items():
            filename = img_info['filename']

            # Handle cả 2 key: "bboxes" hoặc "coordinates" ở cấp độ ảnh
            bbox_list = img_info.get('bboxes', img_info.get('coordinates', []))

            for bbox_item in bbox_list:
                # Handle cả 2 key "points" hoặc "coordinates" ở cấp độ bbox
                try:
                    points = bbox_item.get('points', bbox_item.get('coordinates'))
                except:
                    print("gặp lỗi ở item ", filename)
                    raise
                if points is None:
                    continue

                # Chuẩn hóa points thành tọa độ [x1, y1, x2, y2]
                if len(points) == 4 and not isinstance(points[0], (list, tuple)):
                    # Trường hợp points là flat list: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = points
                elif len(points) >= 2:
                    # Trường hợp points là list of [x, y] (ví dụ: [[x1, y1], [x2, y2]] hoặc 4 điểm góc)
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                else:
                    print("gặp lỗi ở item ", filename)
                    continue

                # Handle cả 2 key "description" hoặc "descriptions"
                descriptions = bbox_item.get('description', bbox_item.get('descriptions', []))
                # Nếu descriptions là single string, chuyển thành list của 1 string để chọn ngẫu nhiên
                if isinstance(descriptions, str):
                    descriptions = [descriptions]

                self.samples.append({
                    'filename': filename,
                    'bbox_xyxy': [x1, y1, x2, y2],
                    'descriptions': descriptions,
                })

        print(f"[{split}] Loaded {len(self.samples)} samples from {len(data)} images")

        # =====================================================================
        # Image transform: train có augmentation, dev/test thì không
        # =====================================================================
        aug_split = 'train' if split == 'train' else 'val'
        self.img_transform = make_transforms(config, aug_split)

        # Text transform (tokenize cho PhoBERT)
        self.text_transform = TextTransform(
            bert_model=config.bert_model,
            max_query_len=config.max_query_len
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # =====================================================================
        # Bước 1: Load ảnh (PIL Image)
        # =====================================================================
        img_path = os.path.join(self.img_dir, sample['filename'])
        pil_img = Image.open(img_path).convert('RGB')

        # =====================================================================
        # Bước 2: Lấy text expression
        # =====================================================================
        descriptions = sample['descriptions']
        if self.split == 'train':
            expression = random.choice(descriptions)
        else:
            expression = descriptions[0]

        # =====================================================================
        # Bước 3: Lấy bbox [x1, y1, x2, y2]
        # =====================================================================
        bbox_xyxy = sample['bbox_xyxy']

        # =====================================================================
        # Bước 4: Image transform (augment + normalize + pad + mask)
        # =====================================================================
        # Text có thể bị swap "trái"↔"phải" nếu ảnh bị flip
        img, img_mask, bbox, expression = self.img_transform(
            pil_img, bbox_xyxy, expression
        )

        # =====================================================================
        # Bước 5: Text transform (tokenize cho PhoBERT)
        # =====================================================================
        input_ids, attention_mask = self.text_transform(expression)
        word_id = torch.tensor(input_ids, dtype=torch.long)
        word_mask = torch.tensor(attention_mask, dtype=torch.long)

        return img, img_mask, word_id, word_mask, bbox


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config

    print("=" * 60)
    print("TEST VisualGroundingDataset")
    print("=" * 60)

    # Test với dữ liệu giả
    fake_data = {
        "000001": {
            "filename": "000001.jpg",
            "width": 800, "height": 600,
            "bboxes": [
                {
                    "points": [[100, 50], [300, 50], [300, 250], [100, 250]],
                    "description": ["Người đàn ông đứng bên trái"]
                }
            ]
        },
        "000002": {
            "filename": "000002.jpg",
            "width": 640, "height": 480,
            "coordinates": [
                {
                    "points": [[50, 100], [200, 100], [200, 400], [50, 400]],
                    "description": ["Con mèo ngồi trên bàn"]
                },
                {
                    "points": [[300, 50], [500, 50], [500, 300], [300, 300]],
                    "description": ["Cái bình hoa bên phải"]
                }
            ]
        }
    }

    # Parse samples
    samples = []
    for img_id, img_info in fake_data.items():
        filename = img_info['filename']
        bbox_list = img_info.get('bboxes', img_info.get('coordinates', []))
        for bbox_item in bbox_list:
            points = bbox_item['points']
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            samples.append({
                'filename': filename,
                'bbox_xyxy': [x1, y1, x2, y2],
                'descriptions': bbox_item['description'],
            })

    print(f"\nTotal samples: {len(samples)}")  # Expected: 3
    for i, s in enumerate(samples):
        print(f"  [{i}] {s['filename']}: bbox={s['bbox_xyxy']}, text='{s['descriptions'][0]}'")

    print("\n✅ Dataset parsing test passed!")

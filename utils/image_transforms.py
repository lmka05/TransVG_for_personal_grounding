import numpy as np
import torch
import random
from PIL import Image

import albumentations as A


# Danh sách từ chỉ hướng — nếu câu chứa từ này thì KHÔNG crop
DIRECTIONAL_WORDS = [
    "bên trái", "phía trái", "trái",
    "bên phải", "phía phải", "phải",
    "bên trên", "phía trên", "trên",
    "bên dưới", "phía dưới", "dưới",
    "ở giữa", "chính giữa", "giữa",
]

# Cặp swap khi lật ảnh ngang — thứ tự: cụm dài trước, từ đơn sau
SWAP_PAIRS = [
    ("bên trái", "bên phải"),
    ("phía trái", "phía phải"),
    ("trái", "phải"),
]


def has_directional_words(text):
    """Kiểm tra câu mô tả có chứa từ chỉ hướng không."""
    for word in DIRECTIONAL_WORDS:
        if word in text:
            return True
    return False


def swap_directional_words(text):
    """
    Swap trái↔phải trong text khi ảnh bị lật ngang.

    """
    for i, (w1, w2) in enumerate(SWAP_PAIRS):
        text = text.replace(w1, f"__PH_{i}_A__")
        text = text.replace(w2, f"__PH_{i}_B__")

    for i, (w1, w2) in enumerate(SWAP_PAIRS):
        text = text.replace(f"__PH_{i}_A__", w2)
        text = text.replace(f"__PH_{i}_B__", w1)

    return text


# ==============================================================================
# 2. Geometric transforms 
# ==============================================================================

def resize_long_side(img, bbox, target_size):
    """
    Resize ảnh sao cho cạnh dài = target_size, giữ tỷ lệ.

    Args:
        img:  numpy [H, W, 3] uint8
        bbox: [x1, y1, x2, y2] pixel coords
        target_size: int
    Returns:
        img:  numpy [new_H, new_W, 3]
        bbox: [x1, y1, x2, y2] đã scale
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img = np.array(pil_img)

    x1, y1, x2, y2 = bbox
    bbox = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]

    return img, bbox


def resize_short_side(img, bbox, target_size):
    """
    Resize ảnh sao cho cạnh ngắn = target_size, giữ tỷ lệ.
    Dùng trước bước RandomCrop (crop cần ảnh đủ lớn).
    """
    h, w = img.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img = np.array(pil_img)

    x1, y1, x2, y2 = bbox
    bbox = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]

    return img, bbox


def random_crop(img, bbox, min_size, max_size, max_tries=20):
    """
    Cắt ngẫu nhiên, đảm bảo tâm bbox nằm trong vùng cắt.
    Thử tối đa max_tries lần. Nếu thất bại, trả về ảnh gốc.

    Args:
        img:  numpy [H, W, 3]
        bbox: [x1, y1, x2, y2]
        min_size: kích thước cắt tối thiểu
        max_size: kích thước cắt tối đa
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 # toạ độ tâm bounding box

    for _ in range(max_tries):
        # chiều rộng, chiều cao vùng cắt ngẫu nhiên
        crop_w = random.randint(min_size, min(w, max_size))
        crop_h = random.randint(min_size, min(h, max_size))

        # Vị trí góc trên-trái ngẫu nhiên
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)

        # Kiểm tra tâm bbox có nằm trong vùng cắt không
        if left <= cx < left + crop_w and top <= cy < top + crop_h:
            # Cắt ảnh
            img = img[top:top + crop_h, left:left + crop_w].copy()

            # Dịch chuyển + clip bbox
            new_x1 = np.clip(x1 - left, 0, crop_w)
            new_y1 = np.clip(y1 - top, 0, crop_h)
            new_x2 = np.clip(x2 - left, 0, crop_w)
            new_y2 = np.clip(y2 - top, 0, crop_h)
            bbox = [new_x1, new_y1, new_x2, new_y2]

            return img, bbox

    # Thất bại sau max_tries lần → trả về gốc
    return img, [x1, y1, x2, y2]


def horizontal_flip(img, bbox):
    """
    Lật ảnh theo chiều ngang (trái↔phải).

    Args:
        img:  numpy [H, W, 3]
        bbox: [x1, y1, x2, y2]
    Returns:
        img:  numpy [H, W, 3] đã lật
        bbox: [x1, y1, x2, y2] đã lật
    """
    w = img.shape[1]
    img = img[:, ::-1, :].copy()

    x1, y1, x2, y2 = bbox
    bbox = [w - x2, y1, w - x1, y2]

    return img, bbox


# ==============================================================================
# 3. Color augmentation 
# ==============================================================================

def build_color_augmentation():
    """
    Tạo pipeline Albumentations cho color augmentation.

    ColorJitter: thay đổi brightness/contrast/saturation (80% prob)
    GaussianBlur: làm mờ Gaussian (50% prob)
    """
    return A.Compose([
        A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0,
            p=0.8,
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            sigma_limit=(0.1, 2.0),
            p=0.5,
        ),
    ])


# ==============================================================================
# 4. Final processing: Normalize + Pad + Mask + To Tensor
# ==============================================================================

def normalize_pad_to_tensor(img, bbox, target_size):
    """
    Bước cuối cùng: chuẩn hóa ImageNet, pad về hình vuông, tạo mask,
    chuyển bbox sang normalized [cx, cy, w, h].

    Args:
        img:  numpy [H, W, 3] uint8
        bbox: [x1, y1, x2, y2] pixel coords
        target_size: int (ví dụ 640)
    Returns:
        img_tensor: Tensor [3, target_size, target_size] float32
        mask:       Tensor [target_size, target_size] bool (True=padding)
        bbox_norm:  Tensor [4] float32 — normalized [cx, cy, w, h] ∈ [0, 1]
    """
    h, w = img.shape[:2]

    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
    padded[:h, :w, :] = img

    mask = torch.ones(target_size, target_size, dtype=torch.bool)
    mask[:h, :w] = False

    x1, y1, x2, y2 = bbox
    cx = np.clip((x1 + x2) / 2.0 / target_size, 0, 1)
    cy = np.clip((y1 + y2) / 2.0 / target_size, 0, 1)
    bw = np.clip((x2 - x1) / target_size, 0, 1)
    bh = np.clip((y2 - y1) / target_size, 0, 1)
    bbox_norm = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

    img_tensor = torch.from_numpy(
        np.ascontiguousarray(np.transpose(padded, (2, 0, 1)))
    )

    return img_tensor, mask, bbox_norm


# ==============================================================================
# 5. TransVGTransform 
# ==============================================================================

class ImageTransform:
    """
    Data augmentation pipeline cho

    Training pipeline:
        1. RandomSelect:
           - Nếu câu chứa từ chỉ hướng → chỉ resize (tránh crop làm sai ngữ nghĩa)
           - Nếu không → 50% resize, 50% resize+crop+resize
        2. ColorJitter + GaussianBlur (Albumentations)
        3. RandomHorizontalFlip (50%, swap "trái"↔"phải" trong text)
        4. Normalize + Pad (640×640) + Mask

    Val/Test pipeline:
        1. Resize cạnh dài = imsize
        2. Normalize + Pad + Mask
    """

    def __init__(self, config, split):
        self.split = split
        self.imsize = config.imsize

        if split == 'train':
            # Multi-scale sizes: [640, 608, 576, 544, 512, 480, 448]
            self.scales = [self.imsize - i * 32 for i in range(7)]

            # RandomSizeCrop settings
            self.crop_scales = [400, 500, 600]   # resize trước crop (short side)
            self.crop_min_size = 384
            self.crop_max_size = 600
            self.crop_prob = 0.5                  # xác suất chọn nhánh crop

            # Color augmentation
            self.color_aug = build_color_augmentation()

    def __call__(self, img_pil, bbox_xyxy, text):
        """
        Args:
            img_pil:   PIL Image (RGB)
            bbox_xyxy: list [x1, y1, x2, y2] pixel coords (tọa độ ảnh gốc)
            text:      str — câu mô tả tiếng Việt

        Returns:
            img_tensor: Tensor [3, imsize, imsize] float32
            img_mask:   Tensor [imsize, imsize] bool (True=padding, False=real)
            bbox:       Tensor [4] normalized [cx, cy, w, h] ∈ [0, 1]
            text:       str (có thể đã swap trái↔phải nếu bị flip)
        """
        img = np.array(img_pil)
        bbox = list(bbox_xyxy)

        if self.split == 'train':
            img, bbox, text = self._train_transform(img, bbox, text)
        else:
            img, bbox = self._val_transform(img, bbox)

        # Bước cuối: normalize + pad + mask + to tensor
        img_tensor, img_mask, bbox_norm = normalize_pad_to_tensor(
            img, bbox, self.imsize
        )

        return img_tensor, img_mask, bbox_norm, text

    def _train_transform(self, img, bbox, text):
        """Pipeline augmentation cho training."""

        use_crop = (
            not has_directional_words(text)
            and random.random() < self.crop_prob
        )

        if use_crop:
            # Nhánh crop: resize(short side) → crop → resize(long side)
            size1 = random.choice(self.crop_scales)
            img, bbox = resize_short_side(img, bbox, size1)

            img, bbox = random_crop(
                img, bbox, self.crop_min_size, self.crop_max_size
            )

            size2 = random.choice(self.scales)
            img, bbox = resize_long_side(img, bbox, size2)
        else:
            # Nhánh resize only
            size = random.choice(self.scales)
            img, bbox = resize_long_side(img, bbox, size)

        img = self.color_aug(image=img)['image']

        if random.random() < 0.5:
            img, bbox = horizontal_flip(img, bbox)
            text = swap_directional_words(text)

        return img, bbox, text

    def _val_transform(self, img, bbox):
        """Pipeline cho val/test — chỉ resize, không augment."""
        img, bbox = resize_long_side(img, bbox, self.imsize)
        return img, bbox


def make_transforms(config, split):
    """Tạo transform pipeline phù hợp với split (train/val/test)."""
    return ImageTransform(config, split)


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST IMAGE TRANSFORMS")
    print("=" * 60)

    # --- Test text helpers ---
    print("\n--- Test Vietnamese text helpers ---")
    text1 = "người đàn ông bên trái bức ảnh"
    print(f"Original:  '{text1}'")
    print(f"Has dir:   {has_directional_words(text1)}")  # True
    print(f"Swapped:   '{swap_directional_words(text1)}'")
    # Expected: "người đàn ông bên phải bức ảnh"

    text2 = "con mèo đang ngủ"
    print(f"\nOriginal:  '{text2}'")
    print(f"Has dir:   {has_directional_words(text2)}")  # False

    # --- Test geometric transforms ---
    print("\n--- Test geometric transforms ---")
    img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    bbox = [100, 50, 300, 250]  # [x1, y1, x2, y2]
    print(f"Original:  img={img.shape}, bbox={bbox}")

    img_r, bbox_r = resize_long_side(img, bbox, 640)
    print(f"Resize(640): img={img_r.shape}, bbox=[{bbox_r[0]:.1f}, {bbox_r[1]:.1f}, {bbox_r[2]:.1f}, {bbox_r[3]:.1f}]")

    img_f, bbox_f = horizontal_flip(img_r, bbox_r)
    print(f"Flip:      img={img_f.shape}, bbox=[{bbox_f[0]:.1f}, {bbox_f[1]:.1f}, {bbox_f[2]:.1f}, {bbox_f[3]:.1f}]")

    # --- Test full pipeline ---
    print("\n--- Test full pipeline (val) ---")

    class MockConfig:
        imsize = 640

    transform = ImageTransform(MockConfig, 'val')
    pil_img = Image.fromarray(img)
    img_t, mask_t, bbox_t, text_t = transform(pil_img, [100, 50, 300, 250], "con mèo")
    print(f"img_tensor: {img_t.shape}, dtype={img_t.dtype}")
    print(f"mask:       {mask_t.shape}, real={((~mask_t).sum().item())}, pad={mask_t.sum().item()}")
    print(f"bbox_norm:  {bbox_t}")
    print(f"text:       '{text_t}'")

    print("\n--- Test full pipeline (train) ---")
    transform_train = ImageTransform(MockConfig, 'train')
    img_t2, mask_t2, bbox_t2, text_t2 = transform_train(
        pil_img, [100, 50, 300, 250], "người bên trái"
    )
    print(f"img_tensor: {img_t2.shape}")
    print(f"bbox_norm:  {bbox_t2}")
    print(f"text:       '{text_t2}'")

    print("\n✅ image_transforms test passed!")

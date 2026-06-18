import torch


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        tensors = self.tensors.to(device)
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(tensors, mask)

    def decompose(self):
        return self.tensors, self.mask


def collate_fn(batch):

    imgs, img_masks, word_ids, word_masks, bboxes = zip(*batch)
    imgs = torch.stack(imgs, dim=0)           
    img_masks = torch.stack(img_masks, dim=0)
    word_ids = torch.stack(word_ids, dim=0)   
    word_masks = torch.stack(word_masks, dim=0)  
    bboxes = torch.stack(bboxes, dim=0)  
    img_data = NestedTensor(imgs, img_masks)
    text_data = NestedTensor(word_ids, word_masks)

    return img_data, text_data, bboxes


# # Test
# if __name__ == "__main__":
#     print("=== Test misc.py ===\n")

#     # Test NestedTensor
#     imgs = torch.randn(2, 3, 640, 640)
#     masks = torch.zeros(2, 640, 640, dtype=torch.bool)
#     nt = NestedTensor(imgs, masks)

#     t, m = nt.decompose()
#     print(f"NestedTensor: tensors={t.shape}, mask={m.shape}")

#     # Test collate_fn
#     batch = [
#         (torch.randn(3, 640, 640), torch.zeros(640, 640, dtype=torch.bool),
#          torch.randint(0, 100, (17,)), torch.ones(17, dtype=torch.long),
#          torch.tensor([0.5, 0.5, 0.3, 0.4])),
#         (torch.randn(3, 640, 640), torch.zeros(640, 640, dtype=torch.bool),
#          torch.randint(0, 100, (17,)), torch.ones(17, dtype=torch.long),
#          torch.tensor([0.3, 0.3, 0.2, 0.2])),
#     ]
#     img_data, text_data, bbox = collate_fn(batch)
#     print(f"\nCollate output:")
#     print(f"  img_data:  tensors={img_data.tensors.shape}, mask={img_data.mask.shape}")
#     print(f"  text_data: tensors={text_data.tensors.shape}, mask={text_data.mask.shape}")
#     print(f"  bbox:      {bbox.shape}")

#     print("misc.py test passed")

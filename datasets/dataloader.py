from torch.utils.data import DataLoader
from utils.misc import collate_fn


def build_dataloader(dataset, batch_size, shuffle=True, num_workers=2):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=(shuffle == True),  
        persistent_workers=(num_workers > 0)
    )

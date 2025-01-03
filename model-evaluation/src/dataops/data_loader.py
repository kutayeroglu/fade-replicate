import os

from torch.utils.data import DataLoader

from dataops.utils import generate_image_IDs, ensure_datadir_exists, collate_fn
from dataops.coco_custom import COCODatasetWithIDs
from dataops.transformations import get_transform


def get_data_loader(drive_images_dir, drive_annotations_full_path):
    # Sanity check
    ensure_datadir_exists(drive_images_dir, drive_annotations_full_path)

    # Construct COCO dataset object
    dataset = COCODatasetWithIDs(
        root=drive_images_dir,
        annFile=drive_annotations_full_path,
        transform=get_transform(train=False),
        available_image_IDs=generate_image_IDs(drive_images_dir)
    )

    print(f'\nDataset size is {dataset.__len__()}')

    # Create dataloader object
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    return dataloader

import os

from torch.utils.data import DataLoader

from dataops.utils import generate_image_IDs, ensure_datadir_exists, collate_fn, get_data_subset
from dataops.coco_custom import COCODatasetWithIDs
from dataops.transformations import get_transform



def get_data_loader(drive_images_dir, drive_annotations_full_path, train=True, subset=False, subset_size=5000):
    # Sanity check
    ensure_datadir_exists(drive_images_dir, drive_annotations_full_path)

    # Construct COCO dataset object
    dataset = COCODatasetWithIDs(
        root=drive_images_dir,
        annFile=drive_annotations_full_path,
        transform=get_transform(train=train),
        available_image_IDs=generate_image_IDs(drive_images_dir),
        train=train
    )

    print(f'\nDataset size is {dataset.__len__()}')

    if subset: # TODO: ensure class distribution stays similar
        # Get subset of dataset, workaround for limited computational resources
        dataset = get_data_subset(dataset, subset_size)


    print("Creating dataloader object...")
    # Create dataloader object
    dataloader = DataLoader(
        dataset,
        batch_size=2 if train else 1,
        shuffle=train,
        num_workers=4,
        collate_fn=collate_fn
    )

    return dataloader

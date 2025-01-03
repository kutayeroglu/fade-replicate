import os
from src.dataops.utils import generate_image_IDs, ensure_datadir_exists, collate_fn
from src.dataops.coco_custom import COCODatasetWithIDs
from src.dataops.transformations import get_transform
from torch.utils.data import DataLoader


def main():
    # Data split type (train2017, val2017, test2017)
    dataset_split = "val2017"

    # Paths in Google Drive
    ## Images
    drive_coco_base_dir = '/content/drive/MyDrive/coco'
    drive_images_dir = os.path.join(drive_coco_base_dir, 'images', dataset_split)

    ## Annotations
    ann_filename = f'instances_{dataset_split}.json'
    drive_annotations_full_path = os.path.join(drive_coco_base_dir, 'annotations', ann_filename)

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


if __name__ == '__main__':
    main()
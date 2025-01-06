import os
from model_with_fade.data_ops.custom_COCO import COCODatasetWithIDs
from model_with_fade.data_ops.transformations import get_transform


def construct_train_dataset():
    ## Generate a set of available image IDs
    available_images = set()
    image_dir = '/content/drive/MyDrive/coco/images/train2017/'

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # Extract image ID from filename (e.g., '000000101243.jpg' -> 101243)
            image_id = int(filename.split('.')[0])
            available_images.add(image_id)

    ## Define data path
    # Data split type: (train, test, val)
    dataset_split = "train2017"

    # Paths in Google Drive
    ## Images
    drive_coco_base_dir = '/content/drive/MyDrive/coco'
    drive_images_dir = os.path.join(drive_coco_base_dir, 'images', dataset_split)
    print(f"Image source directory: {drive_images_dir}")
    print("Does image directory exist?", os.path.exists(drive_images_dir))

    ## Annotations
    ann_fname = f'instances_{dataset_split}.json'
    drive_annotations_file = os.path.join(drive_coco_base_dir, 'annotations', ann_fname)

    print(f"Annotation file path: {drive_annotations_file}")
    print("Does annotation file exist?", os.path.exists(drive_annotations_file))

    ## Construct train data
    train_dataset = COCODatasetWithIDs(
        root=drive_images_dir,
        annFile=drive_annotations_file,
        transform=get_transform(train=True)
    )

    print(train_dataset.__len__())



# def _tmp():
#     # Visualize the sample
#     visualize_image_with_boxes(img, target)

#     """## Create a DataLoader and Test-Fetch Batches"""


#     from torch.utils.data import DataLoader

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=2,  # Adjust based on your system's memory
#         shuffle=True,
#         num_workers=2,  # Adjust based on your system's capabilities
#         collate_fn=collate_fn
#     )

#     # Fetch multiple batches to ensure consistency
#     for _ in range(5):
#         images, targets = next(iter(train_loader))
#         print(f"Number of images: {len(images)}")

#     # Fetch a batch
#     images, targets = next(iter(train_loader))

#     # Verify the batch
#     print(f"Number of images: {len(images)}")
#     print(f"Image 0 shape: {images[0].shape}")
#     print(f"Image 1 shape: {images[1].shape}")
#     print(f"Target 0 keys: {targets[0].keys()}")
#     print(f"Target 1 keys: {targets[1].keys()}")
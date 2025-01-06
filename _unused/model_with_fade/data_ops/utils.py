def collate_fn(batch):
    return tuple(zip(*batch))



def inspect_sample(train_dataset):
    """## Inspect a sample"""

    # Load a sample from the dataset
    img, target = train_dataset[0]

    # Verify the image
    print(f"Image shape: {img.shape}")  # Should be [C, H, W]
    print(f"Image dtype: {img.dtype}")  # Should be torch.float32

    # Verify the target
    print(f"Target keys: {target.keys()}")

    # Verify the bounding boxes
    print(f"Boxes: {target['boxes']}")
    print(f"Boxes shape: {target['boxes'].shape}")

    # Verify the labels
    print(f"Labels: {target['labels']}")
    print(f"Labels shape: {target['labels'].shape}")

    # Verify other target fields
    print(f"Image ID: {target['image_id']}")
    print(f"Area: {target['area']}")
    print(f"Iscrowd: {target['iscrowd']}")
import torch
from torchvision import datasets

class COCODatasetWithIDs(datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform=None)
        self.ids = [img_id for img_id in self.ids if img_id in available_images] # Filter out missing images
        self.transform = transform
        # Create a mapping from COCO category IDs to a contiguous range
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_contiguous_id = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}  # Start from 1
        # Add background class (0)
        self.contiguous_id_to_cat_id = {0: 0}
        self.contiguous_id_to_cat_id.update({v: k for k, v in self.cat_id_to_contiguous_id.items()})

    def __getitem__(self, index):
        img, annotations = super().__getitem__(index)
        img_id = self.ids[index]

        # Convert annotations to target format
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for annotation in annotations:
            # COCO bbox format: [xmin, ymin, width, height]
            xmin = annotation['bbox'][0]
            ymin = annotation['bbox'][1]
            width = annotation['bbox'][2]
            height = annotation['bbox'][3]
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            # Map category_id to a contiguous id
            labels.append(self.cat_id_to_contiguous_id[annotation['category_id']])
            areas.append(annotation['area'])
            iscrowd.append(annotation.get('iscrowd', 0))

        if len(boxes) == 0:
            # If there are no annotations, create empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([img_id])

        # Expected target format for Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
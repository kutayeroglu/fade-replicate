import torch
from torchvision import datasets


class COCODatasetWithIDs(datasets.CocoDetection):
    def __init__(self, root, annFile, available_image_IDs, transform=None):
        super().__init__(root, annFile, transform=None)
        self.ids = [img_id for img_id in self.ids if img_id in available_image_IDs] # Filter out missing images
        self.transform = transform

        # Create a mapping from COCO category IDs to a contiguous range
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_contiguous_id = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}  # Start from 1

        # Add background class (0), this ensures dataset itself can refer to "background = 0"
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

            # Custom bbox format: [xmin, ymin, xmax, ymax]
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
    

def format_predictions_for_coco(predictions, image_ids):
    """
    Converts raw model predictions into the COCO evaluation format.

    This function processes the predictions from an object detection model
    and prepares them for evaluation using the COCO API. The raw predictions
    are transformed into a list of dictionaries, each representing a single
    detection in the required COCO format.

    Args:
        predictions (list of dict): Model outputs for each image. Each dictionary
            contains:
                - "boxes" (Tensor): Bounding boxes in [xmin, ymin, xmax, ymax] format.
                - "scores" (Tensor): Confidence scores for each bounding box.
                - "labels" (Tensor): Predicted class labels for each bounding box.
        image_ids (list of int): List of image IDs corresponding to the predictions.

    Returns:
        list of dict: A list of dictionaries, where each dictionary represents
        a single detection in the COCO format with the following keys:
            - "image_id" (int): The ID of the image.
            - "category_id" (int): The predicted category ID.
            - "bbox" (list of float): Bounding box in [x, y, width, height] format.
            - "score" (float): Confidence score for the detection.

    Raises:
        ValueError: If the number of predictions does not match the number of image IDs.
    """
    # Validate input lengths
    if len(predictions) != len(image_ids):
        raise ValueError("Number of predictions does not match number of image IDs.")
    
    coco_results = []  # List to store formatted results

    for prediction, image_id in zip(predictions, image_ids):
        # Extract bounding boxes, scores, and labels from the prediction
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        # Convert tensors to NumPy arrays for easier processing
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        # Convert bounding boxes from [xmin, ymin, xmax, ymax] to [x, y, width, height]
        boxes[:, 2:] -= boxes[:, :2]

        # Iterate over each detection in the prediction
        for box, score, label in zip(boxes, scores, labels):
            coco_result = {
                "image_id": int(image_id),    # Image ID as integer
                "category_id": int(label),   # Predicted category ID
                "bbox": box.tolist(),        # Bounding box in COCO format
                "score": float(score)        # Confidence score
            }
            # Append the formatted result to the list
            coco_results.append(coco_result)

    return coco_results

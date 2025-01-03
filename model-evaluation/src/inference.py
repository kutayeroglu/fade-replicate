from tqdm import tqdm

import torch
from pycocotools.coco import COCO

from src.dataops.coco_custom import format_predictions_for_coco


def predict(model, data_loader, annotations_full_path, device):
    # Initialize ground truth
    coco_gt = COCO(annotations_full_path)

    # Run predictions
    with torch.no_grad():
        results = []
        for images, targets, image_ids in tqdm(data_loader):
            # Move images to device
            images = [image.to(device) for image in images]

            # Run inference
            outputs = model(images)

            # Move outputs to CPU
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # Prepare results for COCO evaluation
            results.extend(format_predictions_for_coco(outputs, image_ids))

    return results, coco_gt

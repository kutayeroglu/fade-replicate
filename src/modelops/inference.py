from tqdm import tqdm

import torch
from pycocotools.coco import COCO

import importlib.util
# Path to the `coco_custom.py` file
coco_custom_path = "/content/drive/MyDrive/Colab Notebooks/cmpe593/term-project/src/dataops/coco_custom.py"
# Load the module dynamically
spec = importlib.util.spec_from_file_location("coco_custom", coco_custom_path)
coco_custom_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coco_custom_module)

# Import the specific function
format_predictions_for_coco = coco_custom_module.format_predictions_for_coco


def predict(model, data_loader, device):
    with torch.no_grad():
        results = []
        for images, _, image_ids in tqdm(data_loader):
            # Move images to device
            images = [image.to(device) for image in images]

            # Run inference
            outputs = model(images)

            # Move outputs to CPU
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # Prepare results for COCO evaluation
            results.extend(format_predictions_for_coco(outputs, image_ids))

    return results

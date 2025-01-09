import os
import shutil

##############################################################################
base_path = "/content/drive/MyDrive"

## Dataset paths
coco_base_dir = f'{base_path}/coco'
val_images_dir = os.path.join(coco_base_dir, 'images', 'val2017')
val_annotations_path = os.path.join(coco_base_dir, 'annotations', 'instances_val2017.json')

### Local dataset paths
local_coco_base_dir = '/content/coco'
local_val_images_dir = os.path.join(local_coco_base_dir, 'images', 'val2017')
local_val_annotations_dir = os.path.join(local_coco_base_dir, 'annotations')
##############################################################################


def copy_data_to_local_storage():
  '''
  Designed to run in Google Colab. Copies the COCO validation dataset to the local storage.
  Not needed since we store the dataset in drive and mount it.
  '''
  # Create local directories
  os.makedirs(local_val_images_dir, exist_ok=True)
  os.makedirs(local_val_annotations_dir, exist_ok=True)

  # Copy images
  # !rsync -ah --progress "{val_images_dir}/" "{local_val_images_dir}/"

  # Copy annotations
  shutil.copy2(val_annotations_path, local_val_annotations_dir)

  # Paths to the validation images and annotations in local storage
  val_images_dir = local_val_images_dir
  val_annotations_full_path = os.path.join(local_val_annotations_dir, 'instances_val2017.json')

  return val_images_dir, val_annotations_full_path


# Ensure forward pass stability of the model
def test_model(input_type="real"):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  # Change to random noise if prompted
  if input_type == "noise":
    sample_input = torch.randn(3, 800, 800)
    sample_input = sample_input.to(device)
  elif input_type == "real": # TODO: Implement
    pass

  with torch.no_grad():
      outputs = model([sample_input])

  print(f'Model output for input type: {input_type}')
  print(outputs)

test_model(input_type="noise")


def inspect_sample(dataset, index):
  # Load a sample from the dataset
  img, target = dataset[index]

  # Verify the image
  print(f"Image shape: {img.shape}")  # Should be [C, H, W]
  print(f"Image dtype: {img.dtype}")  # Should be torch.float32

  # Verify the target
  print(f"\nTarget keys: {target.keys()}")

  # Verify the bounding boxes
  print(f"\nBoxes: {target['boxes']}")
  print(f"Boxes shape: {target['boxes'].shape}")

  # Verify the labels
  print(f"\nLabels: {target['labels']}")
  print(f"Labels shape: {target['labels'].shape}")

  # Verify other target fields
  print(f"\nImage ID: {target['image_id']}")
  print(f"Area: {target['area']}")
  print(f"Iscrowd: {target['iscrowd']}")

inspect_sample(train_dataset, 0)

# TODO: debug: for checking validity of bounding box bounds
# def validate_bboxes_subset(dataset, sample_size=2000):
#     """
#     Checks a random subset of the dataset to ensure bounding boxes are valid.
#     sample_size: how many samples to validate
#     """
#     num_samples = len(dataset)
#     sampled_indices = random.sample(range(num_samples), min(sample_size, num_samples))

#     invalid_samples = []
#     for idx in tqdm(sampled_indices, desc="Validating bounding boxes (subset)", unit="sample"):
#         img, target = dataset[idx]

#         # get image dimensions
#         _, img_height, img_width = img.shape
#         boxes = target['boxes'].cpu()

#         if boxes.size(0) == 0:
#             continue

#         widths = boxes[:, 2] - boxes[:, 0]
#         heights = boxes[:, 3] - boxes[:, 1]

#         invalid_mask = (
#             (widths <= 0) |
#             (heights <= 0) |
#             (boxes[:, 0] < 0) |
#             (boxes[:, 1] < 0) |
#             (boxes[:, 2] > img_width) |
#             (boxes[:, 3] > img_height)
#         )

#         if invalid_mask.any():
#             invalid_samples.append(idx)

#     if invalid_samples:
#         print("Found invalid bounding boxes in these samples:", invalid_samples)
#     else:
#         print(f"\nNo invalid bounding boxes found in this random subset of {len(sampled_indices)} samples.")

#     return invalid_samples


# invalid_samples = validate_bboxes_subset(train_dataset, sample_size=2000)
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
  !rsync -ah --progress "{val_images_dir}/" "{local_val_images_dir}/"

  # Copy annotations
  shutil.copy2(val_annotations_path, local_val_annotations_dir)

  # Paths to the validation images and annotations in local storage
  val_images_dir = local_val_images_dir
  val_annotations_full_path = os.path.join(local_val_annotations_dir, 'instances_val2017.json')

  return val_images_dir, val_annotations_full_path
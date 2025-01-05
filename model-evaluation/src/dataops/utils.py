import os
from tqdm import tqdm
from google.colab import drive


def _ensure_drive_mounted(mount_path='/content/drive'):
  """
  Ensures that Google Drive is mounted at the specified path.

  Args:
    mount_path: The path where Google Drive should be mounted.

  Raises:
    OSError: If the drive is not mounted or is inaccessible.
  """

  # Check if the mount path exists and is a directory.
  if not os.path.exists(mount_path) or not os.path.isdir(mount_path):
    # Mount the drive using google.colab.drive.
    drive.mount(mount_path)
  else:
    print(f"Drive already mounted at {mount_path}.")

  # Check if the mount was successful.
  if not os.path.exists(mount_path) or not os.path.isdir(mount_path):
    raise OSError(f"Could not mount Google Drive at {mount_path}. "
                  "Please check your connection and try again.")


def ensure_datadir_exists(drive_images_dir, drive_annotations_path):
  _ensure_drive_mounted()

  print(f"\nImage file directory: {drive_images_dir}")
  print("Does image directory exist?", os.path.exists(drive_images_dir))

  print(f"\nAnnotation file path: {drive_annotations_path}")
  print("Does annotation file exist?", os.path.exists(drive_annotations_path))


def generate_image_IDs(image_dir):
  # Ensure drive is mounted to prevent I/O errors
  _ensure_drive_mounted()

  # Define path for images
  available_images = set()

  print('\nGenerating image IDs...')
  # Get IDs of available images
  for filename in tqdm(os.listdir(image_dir), desc="Processing images", unit="file"):
      if filename.endswith('.jpg'):
          # Extract image ID from filename (e.g., '000000101243.jpg' -> 101243)
          image_id = int(filename.split('.')[0])
          available_images.add(image_id)

  print(f'Total of {len(available_images)} image IDs were generated.')
  return available_images

def collate_fn(batch):
    # Ensure data loader skips None values
    return tuple(zip(*[b for b in batch if b is not None]))


def get_data_subset(dataset, subset_size):
    print(f'\nCreating subset of dataset with size {subset_size}...')
    import random
    from torch.utils.data import Subset

    dataset_size = len(dataset)
    subset_indices = random.sample(range(dataset_size), min(subset_size, dataset_size))

    return Subset(dataset, subset_indices)

import os
import time
import datetime
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    num_epochs,
    drive_checkpoint_dir,
    lr_scheduler=None,
    val_loader=None,
):
  epoch_losses = []
  batch_losses = [] # Track batch-level losses
  error_batches = []  # To store details of failing batches for debug

  # File paths for saving loss data
  epoch_loss_file = os.path.join(drive_checkpoint_dir, "epoch_losses.txt")
  batch_loss_file = os.path.join(drive_checkpoint_dir, "batch_losses.txt")

  # Initialize StepLR scheduler for dynamic adjustment after warm-up
  post_warmup_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

  for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    batch_count = 0

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print('-' * 20)

    # Training loop
    for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
      batch_count += 1

      # Move images and targets to the device
      images = [img.to(device) for img in images]
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

      try:
        # Forward pass
        loss_dict = model(images, targets)
      except AssertionError as e:
        print(f"\n[ERROR] Bounding box error in batch {batch_count}!")
        batch_error_info = {"epoch": epoch+1, "batch_idx": batch_count, "boxes": []}

        for i, tgt in enumerate(targets):
          batch_error_info["boxes"].append(tgt['boxes'].cpu().tolist())

        error_batches.append(batch_error_info)
        continue

      # Compute total loss
      losses = sum(loss for loss in loss_dict.values())
      loss_value = losses.item()
      epoch_loss += loss_value
      batch_losses.append(loss_value)

      # Backward pass and optimization
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      # Display intermediate losses
      if batch_count % 250 == 0:
        print(f"Batch {batch_count}, Loss: {loss_value:.4f}")


    # Transition from warm-up to StepLR after the warm-up phase
    if epoch == 3:
      print(f"Transitioning from warm-up to StepLR at learning rate: {optimizer.param_groups[0]['lr']:.6f}")
      lr_scheduler = post_warmup_scheduler

    # Step the learning rate scheduler
    if lr_scheduler:
      lr_scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    epoch_duration = time.time() - start_time
    average_epoch_loss = epoch_loss / len(train_loader)
    epoch_losses.append(average_epoch_loss)

    # Save epoch summary
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}")
    print(f"Average Loss: {average_epoch_loss:.4f}")
    print(f"Current Learning Rate: {current_lr:.6f}")

    # Save checkpoint
    drive_checkpoint_path = os.path.join(drive_checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), drive_checkpoint_path)
    print(f"Checkpoint saved to Google Drive at {drive_checkpoint_path}")

  # =========================
  # Training loop completes
  # =========================
  print("Training loop finished.")

  # Save the model state dictionary
  drive_final_checkpoint_path = os.path.join(drive_checkpoint_dir, 'model_final.pth')
  torch.save(model.state_dict(), drive_final_checkpoint_path)
  print(f"Final model saved to Google Drive at {drive_final_checkpoint_path}")



  plot_filename_drive = os.path.join(plot_dir, 'training_loss_plot.png')
  plt.figure()
  plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
  plt.xlabel('Epoch')
  plt.ylabel('Average Loss')
  plt.title('Training Loss Over Epochs')
  plt.grid(True)
  plt.savefig(plot_filename_drive)
  print(f"Training loss plot saved to Google Drive at {plot_filename_drive}")
  plt.show()

  if error_batches:
      print("\nSome batches triggered bounding box assertions. Their info:")
      for info in error_batches:
          print(f"Epoch {info['epoch']}, Batch {info['batch_idx']}, Boxes: {info['boxes']}")
  else:
      print("No bounding box errors encountered!")
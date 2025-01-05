import time
import shutil
import datetime
from tqdm import tqdm

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
  error_batches = []  # To store details of failing batches for debug
  ap_history = []     # To store AP metrics after each epoch

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
              print("Inspecting bounding boxes for each sample in this batch:")
              batch_error_info = {"epoch": epoch+1, "batch_idx": batch_count, "boxes": []}

              for i, tgt in enumerate(targets):
                  print(f"Sample index in batch: {i}, boxes:\n{tgt['boxes']}")
                  # Collect boxes info for later debug
                  batch_error_info["boxes"].append(tgt['boxes'].cpu().tolist())

              # Store the error info
              error_batches.append(batch_error_info)
              # Skip this batch rather than crashing
              continue

          # Compute total loss
          losses = sum(loss for loss in loss_dict.values())
          loss_value = losses.item()
          epoch_loss += loss_value

          # Backward pass and optimization
          optimizer.zero_grad()
          losses.backward()
          optimizer.step()

          # Display intermediate losses
          if batch_count % 100 == 0:
              print(f"Batch {batch_count}, Loss: {loss_value:.4f}")

      current_lr = optimizer.param_groups[0]['lr']

      # Step the learning rate scheduler
      if lr_scheduler:
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

      epoch_duration = time.time() - start_time
      epoch_duration_str = str(datetime.timedelta(seconds=int(epoch_duration)))

      average_epoch_loss = epoch_loss / len(train_loader)
      epoch_losses.append(average_epoch_loss)

      # Display epoch summary
      print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration_str}")
      print(f"Average Loss: {average_epoch_loss:.4f}")
      print(f"Current Learning Rate: {current_lr:.6f}")

      # Save the checkpoint directly to Google Drive
      drive_checkpoint_path = os.path.join(drive_checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
      torch.save(model.state_dict(), drive_checkpoint_path)
      print(f"Checkpoint saved to Google Drive at {drive_checkpoint_path}")

  # =========================
  # Training loop completes
  # =========================
  print("Training loop finished.")

  # Save the model state dictionary directly to the target path
  drive_final_checkpoint_path = os.path.join(drive_checkpoint_dir, 'model_final.pth')
  torch.save(model.state_dict(), drive_final_checkpoint_path)
  print(f"Final model saved to Google Drive at {drive_final_checkpoint_path}")

  # Save then display the training loss plot
  import matplotlib.pyplot as plt

  plot_filename_drive = os.path.join(drive_checkpoint_dir, 'training_loss_plot.png')
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
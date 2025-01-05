import shutil
import datetime

def train_model(
    model,
    train_loader,
    optimizer,
    lr_scheduler,
    device,
    num_epochs,
    drive_checkpoint_dir,
    val_loader=None,
):
  epoch_losses = []
  error_batches = []  # To store details of failing batches for debug
  ap_history = []     # To store AP metrics after each epoch

  for epoch in range(num_epochs):
      model.train()
      epoch_loss = 0.0
      start_time = time.time()

      print(f"Epoch {epoch + 1}/{num_epochs}")
      print('-' * 20)

      batch_count = 0

      # Training loop
      for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
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

          # Optional: Print intermediate losses
          if batch_count % 100 == 0:
              print(f"Batch {batch_count}, Loss: {loss_value:.4f}")

      # Step the learning rate scheduler
      lr_scheduler.step()
      current_lr = optimizer.param_groups[0]['lr']

      epoch_duration = time.time() - start_time
      epoch_duration_str = str(datetime.timedelta(seconds=int(epoch_duration)))

      average_epoch_loss = epoch_loss / len(train_loader)
      epoch_losses.append(average_epoch_loss)

      # Print epoch summary
      print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration_str}")
      print(f"Average Loss: {average_epoch_loss:.4f}")
      print(f"Current Learning Rate: {current_lr:.6f}")

      # Save checkpoint locally
      checkpoint_path = f'model_epoch_{epoch + 1}.pth'
      torch.save(model.state_dict(), checkpoint_path)

      # Copy checkpoint to Drive
      drive_checkpoint_path = os.path.join(drive_checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
      shutil.copyfile(checkpoint_path, drive_checkpoint_path)
      print(f"Checkpoint saved to Google Drive at {drive_checkpoint_path}")

      # # Evaluate the model after each epoch (optional but recommended)
      # print("Evaluating model on validation set...")
      # ap_results = evaluate_model(model, val_loader, device)
      # print("AP results:", ap_results)

      # # Store these AP results
      # ap_history.append({
      #     "epoch": epoch+1,
      #     "AP": ap_results.get("AP", 0.0),
      #     "AP50": ap_results.get("AP50", 0.0),
      #     "AP75": ap_results.get("AP75", 0.0),
      #     "APs": ap_results.get("APs", 0.0),
      #     "APm": ap_results.get("APm", 0.0),
      #     "APl": ap_results.get("APl", 0.0)
      # })

      # Optionally remove local checkpoint to save space
      # os.remove(checkpoint_path)

  # =========================
  # Training loop completes
  # =========================
  print("Training loop finished.")

  # Save final model
  final_checkpoint_path = 'model_final.pth'
  torch.save(model.state_dict(), final_checkpoint_path)
  drive_final_checkpoint_path = os.path.join(drive_checkpoint_dir, 'model_final.pth')
  shutil.copyfile(final_checkpoint_path, drive_final_checkpoint_path)
  print(f"Final model saved to Google Drive at {drive_final_checkpoint_path}")

  # Save the training loss plot
  import matplotlib.pyplot as plt

  plt.figure()
  plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
  plt.xlabel('Epoch')
  plt.ylabel('Average Loss')
  plt.title('Training Loss Over Epochs')
  plt.grid(True)
  plt.show()

  plot_filename_local = 'training_loss_plot.png'
  plt.savefig(plot_filename_local)
  plot_filename_drive = os.path.join(drive_checkpoint_dir, 'training_loss_plot.png')
  shutil.copyfile(plot_filename_local, plot_filename_drive)
  print(f"Training loss plot saved to Google Drive at {plot_filename_drive}")
  # os.remove(plot_filename_local)  # if you want to clear local

  # TODO: Uncomment when ap metric eval is implemented
  # # Save AP history to a CSV
  # csv_path_local = "ap_metrics.csv"
  # fieldnames = ["epoch", "AP", "AP50", "AP75", "APs", "APm", "APl"]
  # with open(csv_path_local, mode='w', newline='') as f:
  #     writer = csv.DictWriter(f, fieldnames=fieldnames)
  #     writer.writeheader()
  #     for row in ap_history:
  #         writer.writerow(row)

  # # Copy CSV to Drive
  # csv_path_drive = os.path.join(drive_checkpoint_dir, "ap_metrics.csv")
  # shutil.copyfile(csv_path_local, csv_path_drive)
  # print(f"AP metrics saved to Google Drive at {csv_path_drive}")

  # If you want to see if any error batches occurred
  if error_batches:
      print("\nSome batches triggered bounding box assertions. Their info:")
      for info in error_batches:
          print(f"Epoch {info['epoch']}, Batch {info['batch_idx']}, Boxes: {info['boxes']}")
  else:
      print("No bounding box errors encountered!")
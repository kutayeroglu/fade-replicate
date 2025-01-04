# TODO: This is a fallback code, delete if new training loop works.

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0.0  # To accumulate loss over the epoch
    start_time = time.time()  # To measure epoch duration

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print('-' * 20)

    # Initialize a counter for batches
    batch_count = 0

    for images, targets in train_loader:
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
          for i, tgt in enumerate(targets):
              print(f"Sample index in batch: {i}, boxes:\n{tgt['boxes']}")
          continue


        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss += loss_value

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Print loss every N batches
        if batch_count % 100 == 0:
            print(f"Batch {batch_count}, Loss: {loss_value:.4f}")

    # Update the learning rate
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Calculate epoch duration
    epoch_duration = time.time() - start_time
    epoch_duration_str = str(datetime.timedelta(seconds=int(epoch_duration)))

    # Calculate average loss for the epoch
    average_epoch_loss = epoch_loss / len(train_loader)
    epoch_losses.append(average_epoch_loss)

    # Print epoch summary
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration_str}")
    print(f"Average Loss: {average_epoch_loss:.4f}")
    print(f"Current Learning Rate: {current_lr:.6f}")

    # Save the model checkpoint locally
    checkpoint_path = f'model_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), checkpoint_path)

    # Save the model checkpoint to Google Drive
    drive_checkpoint_path = os.path.join(drive_checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
    shutil.copyfile(checkpoint_path, drive_checkpoint_path)
    print(f"Checkpoint saved to Google Drive at {drive_checkpoint_path}")

    # Optional: Remove the local checkpoint file to save space
    # os.remove(checkpoint_path)


# TODO: Implement rest of this to save AP metrics during training.
# # Pseudocode for evaluating your model; you need to implement 'evaluate_model'
# # to return a dict with keys: "AP", "AP50", "AP75", "APs", "APm", "APl"
# def evaluate_model(model, val_loader, device):
#     """
#     Evaluate the model on a validation set and return COCO-style metrics.
#     This is a placeholder function. You'll need to implement it or
#     use an existing COCO evaluation pipeline.
#     """
#     model.eval()
#     # Perform inference on the validation set, gather predictions,
#     # use pycocotools or a custom function to compute APs.
#     ap_results = {
#         "AP": 0.0,
#         "AP50": 0.0,
#         "AP75": 0.0,
#         "APs": 0.0,
#         "APm": 0.0,
#         "APl": 0.0
#     }
#     # ...
#     return ap_results
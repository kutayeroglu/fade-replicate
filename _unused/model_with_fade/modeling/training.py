import datetime
import shutil


def initialize_model():
    """# Modeling"""

    num_classes = len(train_dataset.cat_ids) + 1  # +1 for background

    """## Set params"""

    # Get the model parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    # Define the optimizer
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Define the learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    """## Move to GPU (if any)"""

    # Check if GPU is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # Move model to the appropriate device
    model.to(device)

    """## Sanity Check"""

    model.train()
    images, targets = next(iter(train_loader))
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()

    print(f"Initial test run loss: {losses.item():.4f}")


def train_model(num_epochs):
    """## Training loop"""

    # Set the number of epochs
    num_epochs = 10  # Adjust as needed

    # Lists to store loss values for analysis
    epoch_losses = []

    # Define the directory in Google Drive where you want to save the model checkpoints
    drive_checkpoint_dir = '/content/drive/MyDrive/model_checkpoints'

    # Create the directory if it doesn't exist
    if not os.path.exists(drive_checkpoint_dir):
        os.makedirs(drive_checkpoint_dir)

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

            # Forward pass
            loss_dict = model(images, targets)

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

    # Save the final model locally
    final_checkpoint_path = 'model_final.pth'
    torch.save(model.state_dict(), final_checkpoint_path)

    # Save the final model to Google Drive
    drive_final_checkpoint_path = os.path.join(drive_checkpoint_dir, 'model_final.pth')
    shutil.copyfile(final_checkpoint_path, drive_final_checkpoint_path)
    print(f"Final model saved to Google Drive at {drive_final_checkpoint_path}")

    # Optional: Remove the local final model file to save space
    # os.remove(final_checkpoint_path)


def warmup_schedule(epoch):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # Start from a small value and scale up
    return 1.0  # dummy

from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch.optim as optim

# Mock model with parameters
class DummyModel:
    def parameters(self):
        return [torch.randn(1, requires_grad=True)]

# Initialize dummy model
model = DummyModel()

# Initialize model parameters that require gradients
params = [p for p in model.parameters() if p.requires_grad]

# Define the optimizer
optimizer = optim.SGD(
    params,
    lr=0.01,  # Base learning rate
    momentum=0.9,
    weight_decay=0.0005
)

# Define warmup schedule
def warmup_schedule(epoch):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # Gradual increase
    return 1.0  # Maintain the base learning rate after warm-up

# Initialize the warm-up scheduler
warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)

# Initialize the post-warmup scheduler
post_warmup_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Validate the learning rate schedule
def validate_learning_rate_schedule(optimizer, warmup_scheduler, post_warmup_scheduler, num_epochs=10):
    lr_scheduler = warmup_scheduler  # Start with the warm-up scheduler
    
    for epoch in range(num_epochs):
      # Log the learning rate at the start of the epoch
      current_lr = optimizer.param_groups[0]['lr']
      
      # Log initial learning rate (before scheduler is applied)
      print(f"Epoch {epoch}: Initial Optimizer LR: {optimizer.param_groups[0]['lr']:.6f}")

      # Apply the first scheduler step
      lr_scheduler.step()

      # Log learning rate after scheduler step
      print(f"Learning Rate after first step: {optimizer.param_groups[0]['lr']:.6f}")
      
      # Transition from warm-up to StepLR after epoch 3
      if epoch == 2:  # After epoch 3 (0-indexed), switch to StepLR
          print("\nTransitioning to StepLR...")
          lr_scheduler = post_warmup_scheduler

# Run validation
validate_learning_rate_schedule(optimizer, warmup_scheduler, post_warmup_scheduler, num_epochs=10)

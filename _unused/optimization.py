def warmup_schedule(epoch):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # Start from a small value and scale up
    return 1.0  # dummy

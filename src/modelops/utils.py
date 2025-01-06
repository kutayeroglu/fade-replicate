def sanity_check(model, data_loader, device, optimizer):
    model.train()
    images, targets = next(iter(data_loader))
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()

    print(f"Initial test run loss: {losses.item():.4f}")
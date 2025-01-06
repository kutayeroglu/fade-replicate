"""## Visualize the Image and Bounding Boxes"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image_with_boxes(img, target):
    # Convert the image tensor to a NumPy array and transpose to [H, W, C]
    img_np = img.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_np)

    boxes = target['boxes']
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    plt.show()
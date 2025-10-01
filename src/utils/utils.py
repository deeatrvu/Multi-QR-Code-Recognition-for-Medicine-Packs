import os
import torch
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save checkpoint to disk"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth'))

def visualize_detections(image, boxes, scores=None, save_path=None):
    """
    Visualize detected QR codes on an image
    
    Args:
        image: The image as a numpy array (H, W, C)
        boxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
        scores: Optional list of confidence scores
        save_path: Optional path to save the visualization
    """
    # Create figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Add bounding boxes
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add score if available
        if scores is not None:
            plt.text(
                x_min, y_min - 5,
                f"Score: {scores[i]:.2f}",
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5)
            )
    
    # Remove axis ticks
    plt.axis('off')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()
    else:
        plt.show()

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes
    
    Args:
        box1: First box in format [x_min, y_min, x_max, y_max]
        box2: Second box in format [x_min, y_min, x_max, y_max]
        
    Returns:
        IoU score
    """
    # Get coordinates of intersection
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    # Compute area of intersection
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    
    # Compute area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou
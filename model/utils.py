from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import torch


def plot_result(img, seg, cls_dict):
    # Labels and classes names correspondance
    id2label = dict(cls_dict)
    label2id = {v: k for k, v in cls_dict.items()}

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    img = np.array(img) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    legend_elements = [Patch(facecolor=(clr[::-1] / 255), label=cls_name) for [cls_name, clr] in zip(label2id.keys(), palette)]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.show()

def save_checkpoint(cfg, best):
    save_path = f"{cfg.checkpoint_save_path}_{cfg.model_type.split('/')[-1]}_{best['epoch']}_{'%.4f' % best['loss']}.pt"
    torch.save({
        'epoch': best['epoch'],
        'model_state_dict': best['model_state_dict'],
        'optimizer_state_dict': best['optimizer_state_dict'],
        'loss': best['loss'],
        'val_iou_per_class': best['val_iou_per_class'],  # Save the validation IoU per class in the checkpoint
    }, save_path)


def load_checkpoint(cfg, model, optimizer):
    checkpoint = torch.load(cfg.checkpoint_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_iou_per_class = checkpoint['val_iou_per_class']

    print(f"Loaded checkpoint: {cfg.checkpoint_load_path}")
    print(f"Start epoch: {start_epoch}")
    print(f"Loss: {loss}")

    return model, optimizer, start_epoch, loss, val_iou_per_class


def calculate_iou(pred, target, class_id):
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    iou = 0.0 if union == 0 else intersection / union

    return iou  # Convert to a Python float


def ade_palette():
    """Custom palette that maps each class to RGB values."""
    return [
        [0, 0, 0],            # background (BLAK)
        [255, 0, 0],          # Damaged_buildings (BLUE)
        [0, 255, 0],          #
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 128],
        [128, 128, 0],
        [0, 128, 128],
        [0, 0, 128],
    ]


def plot_loss(mode, pixelwise_accuracies, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pixelwise_accuracies) + 1), pixelwise_accuracies, label=f'{mode} Pixel-wise Accuracy')
    plt.plot(range(1, len(losses) + 1), losses, label=f'{mode} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'{mode} Pixel-wise Accuracy and Loss')
    plt.legend()
    plt.show()


class IoUTable:

    def __init__(self, cfg, cls_dict):
        self.cfg = cfg
        self.cumulative_iou_table_data = []
        self.id2label = dict(cls_dict)
        self.label2id = {v: k for k, v in cls_dict.items()}
        self.cumulative_iou_values_per_class = {class_name: 0.0 for class_name in self.id2label.values()}
        self.val_iou_per_class_epoch = {class_name: [] for class_name in self.id2label.values()}

    def get_val_iou_per_class_epoch(self, predicted, labels):

        for class_id, class_name in enumerate(self.id2label.values()):
            class_iou = calculate_iou(predicted.cpu(), labels.cpu(), class_id)
            self.val_iou_per_class_epoch[class_name].append(class_iou)

    def update_data(self):
        print("Validation IoU per class:")

        self.iou_table_data = []

        for class_name in self.id2label.values():  # Update variable name here
            # Calculate the average IoU for the current epoch
            average_iou = np.mean(self.val_iou_per_class_epoch[class_name])

            # Add the current epoch's IoU to the cumulative IoU for this class
            self.cumulative_iou_values_per_class[class_name] += average_iou

            self.iou_table_data.append([class_name, average_iou])

        print(tabulate(self.iou_table_data, headers=["Class", "IoU per class"], tablefmt="grid"))

        self.val_iou_per_class_epoch = {class_name: [] for class_name in self.id2label.values()}

    def print_cumulative_iou(self):
        print("\nCumulative IoU per class:")

        for class_name, cumulative_iou in self.cumulative_iou_values_per_class.items():
            # Divide the cumulative IoU by the divisor
            cumulative_iou /= self.cfg.num_epochs
            self.cumulative_iou_table_data.append([class_name, cumulative_iou])

        print(tabulate(self.cumulative_iou_table_data, headers=["Class", "Mean IoU by class"], tablefmt="grid"))

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


# def drawMask(src, mask, color):
#     cv2.imwrite('source.png', src)
#     cv2.imwrite('mask.png', mask)
#
#     src = cv2.imread("source.png", -1)
#     mask = cv2.imread("mask.png", -1)
#
#     # print(src.shape, mask.shape)
#
#     # convert mask to gray and then threshold it to convert it to binary
#     gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     ret, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
#
#     # find contours of two major blobs present in the mask
#     contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
#     # draw the found contours on to source image
#     # color = (color[0],color[1],color[2])
#     # print('Draw',color)
#     for contour in contours:
#         # cv2.drawContours(src, contour, -1, (255,0,0), thickness=cv2.FILLED)
#         cv2.fillPoly(src, pts =[contour], color=(int(color[0]),int(color[1]),int(color[2])))
#
#     # # split source to B,G,R channels
#     # b,g,r = cv2.split(src)
#
#     # # add a constant to R channel to highlight the selected area in reed
#     # r = cv2.add(b, 30, dst = b, mask = binary, dtype = cv2.CV_8U)
#
#     # # merge the channels back together
#     # cv2.merge((b,g,r), src)
#     return src


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

# def visualizeMasks(seg, temp_image, shouldShowIntermediateClasses = False, fromDataset=False,
#                    model=None, palette=ade_palette(), map=None):
#     map = np.array(map).astype(np.uint8)
#     palette = np.array(palette)
#     classes_map = np.unique(map).tolist()
#     unique_classes = [model.config.id2label[idx] if idx != 255 else None for idx in classes_map]
#     print("Classes in this image:", unique_classes, classes_map)
#     detected_classes= classes_map
#     map_cpy = np.array(map).copy()
#     updated_pallete  = []
#
#     for cls in detected_classes:
#
#         if(cls==0):
#             continue
#
#         temp_map = map_cpy.copy()
#         # temp_map = np.expand_dims(np.array(temp_map), axis=2)
#         cls_name = model.config.id2label[cls]
#         cls_color = palette[cls]
#         B, G, R = cls_color
#         colored_string = '\033[48;2;'+str(R)+';'+str(G)+';'+str(B)+'m'+str(cls_color)+'!\033[0m'
#         print('Class Name:', cls_name, ', Class ID:', cls, ', Class Color:',colored_string )
#         # color_seg[map == label, :] = palette[cls]
#         # color_seg = color_seg[..., ::-1]
#
#         translate_bit = 0 if fromDataset else 1
#         temp_map[temp_map==cls-translate_bit] = 255
#         temp_map[temp_map!=255] = 0
#
#         if(len(np.unique(temp_map).tolist())==1):
#             temp_map = map_cpy.copy()
#             temp_map[temp_map==cls] = 255
#             temp_map[temp_map!=255] = 0
#
#         # print(temp_map.shape, 'Shape', temp_image.shape)
#         t = drawMask(np.array(temp_image), cv2.merge((temp_map,temp_map,temp_map)),palette[cls] )
#         updated_pallete.append([cls_name, palette[cls]])
#
#         # temp_map = cv2.bitwise_not(temp_map)
#
#         # res = cv2.bitwise_and(temp_image, cv2.merge((temp_map,temp_map,temp_map)))
#         # # print(temp_image[temp_map==255])
#         # res[temp_map==0]=255
#
#         # final = alphaBlendImages(cv2.addWeighted(res, 0.5, temp_image, 0.5, 0.0), temp_image, temp_map)
#
#         r = cv2.addWeighted(t, 0.5, np.array(temp_image), 0.5, 0.0)
#         temp_image = r
#         if(shouldShowIntermediateClasses):
#             # temp_image = temp_image.astype(np.uint8)
#             # plt.figure(figsize=(15, 10))
#             # plt.imshow(temp_image)
#             # plt.show()
#
#             plt_img = temp_image.copy()
#             plt_img = plt_img[..., ::-1]
#             # cv2_imshow_temp(temp_image)
#             img = plt_img.astype(np.uint8)
#             plt.figure(figsize=(15, 10))
#             plt.imshow(img)
#             plt.show()
#
#     return temp_image, updated_pallete, detected_classes


def plot_loss(mode, pixelwise_accuracies, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pixelwise_accuracies) + 1), pixelwise_accuracies,
             label=f'{mode} Pixel-wise Accuracy')
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

    # def reinit_epoch_placeholder(self):
    #     self.val_iou_per_class_epoch = {class_name: [] for class_name in self.id2label.values()}

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

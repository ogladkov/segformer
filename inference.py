import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch import nn

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from model.utils import plot_result


# Read inference config
cfg = OmegaConf.load('./configs/infer_cfg.yml')
cls_dict = OmegaConf.load(cfg.dataset_root + '/classes.json')

# Labels and classes names correspondance
id2label = dict(cls_dict)
label2id = {v: k for k, v in cls_dict.items()}

feature_extractor = SegformerImageProcessor(do_reduce_labels=False)
model = SegformerForSemanticSegmentation.from_pretrained(
    cfg.model_type,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

checkpoint = torch.load(cfg.checkpoint_load_path)
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # initialize available device
model.to(device)  # send model to device

img = Image.fromarray(np.array(Image.open(cfg.input_path)))

encoding = feature_extractor(img, return_tensors="pt")
pixel_values = encoding.pixel_values.to(device)
outputs = model(pixel_values=pixel_values)
pred_masks = outputs.logits
logits = outputs.logits.cpu()
upsampled_logits = nn.functional.interpolate(logits,
                size=img.size[::-1],  # (height, width)
                mode='bilinear',
                align_corners=False)
upsampled_logits = torch.sigmoid(upsampled_logits)
seg = upsampled_logits.argmax(dim=1)[0]

# Plot result
plot_result(img, seg, cls_dict)

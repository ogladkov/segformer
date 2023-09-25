# SegFormer setup

## Installation

Prepare your environment with conda or venv and run installation of requirements with:
```pip install -r requirements.txt```

## Run train
* Setup configuration file for training: ```./config/train_cfg.yml``` (description of parameters is below)
* Run ```python train.py```

```
dataset_root - root folder of your dataset
batch_size - size of you train batch
num_epochs - sets the quantity of epoch for training process
lr - learning rate
model_type - type of SegFormer model
checkpoint_load_path - model's checkpoint path (.pt file) if you want to continue training. Set False for train from the beginning
checkpoint_save_path - path for saving the best model after the training process
```

## Inference
* Setup configuration file for training: ```./config/infer_cdf.yml``` (description of parameters is below)
* Run ```python inference.py```

```
dataset_root - root folder of your dataset
model_type - type of SegFormer model
checkpoint_load_path - model's pretrained weights path (.pt file)
input_path - input image path
```
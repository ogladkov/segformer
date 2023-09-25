from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from augs.base_augs import get_transforms
from model.dataset import SegFormerDataset
from model.utils import calculate_iou, save_checkpoint, load_checkpoint, plot_loss, IoUTable


def main():
    # Read train config file
    cfg_path = './configs/train_cfg.yml'
    cfg = OmegaConf.load(cfg_path)

    # Read classes correspondance
    cls_dict = OmegaConf.load(cfg.dataset_root + '/classes.json')

    # Run training
    train(cfg=cfg, cls_dict=cls_dict)


def train(cfg, cls_dict):
    # Labels and classes names correspondance
    id2label = dict(cls_dict)
    label2id = {v: k for k, v in cls_dict.items()}

    feature_extractor = SegformerImageProcessor(reduce_zero_labels=True)  # get feature extractor of SegFormer

    # Set transforms (augmentations)
    train_transforms = get_transforms(mode='train')
    val_transforms = None

    # Get datasets
    train_dataset = SegFormerDataset(
        root_dir=cfg.dataset_root,
        feature_extractor=feature_extractor,
        train=True,
        transforms=train_transforms,
    )
    valid_dataset = SegFormerDataset(
        root_dir=cfg.dataset_root,
        feature_extractor=feature_extractor,
        train=False,
        transforms=val_transforms,
    )

    # Get dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size)

    # Get the model
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.model_type,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Load checkpoint if necessary
    if cfg.checkpoint_load_path:
        model, optimizer, start_epoch, loss, val_iou_per_class = load_checkpoint(cfg, model, optimizer)

    else:
        start_epoch = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # initialize available device
    model.to(device)  # send model to device

    # Losses placeholders
    train_losses = []
    val_losses = []

    # Pixelwise accs placeholders
    train_pixelwise_accuracies = []
    val_pixelwise_accuracies = []

    # Initialize the list to store IoU values for each class in the validation set
    val_iou_per_class = []

    # # Initialize a variable to keep track of the best training pixel-wise accuracy
    val_pixelwise_accuracies_epoch = []
    val_losses_epoch = []

    # Best values placeholder
    best = dict()
    best['val_pixelwise_accuracy'] = 0.0  # Initialize the best validation pixel-wise accuracy
    best['model_state_dict'] = None  # Initialize the state_dict of the best model

    # IoU table data accumulation
    iou_table = IoUTable(cfg, cls_dict)

    # Iterate through epochs
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print("Epoch:", epoch)
        pbar = tqdm(
            train_dataloader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            colour='green'
        )

        # Epoch placeholders
        accuracies = []
        val_accuracies = []
        losses = []
        val_losses = []

        model.train()

        for idx, batch in enumerate(pbar):
            # Get the inputs
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(pixel_values=pixel_values, labels=labels)

            # Evaluate
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            mask = (labels != 255)  # we don't include the background class in the accuracy calculation
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels, true_labels)
            loss = outputs.loss
            accuracies.append(accuracy)
            losses.append(loss.item())
            pbar.set_postfix({'Pixel-wise accuracy': sum(accuracies) / len(accuracies), 'Loss': sum(losses) / len(losses)})

            # Backward + optimize
            loss.backward()
            optimizer.step()

        else:
            model.eval()

            with torch.no_grad():

                for idx, batch in enumerate(valid_dataloader):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(pixel_values=pixel_values, labels=labels)
                    upsampled_logits = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    predicted = upsampled_logits.argmax(dim=1)

                    mask = (labels != 255)  # we don't include the background class in the accuracy calculation
                    pred_labels = predicted[mask].cpu().detach().numpy()
                    true_labels = labels[mask].cpu().detach().numpy()
                    accuracy = accuracy_score(pred_labels, true_labels)
                    val_loss = outputs.loss
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss.item())

                    # Calculate the IoU for each class
                    iou_table.get_val_iou_per_class_epoch(predicted, labels)

        train_accuracy = sum(accuracies) / len(accuracies)
        train_loss = sum(losses) / len(losses)
        val_accuracy = sum(val_accuracies) / len(val_accuracies)
        val_loss = sum(val_losses) / len(val_losses)

        train_pixelwise_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_pixelwise_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        val_pixelwise_accuracies_epoch.append(val_accuracy)
        val_losses_epoch.append(val_loss)

        print(
            f"Train Pixel-wise accuracy: {sum(accuracies) / len(accuracies)} \
            Train Loss: {sum(losses) / len(losses)} \
            Val Pixel-wise accuracy: {sum(val_accuracies) / len(val_accuracies)} \
            Val Loss: {sum(val_losses) / len(val_losses)}"
        )

        # Display the table using tabulate
        iou_table.update_data()

        if val_accuracy > best['val_pixelwise_accuracy']:
            best['val_pixelwise_accuracy'] = val_accuracy
            best['model_state_dict'] = model.state_dict()
            best['optimizer_state_dict'] = optimizer.state_dict()
            best['loss'] = loss
            best['val_iou_per_class'] = val_iou_per_class
            best['epoch'] = epoch

    # Save the best checkpoint after all epochs have completed
    if best['model_state_dict']:
        save_checkpoint(cfg, best)

    # Calculate the cumulative IoU values using the accumulated values
    iou_table.print_cumulative_iou()

    # Plotting training pixel-wise accuracies and losses
    plot_loss('Train', train_pixelwise_accuracies, train_losses)
    plot_loss('Validation', val_pixelwise_accuracies_epoch, val_losses_epoch)


if __name__ == '__main__':
    main()

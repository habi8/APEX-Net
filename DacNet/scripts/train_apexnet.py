import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
import wandb

# Allow running this file directly from scripts\ while importing models\.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.apam import APEXNet


DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
_PRIOR_ARRAY_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train APEX-Net on NIH ChestX-ray14 with lung and prior maps"
    )
    parser.add_argument("--data_dir", required=True, help="NIH dataset directory")
    parser.add_argument(
        "--lung_mask_dir", default=None,
        help="Lung masks root (default: <data_dir>/lung_masks)",
    )
    parser.add_argument(
        "--prior_map_dir", default=None,
        help="Disease .npy maps directory (default: <project>/prior_maps)",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default=None, help="Defaults to CUDA when available, otherwise CPU"
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Initialize DenseNet-121 without ImageNet weights",
    )
    parser.add_argument("--wandb_project", default="APEX-Net ChestX-ray14")
    return parser.parse_args()


def label_vector(label_string):
    labels = label_string.split("|")
    if labels == ["No Finding"]:
        return [0.0] * len(DISEASES)
    return [float(disease in labels) for disease in DISEASES]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.exp(-bce)
        return (self.alpha * (1 - probability).pow(self.gamma) * bce).mean()


class APEXDataset(Dataset):
    """Load aligned image, lung mask, static disease priors, and labels."""

    def __init__(self, dataframe, image_to_path, data_dir, lung_mask_dir,
                 prior_map_dir, training):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_to_path = image_to_path
        self.data_dir = os.path.abspath(data_dir)
        self.lung_mask_dir = os.path.abspath(lung_mask_dir)
        self.training = training
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

        missing_maps = [
            os.path.join(prior_map_dir, f"{disease}.npy")
            for disease in DISEASES
            if not os.path.isfile(os.path.join(prior_map_dir, f"{disease}.npy"))
        ]
        if missing_maps:
            raise FileNotFoundError(
                "Missing disease prior maps: " + ", ".join(missing_maps)
            )
        self.prior_map_paths = []
        for disease in DISEASES:
            path = os.path.join(prior_map_dir, f"{disease}.npy")
            prior = np.load(path, mmap_mode="r")
            if prior.ndim != 2 or not np.isfinite(prior).all():
                raise ValueError(f"Prior map must be a finite 2D array: {path}")
            self.prior_map_paths.append(path)

        missing_masks = []
        self.mask_paths = {}
        for image_name in self.dataframe["Image Index"]:
            image_path = self.image_to_path[image_name]
            relative_path = os.path.relpath(image_path, self.data_dir)
            mask_path = os.path.join(
                self.lung_mask_dir, os.path.splitext(relative_path)[0] + ".png"
            )
            if not os.path.isfile(mask_path):
                missing_masks.append(mask_path)
            self.mask_paths[image_name] = mask_path
        if missing_masks:
            examples = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"{len(missing_masks)} lung masks are missing; examples: {examples}"
            )

    def __len__(self):
        return len(self.dataframe)

    @staticmethod
    def _resize_to_image(image, size, interpolation):
        if image.size != (size[1], size[0]):
            image = TF.resize(image, list(size), interpolation=interpolation)
        return image

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_name = row["Image Index"]
        image_path = self.image_to_path[image_name]

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        mask_path = self.mask_paths[image_name]
        with Image.open(mask_path) as source:
            lung_mask = source.convert("L")

        lung_mask = self._resize_to_image(
            lung_mask, (height, width), InterpolationMode.NEAREST
        )
        cache_key = tuple(self.prior_map_paths)
        if cache_key not in _PRIOR_ARRAY_CACHE:
            _PRIOR_ARRAY_CACHE[cache_key] = [
                np.load(path) for path in self.prior_map_paths
            ]
        priors = [
            self._resize_to_image(
                Image.fromarray(np.asarray(prior, dtype=np.float32)),
                (height, width),
                InterpolationMode.BILINEAR,
            )
            for prior in _PRIOR_ARRAY_CACHE[cache_key]
        ]

        if self.training:
            top, left, crop_height, crop_width = transforms.RandomResizedCrop.get_params(
                image, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)
            )
            resize_options = {
                "top": top,
                "left": left,
                "height": crop_height,
                "width": crop_width,
                "size": [IMAGE_SIZE, IMAGE_SIZE],
            }
            image = TF.resized_crop(
                image, **resize_options, interpolation=InterpolationMode.BILINEAR
            )
            lung_mask = TF.resized_crop(
                lung_mask, **resize_options, interpolation=InterpolationMode.NEAREST
            )
            priors = [
                TF.resized_crop(
                    prior, **resize_options, interpolation=InterpolationMode.BILINEAR
                )
                for prior in priors
            ]
            if random.random() < 0.5:
                image = TF.hflip(image)
                lung_mask = TF.hflip(lung_mask)
                priors = [TF.hflip(prior) for prior in priors]
            image = self.color_jitter(image)
        else:
            resize_size = 256
            image = TF.resize(image, resize_size, interpolation=InterpolationMode.BILINEAR)
            lung_mask = TF.resize(
                lung_mask, resize_size, interpolation=InterpolationMode.NEAREST
            )
            priors = [
                TF.resize(prior, resize_size, interpolation=InterpolationMode.BILINEAR)
                for prior in priors
            ]
            image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
            lung_mask = TF.center_crop(lung_mask, [IMAGE_SIZE, IMAGE_SIZE])
            priors = [
                TF.center_crop(prior, [IMAGE_SIZE, IMAGE_SIZE]) for prior in priors
            ]

        image_tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
        mask_tensor = (TF.to_tensor(lung_mask) > 0).to(dtype=torch.float32)
        prior_tensor = torch.stack([TF.to_tensor(prior).squeeze(0) for prior in priors])
        labels = torch.tensor(label_vector(row["Finding Labels"]), dtype=torch.float32)
        return image_tensor, mask_tensor, prior_tensor, labels


def build_image_index(data_dir):
    image_to_path = {}
    for index in range(1, 13):
        folder = os.path.join(data_dir, f"images_{index:03d}", "images")
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_to_path[filename] = os.path.join(folder, filename)
    return image_to_path


def resolve_lung_mask_dir(requested_dir, data_dir, project_dir, dataframe, image_to_path):
    if requested_dir:
        return os.path.abspath(requested_dir)

    candidates = [
        os.path.join(data_dir, "lung_masks"),
        os.path.join(project_dir, "lung_masks"),
    ]
    candidates = list(dict.fromkeys(os.path.abspath(path) for path in candidates))
    best_candidate = candidates[0]
    best_matches = -1
    for candidate in candidates:
        matches = 0
        for image_name in dataframe["Image Index"]:
            image_path = image_to_path[image_name]
            relative_path = os.path.relpath(image_path, data_dir)
            mask_path = os.path.join(
                candidate, os.path.splitext(relative_path)[0] + ".png"
            )
            matches += os.path.isfile(mask_path)
        if matches > best_matches:
            best_candidate = candidate
            best_matches = matches
    return best_candidate


def optimal_thresholds(labels, predictions):
    thresholds = []
    for index in range(predictions.shape[1]):
        precision, recall, candidates = precision_recall_curve(
            labels[:, index], predictions[:, index]
        )
        f1_values = 2 * precision * recall / (precision + recall + 1e-8)
        thresholds.append(
            candidates[np.argmax(f1_values[:-1])] if len(candidates) else 0.5
        )
    return thresholds


def evaluate(model, loader, criterion, device, description):
    model.eval()
    losses = []
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, lung_masks, priors, labels in tqdm(loader, desc=description):
            images = images.to(device, non_blocking=True)
            lung_masks = lung_masks.to(device, non_blocking=True)
            priors = priors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images, lung_masks, priors)
            losses.append(criterion(logits, labels).item())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(torch.sigmoid(logits).cpu().numpy())

    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    thresholds = optimal_thresholds(labels, predictions)
    auc_scores = [
        roc_auc_score(labels[:, i], predictions[:, i])
        if len(np.unique(labels[:, i])) > 1 else np.nan
        for i in range(len(DISEASES))
    ]
    binary_predictions = predictions > np.asarray(thresholds)[None, :]
    f1_scores = [
        f1_score(labels[:, i], binary_predictions[:, i], zero_division=0)
        for i in range(len(DISEASES))
    ]
    average_auc = float(np.nanmean(auc_scores))
    average_f1 = float(np.mean(f1_scores))
    print(f"{description} loss={np.mean(losses):.4f}, AUC={average_auc:.4f}, F1={average_f1:.4f}")
    return {
        "loss": float(np.mean(losses)),
        "avg_auc": average_auc,
        "avg_f1": average_f1,
        "auc_dict": dict(zip(DISEASES, auc_scores)),
        "f1_dict": dict(zip(DISEASES, f1_scores)),
        "thresholds": dict(zip(DISEASES, thresholds)),
    }


def train_epoch(epoch, model, loader, optimizer, criterion, device, epochs):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
    for step, (images, lung_masks, priors, labels) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        lung_masks = lung_masks.to(device, non_blocking=True)
        priors = priors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, lung_masks, priors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=total_loss / (step + 1))
    return total_loss / len(loader)


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    project_dir = PROJECT_DIR
    prior_map_dir = args.prior_map_dir or os.path.join(project_dir, "prior_maps")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    image_to_path = build_image_index(data_dir)
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe[dataframe["Image Index"].isin(image_to_path)].copy()
    if dataframe.empty:
        raise ValueError(f"No dataset images found under: {data_dir}")
    if dataframe["Patient ID"].nunique() < 3:
        raise ValueError("At least three distinct patients are required for train/val/test splits")
    lung_mask_dir = resolve_lung_mask_dir(
        args.lung_mask_dir, data_dir, project_dir, dataframe, image_to_path
    )
    print(f"Using lung masks from: {lung_mask_dir}")

    train_val_patients, test_patients = train_test_split(
        dataframe["Patient ID"].unique(), test_size=0.02, random_state=args.seed
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.052, random_state=args.seed
    )
    train_df = dataframe[dataframe["Patient ID"].isin(train_patients)]
    val_df = dataframe[dataframe["Patient ID"].isin(val_patients)]
    test_df = dataframe[dataframe["Patient ID"].isin(test_patients)]

    datasets = {
        "train": APEXDataset(
            train_df, image_to_path, data_dir, lung_mask_dir, prior_map_dir, training=True
        ),
        "validation": APEXDataset(
            val_df, image_to_path, data_dir, lung_mask_dir, prior_map_dir, training=False
        ),
        "test": APEXDataset(
            test_df, image_to_path, data_dir, lung_mask_dir, prior_map_dir, training=False
        ),
    }
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **loader_options),
        "validation": DataLoader(datasets["validation"], shuffle=False, **loader_options),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_options),
    }

    model = APEXNet(num_classes=len(DISEASES), pretrained=not args.no_pretrained).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=1, factor=0.1
    )

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "optimizer": "AdamW",
        "loss_fn": "FocalLoss",
        "augmentation": "RandomResizedCrop + RandomHorizontalFlip + ColorJitter",
        "model_architecture": "DenseNet121 + per-class lung/prior APAM",
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["validation"]),
        "test_samples": len(datasets["test"]),
    })
    wandb.init(project=args.wandb_project, config=config)
    wandb.watch(model, log="all")

    checkpoint_dir = os.path.join("models", wandb.run.id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_auc = -float("inf")
    patience_counter = 0
    best_checkpoint = None
    try:
        for epoch in range(args.epochs):
            train_loss = train_epoch(
                epoch, model, loaders["train"], optimizer, criterion, device, args.epochs
            )
            val_stats = evaluate(
                model, loaders["validation"], criterion, device, "[Validate]"
            )
            scheduler.step(val_stats["loss"])
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_stats["loss"],
                "val_auc": val_stats["avg_auc"],
                "val_f1": val_stats["avg_f1"],
                "auc_dict": val_stats["auc_dict"],
                "f1_dict": val_stats["f1_dict"],
                "optimal_thresholds": val_stats["thresholds"],
            })

            if val_stats["avg_auc"] > best_val_auc:
                best_val_auc = val_stats["avg_auc"]
                patience_counter = 0
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                best_checkpoint = os.path.join(
                    checkpoint_dir, f"best_model_{timestamp}.pth"
                )
                torch.save(model.state_dict(), best_checkpoint)
                wandb.save(best_checkpoint)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break

        if best_checkpoint is None:
            raise RuntimeError("Training did not produce a validation checkpoint")
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        test_stats = evaluate(model, loaders["test"], criterion, device, "[Test]")
        wandb.log({
            "test_loss": test_stats["loss"],
            "test_auc": test_stats["avg_auc"],
            "test_f1": test_stats["avg_f1"],
            "test_auc_dict": test_stats["auc_dict"],
            "test_f1_dict": test_stats["f1_dict"],
        })
        print(f"Best checkpoint: {best_checkpoint}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()

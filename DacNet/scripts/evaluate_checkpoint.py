import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import densenet121
from tqdm.auto import tqdm


DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
SEED = 42


class CheXNetDataset(Dataset):
    def __init__(self, dataframe, image_to_folder, transform):
        self.dataframe = dataframe
        self.image_to_folder = image_to_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_name = row["Image Index"]
        image_path = os.path.join(self.image_to_folder[image_name], image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        labels = row["Finding Labels"].split("|")
        label_vector = [int(disease in labels) for disease in DISEASES]
        return image, torch.tensor(label_vector, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DACNet checkpoint on the held-out NIH test split"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="NIH data directory containing Data_Entry_2017.csv and images_001 ... images_012",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pth checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_test_data(data_dir, batch_size):
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    image_to_folder = {}
    for index in range(1, 13):
        folder = os.path.join(data_dir, f"images_{index:03d}", "images")
        if os.path.isdir(folder):
            for image_name in os.listdir(folder):
                if image_name.endswith(".png"):
                    image_to_folder[image_name] = folder

    dataframe = dataframe[dataframe["Image Index"].isin(image_to_folder)].copy()
    patients = dataframe["Patient ID"].unique()
    train_val_patients, test_patients = train_test_split(
        patients, test_size=0.02, random_state=SEED
    )
    _, validation_patients = train_test_split(
        train_val_patients, test_size=0.052, random_state=SEED
    )

    test_dataframe = dataframe[dataframe["Patient ID"].isin(test_patients)]
    validation_dataframe = dataframe[dataframe["Patient ID"].isin(validation_patients)]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    validation_loader = DataLoader(
        CheXNetDataset(validation_dataframe, image_to_folder, transform),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        CheXNetDataset(test_dataframe, image_to_folder, transform),
        batch_size=batch_size,
        shuffle=False,
    )
    return validation_loader, test_loader


def predict(model, loader, device, description):
    model.eval()
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description):
            logits = model(images.to(device))
            labels_all.append(labels.numpy())
            predictions_all.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(predictions_all)


def validation_thresholds(labels, predictions):
    thresholds = []
    for index in range(len(DISEASES)):
        if len(np.unique(labels[:, index])) < 2:
            thresholds.append(0.5)
            continue
        precision, recall, candidates = precision_recall_curve(
            labels[:, index], predictions[:, index]
        )
        f1_values = 2 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-8
        )
        thresholds.append(float(candidates[np.argmax(f1_values)]))
    return np.asarray(thresholds)


def report_metrics(labels, predictions, thresholds):
    binary_predictions = predictions >= thresholds
    rows = []
    for index, disease in enumerate(DISEASES):
        actual = labels[:, index]
        predicted = binary_predictions[:, index]
        true_positive = np.sum((actual == 1) & predicted)
        true_negative = np.sum((actual == 0) & ~predicted)
        false_positive = np.sum((actual == 0) & predicted)
        false_negative = np.sum((actual == 1) & ~predicted)
        auc = (
            roc_auc_score(actual, predictions[:, index])
            if len(np.unique(actual)) > 1 else np.nan
        )
        rows.append({
            "disease": disease,
            "threshold": thresholds[index],
            "auc": auc,
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "specificity": true_negative / max(true_negative + false_positive, 1),
            "f1": f1_score(actual, predicted, zero_division=0),
            "accuracy": (true_positive + true_negative) / len(actual),
        })

    print("\nTest-set metrics (thresholds selected on validation split)")
    print(f"{'Disease':<22} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'Spec':>7} {'F1':>7} {'Acc':>7}")
    for row in rows:
        print(
            f"{row['disease']:<22} {row['auc']:>7.4f} {row['precision']:>7.4f} "
            f"{row['recall']:>7.4f} {row['specificity']:>7.4f} "
            f"{row['f1']:>7.4f} {row['accuracy']:>7.4f}"
        )
    print(
        f"{'Macro average':<22} {np.nanmean([r['auc'] for r in rows]):>7.4f} "
        f"{np.mean([r['precision'] for r in rows]):>7.4f} "
        f"{np.mean([r['recall'] for r in rows]):>7.4f} "
        f"{np.mean([r['specificity'] for r in rows]):>7.4f} "
        f"{np.mean([r['f1'] for r in rows]):>7.4f} "
        f"{np.mean([r['accuracy'] for r in rows]):>7.4f}"
    )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but is not available")

    device = torch.device(args.device)
    validation_loader, test_loader = load_test_data(args.data_dir, args.batch_size)

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    validation_labels, validation_predictions = predict(
        model, validation_loader, device, "Validation"
    )
    thresholds = validation_thresholds(validation_labels, validation_predictions)
    test_labels, test_predictions = predict(model, test_loader, device, "Test")
    report_metrics(test_labels, test_predictions, thresholds)


if __name__ == "__main__":
    main()
import os, argparse, time
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm.auto import tqdm
import wandb

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "model": "dacnet_fzlpr",          # Updated name
    "batch_size": 16,
    "learning_rate": 0.00005,
    "epochs": 25,
    "num_workers": 2,
    "device": cuda",
    "data_dir": None,
    "wandb_project": "X-Ray Classification",
    "patience": 5,
    "seed": 42,
    "image_size": 224,
    "tau": 0.2,                         # Paper 1: FZLPR temperature
    "use_tta": True,                    # Paper 1: test-time augmentation
}

# ─────────────────────────────────────────────────────────────────────
# PAPER 1: FZLPR LOSS
# ─────────────────────────────────────────────────────────────────────
class FZLPRLoss(nn.Module):
    """
    Focal ZLPR loss (Hanif et al., 2025).
    FZLPR = log(1 + Σ e^(-f_i/τ)) + log(1 + Σ e^(f_j/τ))
    where i ∈ positive labels, j ∈ negative labels.
    """
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau

    def forward(self, logits, targets):
        scaled = logits / self.tau
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)

        # Safe handling of edge cases
        pos_sum = torch.sum(torch.exp(-scaled * pos_mask), dim=1)
        neg_sum = torch.sum(torch.exp(scaled * neg_mask), dim=1)

        pos_term = torch.log(1 + pos_sum)
        neg_term = torch.log(1 + neg_sum)

        return torch.mean(pos_term + neg_term)


# ─────────────────────────────────────────────────────────────────────
# PAPER 1: TEST-TIME AUGMENTATION
# ─────────────────────────────────────────────────────────────────────
def apply_tta(model, image_batch, device):
    """Horizontal flip + zoom (±10%), average predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        # Original
        all_preds.append(torch.sigmoid(model(image_batch.to(device))))

        # Horizontal flip
        flipped = torch.flip(image_batch, dims=[3])
        all_preds.append(torch.sigmoid(model(flipped.to(device))))

        # Zoom
        for scale in [0.9, 1.1]:
            size = int(224 * scale)
            zoomed = transforms.functional.resize(image_batch, size)
            zoomed = transforms.functional.center_crop(zoomed, 224)
            all_preds.append(torch.sigmoid(model(zoomed.to(device))))

    return torch.stack(all_preds).mean(dim=0)


# ─────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train DACNet+FZLPR on NIH ChestX-ray14")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to NIH data directory")
    parser.add_argument("--tau", type=float, default=0.2,
                        help="FZLPR temperature (Paper 1)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation")
    return parser.parse_args()

args = parse_args()
CONFIG["data_dir"] = args.data_dir
CONFIG["tau"] = args.tau
if args.no_tta:
    CONFIG["use_tta"] = False

# ─────────────────────────────────────────────────────────────────────
# TRANSFORMS (DACNet augmentations + resize strategy)
# ─────────────────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────
# MODEL (DenseNet121 + Paper 1 custom top network — simplified version)
# ─────────────────────────────────────────────────────────────────────
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features

# Paper 1-style custom top (simplified: dropout + intermediate dense + BatchNorm)
model.classifier = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(num_features, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 14),
    nn.BatchNorm1d(14)
)
model = model.to(CONFIG["device"])

# ─────────────────────────────────────────────────────────────────────
# LOSS, OPTIMIZER, SCHEDULER
# ─────────────────────────────────────────────────────────────────────
criterion = FZLPRLoss(tau=CONFIG["tau"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=1, factor=0.1
)

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
data_path = os.path.abspath(CONFIG["data_dir"])
csv_file = os.path.join(data_path, "Data_Entry_2017.csv")
df = pd.read_csv(csv_file)

image_folders = [
    os.path.join(data_path, f"images_{str(i).zfill(3)}", "images")
    for i in range(1, 13)
]
image_to_folder = {}
for folder in image_folders:
    if os.path.exists(folder):
        for img_file in os.listdir(folder):
            if img_file.endswith('.png'):
                image_to_folder[img_file] = folder

df = df[df['Image Index'].isin(image_to_folder.keys())]

unique_patients = df['Patient ID'].unique()
train_val_patients, test_patients = train_test_split(
    unique_patients, test_size=0.02, random_state=CONFIG["seed"]
)
train_patients, val_patients = train_test_split(
    train_val_patients, test_size=0.052, random_state=CONFIG["seed"]
)
train_df = df[df['Patient ID'].isin(train_patients)]
val_df = df[df['Patient ID'].isin(val_patients)]
test_df = df[df['Patient ID'].isin(test_patients)]

disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

def get_label_vector(labels_str):
    labels = labels_str.split('|')
    if labels == ['No Finding']:
        return [0] * len(disease_list)
    return [1 if disease in labels else 0 for disease in disease_list]

class CheXNetDataset(Dataset):
    def __init__(self, dataframe, image_to_folder, transform=None):
        self.dataframe = dataframe
        self.image_to_folder = image_to_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        folder = self.image_to_folder[img_name]
        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels_str = self.dataframe.iloc[idx]['Finding Labels']
        label_vector = get_label_vector(labels_str)
        labels = torch.tensor(label_vector, dtype=torch.float)
        return image, labels

train_dataset = CheXNetDataset(train_df, image_to_folder, transform_train)
val_dataset = CheXNetDataset(val_df, image_to_folder, transform_test)
test_dataset = CheXNetDataset(test_df, image_to_folder, transform_test)

trainloader = DataLoader(
    train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=CONFIG["num_workers"]
)
valloader = DataLoader(
    val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=CONFIG["num_workers"]
)
testloader = DataLoader(
    test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=CONFIG["num_workers"]
)

# ─────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION (DACNet feature)
# ─────────────────────────────────────────────────────────────────────
def get_optimal_thresholds(labels, preds):
    thresholds = []
    for i in range(preds.shape[1]):
        precision, recall, thresh = precision_recall_curve(
            labels[:, i], preds[:, i]
        )
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5
        thresholds.append(best_threshold)
    return thresholds

# ─────────────────────────────────────────────────────────────────────
# EVALUATION (with optional TTA)
# ─────────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device, desc="[Test]", use_tta=False):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device)

            if use_tta:
                preds = apply_tta(model, inputs, device)
                outputs = model(inputs)           # For loss calculation only
                loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs)

            running_loss += loss.item()
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    thresholds = get_optimal_thresholds(all_labels, all_preds)
    preds_binary = np.zeros_like(all_preds)
    for i in range(all_preds.shape[1]):
        preds_binary[:, i] = (all_preds[:, i] > thresholds[i]).astype(int)

    auc_scores = [
        roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(14)
    ]
    f1_scores = [
        f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(14)
    ]
    avg_auc = np.mean(auc_scores)
    avg_f1 = np.mean(f1_scores)

    for i, disease in enumerate(disease_list):
        print(f"{desc} {disease} AUC: {auc_scores[i]:.4f} | F1: {f1_scores[i]:.4f}")
    print(f"{desc} Avg AUC: {avg_auc:.4f}, Avg F1: {avg_f1:.4f}")

    return {
        "loss": running_loss / len(loader),
        "avg_auc": avg_auc,
        "avg_f1": avg_f1,
        "auc_dict": dict(zip(disease_list, auc_scores)),
        "f1_dict": dict(zip(disease_list, f1_scores)),
        "thresholds": dict(zip(disease_list, thresholds))
    }

# ─────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]",
                        leave=True)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})
    return running_loss / len(trainloader)

def validate(model, valloader, criterion, device):
    return evaluate(model, valloader, criterion, device, desc="[Validate]")

# ─────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────
wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
wandb.watch(model, log="all")
wandb.config.update({
    "model_architecture": "DenseNet121",
    "loss_fn": "FZLPRLoss",
    "tau": CONFIG["tau"],
    "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau",
    "augmentation": "RandomResizedCrop + ColorJitter + HorizontalFlip",
    "use_tta": CONFIG["use_tta"],
    "custom_top": "Dropout + Dense(64) + BN",
})

run_id = wandb.run.id
checkpoint_dir = os.path.join("models", run_id)
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_auc = 0.0
patience_counter = 0

for epoch in range(CONFIG["epochs"]):
    train_loss = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
    val_stats = validate(model, valloader, criterion, CONFIG["device"])
    scheduler.step(val_stats["loss"])

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_stats["loss"],
        "val_auc": val_stats["avg_auc"],
        "val_f1": val_stats["avg_f1"],
        "f1_dict": val_stats["f1_dict"],
        "auc_dict": val_stats["auc_dict"],
        "optimal_thresholds": val_stats["thresholds"],
    })

    if val_stats["avg_auc"] > best_val_auc:
        best_val_auc = val_stats["avg_auc"]
        patience_counter = 0
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered.")
            break

# ─────────────────────────────────────────────────────────────────────
# FINAL TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────
best_checkpoint_path = sorted([
    os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
    if f.startswith('best_model_')
])[-1]
model.load_state_dict(torch.load(best_checkpoint_path))

test_stats = evaluate(
    model, testloader, criterion, CONFIG["device"],
    desc="[Test]", use_tta=CONFIG["use_tta"]
)
wandb.log({
    "test_loss": test_stats["loss"],
    "test_auc": test_stats["avg_auc"],
    "test_f1": test_stats["avg_f1"],
    "test_auc_dict": test_stats["auc_dict"],
    "test_f1_dict": test_stats["f1_dict"],
})
wandb.finish()
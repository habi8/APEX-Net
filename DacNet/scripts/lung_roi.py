import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESIZE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESIZE_NEAREST = getattr(Image, "Resampling", Image).NEAREST

DEFAULT_CONFIG = {
    "images_dir": r"C:\JSRT\jsrt\cxr",
    "masks_dir": r"C:\JSRT\jsrt\masks",
    "output": "models/lung_unet.pth",
    "data_dir": r"C:\NIH_data",
    "checkpoint": "models/lung_unet.pth",
    "device": "cuda",
    "epochs": 50,
    "batch_size": 32,
    "image_size": 256,
    "base_channels": 32,
    "learning_rate": 1e-3,
    "num_workers": 8,
    "seed": 42,
    "threshold": 0.5,
    "dilation_radius": 5,
}


def resolve_device(device):
    return device or DEFAULT_CONFIG["device"]


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class LungUNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.encoder1 = ConvBlock(1, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.decoder4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, inputs):
        encoded1 = self.encoder1(inputs)
        encoded2 = self.encoder2(self.pool(encoded1))
        encoded3 = self.encoder3(self.pool(encoded2))
        encoded4 = self.encoder4(self.pool(encoded3))
        center = self.bottleneck(self.pool(encoded4))
        decoded4 = self.decoder4(torch.cat((self.up4(center), encoded4), dim=1))
        decoded3 = self.decoder3(torch.cat((self.up3(decoded4), encoded3), dim=1))
        decoded2 = self.decoder2(torch.cat((self.up2(decoded3), encoded2), dim=1))
        decoded1 = self.decoder1(torch.cat((self.up1(decoded2), encoded1), dim=1))
        return self.output(decoded1)


def list_images(directory):
    paths = []
    for root, _, filenames in os.walk(directory):
        paths.extend(
            os.path.join(root, filename)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        )
    return sorted(paths)


def find_mask_paths(mask_dir, image_stem):
    masks = []
    for suffix in ("", "_L", "_R", "_l", "_r"):
        for extension in IMAGE_EXTENSIONS:
            candidate = os.path.join(mask_dir, image_stem + suffix + extension)
            if os.path.isfile(candidate):
                masks.append(candidate)
                break
    return masks


class JSRTDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size):
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"JSRT image directory not found: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"JSRT mask directory not found: {masks_dir}")
        self.image_size = image_size
        self.samples = []
        unmatched = []
        for image_path in list_images(images_dir):
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_paths = find_mask_paths(masks_dir, image_stem)
            if not mask_paths:
                unmatched.append(image_path)
                continue
            self.samples.append((image_path, mask_paths))

        if not self.samples:
            raise FileNotFoundError(
                "No paired JSRT images and masks found. Use matching image/mask "
                "stems, or separate masks named <stem>_L and <stem>_R."
            )
        if unmatched:
            examples = ", ".join(os.path.basename(path) for path in unmatched[:5])
            raise ValueError(
                f"{len(unmatched)} JSRT images have no matching mask "
                f"(examples: {examples})"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_paths = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("L").resize(
                (self.image_size, self.image_size), RESIZE_BILINEAR
            )
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        mask_array = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        for mask_path in mask_paths:
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L").resize(
                    (self.image_size, self.image_size), RESIZE_NEAREST
                )
                mask_array |= np.asarray(mask, dtype=np.uint8) > 0

        image_tensor = torch.from_numpy(image_array[None, ...])
        mask_tensor = torch.from_numpy(mask_array.astype(np.float32)[None, ...])
        return image_tensor, mask_tensor


class TrainingDataset(Dataset):
    def __init__(self, dataset, augment):
        self.dataset = dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(2,))
            mask = torch.flip(mask, dims=(2,))
        return image, mask


def parse_train_args(parser):
    parser.add_argument("--images_dir", default=DEFAULT_CONFIG["images_dir"], help="JSRT chest X-ray image directory")
    parser.add_argument("--masks_dir", default=DEFAULT_CONFIG["masks_dir"], help="Directory containing JSRT lung masks")
    parser.add_argument("--output", default=DEFAULT_CONFIG["output"], help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"])
    parser.add_argument("--base_channels", type=int, default=DEFAULT_CONFIG["base_channels"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"], help="Defaults to CUDA when available, otherwise CPU")


def parse_generate_args(parser):
    parser.add_argument("--data_dir", default=DEFAULT_CONFIG["data_dir"], help="NIH image root containing images_001 ... images_012")
    parser.add_argument("--checkpoint", default=DEFAULT_CONFIG["checkpoint"], help="Checkpoint produced by the train command")
    parser.add_argument("--output_dir", default=None, help="Defaults to <data_dir>/lung_masks")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"], help="Foreground probability threshold")
    parser.add_argument("--dilation_radius", type=int, default=DEFAULT_CONFIG["dilation_radius"], help="Post-processing dilation radius in pixels")
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"], help="Defaults to CUDA when available, otherwise CPU")


def dice_bce_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + (1.0 - dice).mean()


def train(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch_size must be positive")
    if args.image_size < 32 or args.image_size % 16:
        raise ValueError("--image_size must be a multiple of 16 and at least 32")
    if args.base_channels < 1:
        raise ValueError("--base_channels must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))

    dataset = JSRTDataset(args.images_dir, args.masks_dir, args.image_size)
    if len(dataset) < 2:
        raise ValueError("At least two paired JSRT images are required for training and validation")
    validation_size = max(1, round(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(args.seed)
    training_subset, validation_subset = random_split(
        dataset, (training_size, validation_size), generator=generator
    )
    training_data = TrainingDataset(training_subset, augment=True)
    validation_data = TrainingDataset(validation_subset, augment=False)
    training_loader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = LungUNet(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    checkpoint_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for images, masks in tqdm(training_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = dice_bce_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * images.size(0)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device)
                validation_loss += dice_bce_loss(model(images), masks).item() * images.size(0)

        training_loss /= training_size
        validation_loss /= validation_size
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={training_loss:.4f}, val_loss={validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                },
                args.output,
            )

    print(f"Best checkpoint saved to {args.output}")


def postprocess_mask(binary_mask, dilation_radius):
    if dilation_radius < 0:
        raise ValueError("--dilation_radius cannot be negative")
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if binary_mask.ndim != 2:
        raise ValueError("Expected a two-dimensional lung mask")

    labels, component_count = ndimage.label(binary_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return np.zeros(binary_mask.shape, dtype=np.uint8)

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_labels = np.argsort(component_sizes)[-2:]
    height, width = binary_mask.shape
    hull_mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(hull_mask)

    for component_label in largest_labels:
        if component_sizes[component_label] == 0:
            continue
        rows, columns = np.nonzero(labels == component_label)
        points = np.column_stack((columns, rows))
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
            except QhullError:
                for column, row in points:
                    draw.point((int(column), int(row)), fill=1)
            else:
                polygon = [tuple(point) for point in points[hull.vertices]]
                draw.polygon(polygon, fill=1)
        else:
            for column, row in points:
                draw.point((int(column), int(row)), fill=1)

    processed = np.asarray(hull_mask, dtype=bool)
    if dilation_radius:
        y, x = np.ogrid[-dilation_radius:dilation_radius + 1, -dilation_radius:dilation_radius + 1]
        disk = x * x + y * y <= dilation_radius * dilation_radius
        processed = ndimage.binary_dilation(processed, structure=disk)
    return processed.astype(np.uint8)


def load_checkpoint(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be created by the lung_roi.py train command")
    image_size = int(checkpoint["image_size"])
    base_channels = int(checkpoint["base_channels"])
    if image_size < 32 or image_size % 16 or base_channels < 1:
        raise ValueError(f"Invalid image size in checkpoint: {image_size}")
    model = LungUNet(base_channels).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, image_size


def is_inside(path, parent):
    try:
        return os.path.normcase(os.path.commonpath((path, parent))) == os.path.normcase(parent)
    except ValueError:
        return False


def generate_masks(args):
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.dilation_radius < 0:
        raise ValueError("--dilation_radius cannot be negative")

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(data_dir, "lung_masks"))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"NIH data directory not found: {data_dir}")
    if os.path.normcase(data_dir) == os.path.normcase(output_dir):
        raise ValueError("Output directory cannot be the NIH data directory itself")

    device = torch.device(resolve_device(args.device))
    model, image_size = load_checkpoint(args.checkpoint, device)
    image_paths = []
    for root, directories, filenames in os.walk(data_dir):
        directories[:] = [
            directory for directory in directories
            if not is_inside(os.path.abspath(os.path.join(root, directory)), output_dir)
        ]
        image_paths.extend(
            os.path.join(root, filename)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        )
    image_paths.sort()
    if not image_paths:
        raise FileNotFoundError(f"No supported NIH images found under: {data_dir}")

    os.makedirs(output_dir, exist_ok=True)
    for image_path in tqdm(image_paths, desc="Generating lung masks"):
        relative_path = os.path.relpath(image_path, data_dir)
        mask_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        with Image.open(image_path) as source:
            image = source.convert("L")
            original_width, original_height = image.size
            resized = image.resize((image_size, image_size), RESIZE_BILINEAR)
            image_array = np.asarray(resized, dtype=np.float32) / 255.0
        inputs = torch.from_numpy(image_array[None, None, ...]).to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(inputs))
            probabilities = F.interpolate(
                probabilities,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
        mask = postprocess_mask(probabilities >= args.threshold, args.dilation_radius)
        Image.fromarray(mask * 255).save(mask_path)

    print(f"Saved {len(image_paths)} lung masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a lung U-Net or precompute NIH lung ROI masks")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train on paired JSRT images and lung masks")
    parse_train_args(train_parser)
    generate_parser = commands.add_parser("generate", help="Generate masks for all NIH chest X-rays")
    parse_generate_args(generate_parser)
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        generate_masks(args)


if __name__ == "__main__":
    main()

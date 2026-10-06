import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.apam import APEXNet
from scripts.lung_roi import load_checkpoint as load_lung_checkpoint
from scripts.lung_roi import postprocess_mask
from scripts.train_apexnet import DISEASES, IMAGE_SIZE, MEAN, STD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run APEX-Net inference on one chest X-ray"
    )
    parser.add_argument("--image", required=True, help="Chest X-ray image path")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            PROJECT_DIR, "models", "ngp6gwkn", "best_model_20261006-050300.pth"
        ),
        help="APEX-Net checkpoint",
    )
    parser.add_argument(
        "--lung_mask",
        default=None,
        help="Optional precomputed lung mask; otherwise generate one with the lung U-Net",
    )
    parser.add_argument(
        "--lung_checkpoint",
        default=os.path.join(PROJECT_DIR, "models", "lung_unet.pth"),
        help="Lung U-Net checkpoint used when --lung_mask is omitted",
    )
    parser.add_argument(
        "--prior_map_dir",
        default=os.path.join(PROJECT_DIR, "prior_maps"),
        help="Directory containing one .npy prior map per disease",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Defaults to CUDA when available, otherwise CPU",
    )
    return parser.parse_args()


def generate_lung_mask(image, checkpoint_path, device):
    model, image_size = load_lung_checkpoint(checkpoint_path, device)
    grayscale = image.convert("L")
    original_width, original_height = grayscale.size
    resized = grayscale.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    inputs = torch.from_numpy(image_array[None, None, ...]).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs))
        probabilities = torch.nn.functional.interpolate(
            probabilities,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
    mask = postprocess_mask(probabilities >= 0.5, dilation_radius=5)
    return Image.fromarray(mask * 255)


def main():
    args = parse_args()
    for path, label in (
        (args.image, "Image"),
        (args.checkpoint, "APEX-Net checkpoint"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.lung_mask and not os.path.isfile(args.lung_mask):
        raise FileNotFoundError(f"Lung mask not found: {args.lung_mask}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but is not available")
    device = torch.device(device_name)

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    width, height = image.size

    if args.lung_mask:
        with Image.open(args.lung_mask) as source:
            lung_mask = source.convert("L")
        if lung_mask.size != image.size:
            lung_mask = TF.resize(
                lung_mask, [height, width], interpolation=InterpolationMode.NEAREST
            )
    else:
        lung_mask = generate_lung_mask(image, args.lung_checkpoint, device)

    missing_maps = [
        os.path.join(args.prior_map_dir, f"{disease}.npy")
        for disease in DISEASES
        if not os.path.isfile(os.path.join(args.prior_map_dir, f"{disease}.npy"))
    ]
    if missing_maps:
        raise FileNotFoundError(
            "Missing disease prior maps: " + ", ".join(missing_maps)
        )

    resize_options = {
        "size": 256,
        "interpolation": InterpolationMode.BILINEAR,
    }
    image = TF.center_crop(TF.resize(image, **resize_options), [IMAGE_SIZE, IMAGE_SIZE])
    lung_mask = TF.center_crop(
        TF.resize(
            lung_mask, 256, interpolation=InterpolationMode.NEAREST
        ),
        [IMAGE_SIZE, IMAGE_SIZE],
    )
    prior_tensors = []
    for disease in DISEASES:
        prior = np.load(
            os.path.join(args.prior_map_dir, f"{disease}.npy"), allow_pickle=False
        )
        if prior.ndim != 2 or not np.isfinite(prior).all():
            raise ValueError(f"Prior map must be a finite 2D array: {disease}.npy")
        prior_image = Image.fromarray(np.asarray(prior, dtype=np.float32))
        if prior_image.size != (width, height):
            prior_image = TF.resize(
                prior_image, [height, width], interpolation=InterpolationMode.BILINEAR
            )
        prior_image = TF.center_crop(
            TF.resize(prior_image, **resize_options), [IMAGE_SIZE, IMAGE_SIZE]
        )
        prior_tensors.append(TF.to_tensor(prior_image).squeeze(0))

    image_tensor = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0)
    mask_tensor = (TF.to_tensor(lung_mask) > 0).float().unsqueeze(0)
    prior_tensor = torch.stack(prior_tensors).unsqueeze(0)

    model = APEXNet(num_classes=len(DISEASES), pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(
                image_tensor.to(device),
                mask_tensor.to(device),
                prior_tensor.to(device),
            )
        )[0].cpu().numpy()

    print(f"APEX-Net probabilities for: {args.image}")
    print("These are model scores, not a clinical diagnosis.")
    for disease, probability in sorted(
        zip(DISEASES, probabilities), key=lambda item: item[1], reverse=True
    ):
        print(f"{disease:<22} {probability * 100:5.1f}%")


if __name__ == "__main__":
    main()

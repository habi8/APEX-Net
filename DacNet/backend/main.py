import base64
import io
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models.apam import APEXNet
from scripts.lung_roi import load_checkpoint as load_lung_checkpoint
from scripts.lung_roi import postprocess_mask


PROJECT_DIR = Path(__file__).resolve().parents[1]
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MODEL_VERSION = "APEX-Net APAM 2026-10-06"


def configured_path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def has_plausible_bilateral_lung_mask(binary_mask):
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) == 0:
        return False

    height, width = mask.shape
    if width < 2:
        return False
    area = float(mask.mean())
    if not 0.04 <= area <= 0.65:
        return False
    if min(float(mask[:, :width // 2].mean()), float(mask[:, width // 2:].mean())) < 0.01:
        return False

    rows, columns = np.nonzero(mask)
    if not len(rows):
        return False
    vertical_span = (int(rows.max()) - int(rows.min()) + 1) / height
    horizontal_span = (int(columns.max()) - int(columns.min()) + 1) / width
    return vertical_span >= 0.30 and horizontal_span >= 0.25


def build_overlay(original, crop_cam, crop_mask, resized_size, crop_origin):
    resized_width, resized_height = resized_size
    crop_left, crop_top = crop_origin
    cam_canvas = np.zeros((resized_height, resized_width), dtype=np.float32)
    height, width = crop_cam.shape
    cam_canvas[crop_top:crop_top + height, crop_left:crop_left + width] = crop_cam
    cam_image = Image.fromarray(np.uint8(np.clip(cam_canvas, 0, 1) * 255), mode="L")
    cam_image = cam_image.resize(original.size, Image.Resampling.BILINEAR).filter(
        ImageFilter.GaussianBlur(radius=max(1, round(min(original.size) / 450)))
    )
    cam = np.asarray(cam_image, dtype=np.float32) / 255.0

    maximum = float(cam.max())
    if maximum > 0:
        cam /= maximum

    soft_mask_image = crop_mask.convert("L").filter(
        ImageFilter.GaussianBlur(radius=max(2, round(min(original.size) / 75)))
    )
    if soft_mask_image.size != original.size:
        soft_mask_image = soft_mask_image.resize(
            original.size, Image.Resampling.BILINEAR
        )
    soft_mask = np.asarray(soft_mask_image, dtype=np.float32) / 255.0

    red = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    colors = np.stack((red, green, blue), axis=-1)
    alpha = (cam * soft_mask * 0.50)[..., None]
    source = np.asarray(original, dtype=np.float32) / 255.0
    blended = source * (1 - alpha) + colors * alpha
    output = Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255), mode="RGB")
    buffer = io.BytesIO()
    output.save(buffer, format="WEBP", lossless=True, method=4)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def gradcam_for_class(model, outputs, activations, class_index, prior_map):
    lung_activation = activations[("lung", class_index)]
    prior_activation = activations[("prior", class_index)]
    lung_gradient, prior_gradient = torch.autograd.grad(
        outputs[0, class_index],
        (lung_activation, prior_activation),
        retain_graph=False,
        create_graph=False,
    )

    lung_cam = F.relu(
        (lung_gradient.mean(dim=(2, 3), keepdim=True) * lung_activation).sum(dim=1, keepdim=True)
    )
    prior_cam = F.relu(
        (prior_gradient.mean(dim=(2, 3), keepdim=True) * prior_activation).sum(dim=1, keepdim=True)
    )

    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    lung_cam = F.interpolate(lung_cam, size=target_size, mode="bilinear", align_corners=False)
    prior_cam = F.interpolate(prior_cam, size=target_size, mode="bilinear", align_corners=False)
    prior_map = F.interpolate(prior_map, size=target_size, mode="bilinear", align_corners=False)

    def normalize(cam):
        maximum = cam.amax(dim=(2, 3), keepdim=True)
        return torch.where(maximum > 0, cam / maximum.clamp_min(1e-8), torch.zeros_like(cam))

    combined = (0.5 * normalize(lung_cam) + 0.5 * normalize(prior_cam) * prior_map)
    combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
    maximum = combined.amax(dim=(2, 3), keepdim=True)
    combined = torch.where(
        maximum > 0, combined / maximum.clamp_min(1e-8), torch.zeros_like(combined)
    )
    return combined[0, 0].detach().cpu().numpy()


def load_resources():
    device_name = os.environ.get(
        "APEX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("APEX_DEVICE=cuda but CUDA is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("APEX_DEVICE=mps but MPS is not available")
    device = torch.device(device_name)

    apex_checkpoint = configured_path(
        "APEX_CHECKPOINT",
        PROJECT_DIR / "models" / "ngp6gwkn" / "best_model_20261006-050300.pth",
    )
    lung_checkpoint = configured_path(
        "LUNG_CHECKPOINT", PROJECT_DIR / "models" / "lung_unet.pth"
    )
    prior_map_dir = configured_path("PRIOR_MAP_DIR", PROJECT_DIR / "prior_maps")
    for path, label in (
        (apex_checkpoint, "APEX-Net checkpoint"),
        (lung_checkpoint, "lung U-Net checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    priors = []
    for disease in DISEASES:
        path = prior_map_dir / f"{disease}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Disease prior map not found: {path}")
        prior = np.load(path, allow_pickle=False)
        if prior.ndim != 2 or not np.isfinite(prior).all():
            raise ValueError(f"Disease prior map must be a finite 2D array: {path}")
        priors.append(prior.astype(np.float32, copy=False))
    prior_shape = priors[0].shape
    if any(prior.shape != prior_shape for prior in priors):
        raise ValueError("All 14 disease prior maps must have the same dimensions")

    model = APEXNet(num_classes=len(DISEASES), pretrained=False).to(device)
    state = torch.load(apex_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    lung_model, lung_image_size = load_lung_checkpoint(str(lung_checkpoint), device)
    return {
        "device": device,
        "model": model,
        "lung_model": lung_model,
        "lung_image_size": lung_image_size,
        "priors": priors,
        "inference_lock": Lock(),
        "checkpoint": str(apex_checkpoint),
    }


@asynccontextmanager
async def lifespan(app):
    app.state.resources = load_resources()
    yield


app = FastAPI(
    title="APEX-Net inference API",
    version="1.0.0",
    description="APEX-Net disease scores and full-image APAM Grad-CAM overlays.",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.get("/health")
def health():
    resources = getattr(app.state, "resources", None)
    return {
        "status": "ok" if resources else "starting",
        "model": MODEL_VERSION,
        "device": str(resources["device"]) if resources else None,
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    heatmap_count: int = Form(default=3),
    x_api_key: str | None = Header(default=None),
):
    expected_key = os.environ.get("APEX_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not 1 <= heatmap_count <= 5:
        raise HTTPException(status_code=422, detail="heatmap_count must be between 1 and 5")

    resources = app.state.resources
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise HTTPException(status_code=415, detail="Use a JPEG, PNG, WEBP, or BMP image")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the supported limit")
            source.verify()
        with Image.open(io.BytesIO(raw)) as source:
            original = source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid supported image") from exc
    if original.width < 32 or original.height < 32:
        raise HTTPException(status_code=400, detail="Image dimensions must be at least 32 by 32 pixels")

    with resources["inference_lock"]:
        device = resources["device"]
        gray = original.convert("L")
        lung_size = resources["lung_image_size"]
        lung_input = np.asarray(
            gray.resize((lung_size, lung_size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        lung_input = torch.from_numpy(lung_input[None, None, ...]).to(device)
        with torch.no_grad():
            lung_probabilities = torch.sigmoid(resources["lung_model"](lung_input))
            lung_probabilities = F.interpolate(
                lung_probabilities,
                size=(original.height, original.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
        if not has_plausible_bilateral_lung_mask(lung_probabilities >= 0.5):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not detect both lungs in this image. "
                    "Please upload a clear frontal chest X-ray."
                ),
            )
        lung_mask_image = Image.fromarray(
            postprocess_mask(lung_probabilities >= 0.5, dilation_radius=5) * 255
        )

        resized_image = TF.resize(
            original, 256, interpolation=InterpolationMode.BILINEAR
        )
        image_crop = TF.center_crop(resized_image, [IMAGE_SIZE, IMAGE_SIZE])
        resized_width, resized_height = resized_image.size
        crop_left = int(round((resized_width - IMAGE_SIZE) / 2))
        crop_top = int(round((resized_height - IMAGE_SIZE) / 2))
        resized_mask = TF.resize(
            lung_mask_image, 256, interpolation=InterpolationMode.NEAREST
        )
        mask_crop = TF.center_crop(resized_mask, [IMAGE_SIZE, IMAGE_SIZE])
        image_tensor = TF.normalize(TF.to_tensor(image_crop), MEAN, STD).unsqueeze(0).to(device)
        lung_roi = (TF.to_tensor(mask_crop) > 0).float().unsqueeze(0).to(device)

        prior_tensors = []
        for prior in resources["priors"]:
            prior_image = Image.fromarray(prior)
            if prior_image.size != original.size:
                prior_image = TF.resize(
                    prior_image,
                    [original.height, original.width],
                    interpolation=InterpolationMode.BILINEAR,
                )
            prior_image = TF.center_crop(
                TF.resize(prior_image, 256, interpolation=InterpolationMode.BILINEAR),
                [IMAGE_SIZE, IMAGE_SIZE],
            )
            prior_tensors.append(TF.to_tensor(prior_image).squeeze(0))
        prior_tensor = torch.stack(prior_tensors).unsqueeze(0).to(device)

        activations = {}
        handles = []
        for branch_name, modules in (
            ("lung", resources["model"].lung_apams),
            ("prior", resources["model"].prior_apams),
        ):
            for class_index, module in enumerate(modules):
                def capture(_module, _inputs, output, key=(branch_name, class_index)):
                    activations[key] = output

                handles.append(module.register_forward_hook(capture))
        try:
            resources["model"].zero_grad(set_to_none=True)
            logits = resources["model"](image_tensor, lung_roi, prior_tensor)
            scores = torch.sigmoid(logits[0]).detach().cpu().numpy()
            top_indices = np.argsort(scores)[::-1][:heatmap_count].tolist()
            heatmaps = {}
            for position, class_index in enumerate(top_indices):
                if position:
                    resources["model"].zero_grad(set_to_none=True)
                    activations.clear()
                    logits = resources["model"](image_tensor, lung_roi, prior_tensor)
                cam = gradcam_for_class(
                    resources["model"],
                    logits,
                    activations,
                    class_index,
                    prior_tensor[:, class_index:class_index + 1],
                )
                heatmaps[DISEASES[class_index]] = build_overlay(
                    original,
                    cam,
                    lung_mask_image,
                    resized_image.size,
                    (crop_left, crop_top),
                )
        finally:
            for handle in handles:
                handle.remove()

    findings = [
        {
            "label": DISEASES[index],
            "confidence": float(scores[index]),
            "severity": "not assessed",
            "location": (
                "Attention overlay available; not a lesion boundary"
                if DISEASES[index] in heatmaps
                else "No overlay generated"
            ),
        }
        for index in np.argsort(scores)[::-1]
    ]
    return {
        "success": True,
        "prediction": {
            "findings": findings,
            "heatmaps": heatmaps,
            "heatmap_method": "Class-specific APAM branch Grad-CAM with a softly feathered U-Net lung ROI overlay",
            "overall_assessment": (
                "APEX-Net produced the listed multi-label model scores. "
                "Scores are not calibrated diagnostic probabilities."
            ),
            "recommendations": [],
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
        },
    }

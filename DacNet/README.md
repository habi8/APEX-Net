# DACNet: Reproduction and Extension of CheXNet for Chest X-ray Classification

This repository contains:
1. a **reproduction** of the CheXNet DenseNet-121 model on the NIH ChestX-ray14 dataset, and  
2. an **extension (DACNet)** designed to improve performance under class imbalance.

It also includes a Vision Transformer baseline and a Streamlit demo for inference.

---

## Reproduction Scope

This work reproduces key components of:

**CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning**  
Rajpurkar et al., 2017 (arXiv:1711.05225)

### What is reproduced
- DenseNet-121 architecture
- Multi-label classification (14 thoracic diseases)
- Patient-level dataset splitting
- Evaluation using AUC-ROC and F1 score

### Known differences from the original work
- No access to the original **expert-labeled test set**
- Labels are based on the publicly available NIH dataset
- Some implementation details are inferred due to lack of official code

---

## Repository Structure
```
DacNet/
├── scripts/
│   ├── replicate_chexnet.py
│   ├── Dacnet.py
│   ├── vit_transformer.py
├── XRay_app/
├── reproducibility/
│   └── test_run.sh
├── project_EDA.ipynb
├── requirements.txt
```
---

## Dataset

This project uses the **NIH ChestX-ray14 dataset**.

- ~112,000 frontal chest X-rays  
- 30,805 patients  
- 14 disease labels  

Download from:
https://www.kaggle.com/datasets/nih-chest-xrays/data

---

## Dataset Setup

Your directory must match what the scripts expect.

Example structure:

```
data/
├── Data_Entry_2017.csv
├── images_001/
├── images_002/
...
├── images_012/
```

If your dataset is stored elsewhere, update the path in the scripts:

```python
CONFIG["data_dir"] = "/path/to/your/data"
```

## Lung ROI Masks

`scripts/lung_roi.py` provides a custom U-Net training workflow for JSRT and a
separate preprocessing command to save one post-processed lung mask per NIH
image. No pretrained weights are bundled or downloaded automatically.

Prepare paired JSRT data with images and masks in separate directories. Masks
may be a single binary image with the same filename stem as its X-ray, or
separate left/right masks named `<image-stem>_L` and `<image-stem>_R` (either
side may be absent). Foreground mask pixels must be nonzero. Then train:

```bash
python scripts/lung_roi.py train --images_dir /path/to/JSRT/images --masks_dir /path/to/JSRT/masks --output models/lung_unet.pth
```

Precompute NIH masks using that checkpoint:

```bash
python scripts/lung_roi.py generate --data_dir /path/to/NIH_data --checkpoint models/lung_unet.pth
```

Masks are saved as 8-bit PNGs under `<data_dir>/lung_masks`, preserving each
image's relative directory and original dimensions. The default post-processing
keeps up to two largest connected components, fills their convex hulls, and
dilates the result by five pixels. Use `--output_dir`, `--threshold`, and
`--dilation_radius` to adjust generation. ROI extraction is currently a
standalone preprocessing step; the existing classification training scripts
do not load masks automatically.

## APEX-Net APAM Classifier

`models/apam.py` provides an APAM block and an `APEXNet` DenseNet-121
classifier. The model takes images, lung ROI masks, and disease-specific prior
maps as separate inputs and returns 14 raw logits by default. Resize the maps
from the preprocessing workflows to the image dimensions before batching; the
model resizes them again to the DenseNet feature-map resolution. Example:

```python
from models.apam import APEXNet

model = APEXNet(pretrained=True)
logits = model(images, lung_masks, disease_prior_maps)
```

The input shapes are `[B, 3, H, W]` for images, `[B, 1, H, W]` for lung
masks, and `[B, 14, H, W]` for disease prior maps, ordered like the NIH
findings list. The APEX-Net training script loads the generated lung mask matching each
image's relative path and the 14 shared prior maps. It applies the same random
crop and horizontal flip to the image, ROI mask, and prior maps, while applying
ColorJitter to the image only.

Generate the precomputed inputs (skip generation if they already exist):

```powershell
python scripts\lung_roi.py generate --data_dir C:\NIH_data --checkpoint models\lung_unet.pth
python scripts\generate_prior_maps.py --data_dir C:\NIH_data
```

Then train from the `DacNet` project directory:

```powershell
python scripts\train_apexnet.py --data_dir C:\NIH_data
```

Run the trained APEX-Net checkpoint on one image:

```powershell
python scripts\predict_apexnet.py --image "C:\path\to\chest-xray.png"
```

The predictor defaults to the latest APEX-Net checkpoint and generates a lung
mask with `models\lung_unet.pth`. To reuse a precomputed mask, pass
`--lung_mask "C:\path\to\lung-mask.png"`. Scores are sigmoid model outputs,
not calibrated diagnostic probabilities or clinical diagnoses.

For the local HTTP API, APEX-Net Grad-CAM overlays, and connecting the
authenticated Next.js frontend, follow [back-front.txt](../back-front.txt).

By default, the trainer checks both `<data_dir>\lung_masks` and the project
`lung_masks\` directory, and selects the location containing the most matching
image masks. Prior maps are read from `prior_maps\`. Training uses
ImageNet-pretrained DenseNet-121 weights, AdamW, and the DACNet-style
focal-loss loop. Use `--lung_mask_dir` and `--prior_map_dir` to explicitly
select other locations. Use `--no_pretrained` if pretrained weights are not
available locally and should not be downloaded.

---

## Reproducible Environment (Docker)

We provide a Docker environment to improve reproducibility.

### Build the container

```bash
docker build -t dacnet-env .
```

### Run the container

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace/DacNet dacnet-env
```

Notes:
- Requires CUDA-compatible GPU  
- ≥8GB VRAM recommended  
- If no GPU, remove `--gpus all`  

---

## Quick Verification

Before running full training, verify the setup:

```bash
bash reproducibility/test_run.sh
```

This checks:
- dataset loading  
- model initialization  
- training loop execution  
If this step fails, verify dataset paths and file structure before proceeding to full training.
---
## Running the Models

Once inside the Docker container, you can execute the training scripts.
 
### 1. Reproduction Baseline (CheXNet)

```bash
python scripts/replicate_chexnet.py
```

### 2. DACNet (Improved CNN)

```bash
python scripts/Dacnet.py
```

### 3. Vision Transformer

```bash
python scripts/vit_transformer.py
```

If you encounter CUDA memory errors, reduce batch size in the scripts.

---
 
Evaluation results (AUC and F1) will print to the console. If WANDB is configured, it will log automatically; otherwise, the container defaults to offline mode. The best model checkpoint will be saved in a `models/<run_id>` folder.

---


## Models

### replicate_chexnet.py (Baseline)
DenseNet-121 reproduction of CheXNet.
- Multi-label classification (14 diseases)  
- BCEWithLogitsLoss  
- Patient-level split  
- Evaluated using AUC and F1  

### Dacnet.py (DACNet)
Improved CNN designed for class imbalance.
- DenseNet-121 backbone  
- Focal Loss  
- Per-class threshold tuning  
- Improved F1 performance  

### vit_transformer.py (ViT)
Transformer-based baseline.
- ViT-Base architecture  
- Multi-label classification  
- Compared against CNN approaches  

---

**Performance (Test AUC per Disease)**
| Pathology           | original CheXNet | Dacnet.py | vit_transformer.py | replicate_chexnet.py |
|---------------------|------------------|----------|-------------------|--------------------|
| Atelectasis         | 0.8094           | **0.817** | 0.774           | 0.762              |
| Cardiomegaly        | 0.9248           | **0.932** | 0.89            | 0.922              |
| Consolidation       | **0.7901**       | 0.783     | 0.789           | 0.746              |
| Edema               | 0.8878           | **0.896** | 0.876           | 0.864              |
| Effusion            | 0.8638           | **0.905** | 0.857           | 0.883              |
| Emphysema           | 0.9371           | **0.963** | 0.828           | 0.85               |
| Fibrosis            | 0.8047           | **0.814** | 0.772           | 0.766              |
| Hernia              | 0.9164           | **0.997** | 0.872           | 0.925              |
| Infiltration        | **0.7345**       | 0.708     | 0.7             | 0.673              |
| Mass                | 0.8676           | **0.919** | 0.783           | 0.824              |
| Nodule              | 0.7802           | **0.789** | 0.673           | 0.646              |
| Pleural Thickening  | **0.8062**       | 0.801     | 0.766           | 0.756              |
| Pneumonia           | **0.768**        | 0.74      | 0.713           | 0.656              |
| Pneumothorax        | **0.8887**       | 0.875     | 0.821           | 0.827              |

---
### Average metrics across all diseases for each model
| Metric  | DacNet | ViT Transformer | Replicate CheXNet |
|---------|----------|------------------|--------------------|
| Loss    | **0.0416** | 0.1589           | 0.1661             |
| AUC     | **0.8527** | 0.7940           | 0.7928             |
| F1      | **0.3861** | 0.1114           | 0.0763             |
---
### F1 Score Comparison for Each Model

| Disease             | DacNet | ViT Transformer  | Replicate CheXNet |
|---------------------|----------|------------------|--------------------|
| **AVERAGE**         | **0.386** | 0.111           | 0.076              |
| Atelectasis         | **0.421** | 0.127           | 0.026              |
| Cardiomegaly        | **0.532** | 0.264           | 0.423              |
| Consolidation       | **0.226** | 0               | 0                  |
| Edema               | **0.286** | 0.004           | 0                  |
| Effusion            | **0.623** | 0.427           | 0.459              |
| Emphysema           | **0.516** | 0.079           | 0                  |
| Fibrosis            | **0.127** | 0               | 0                  |
| Hernia              | **0.750** | 0               | 0                  |
| Infiltration        | **0.395** | 0.193           | 0.061              |
| Mass                | **0.477** | 0.213           | 0.079              |
| Nodule              | **0.352** | 0.041           | 0                  |
| Pleural Thickening  | **0.258** | 0               | 0                  |
| Pneumonia           | **0.082** | 0               | 0                  |
| Pneumothorax        | **0.360** | 0.211           | 0.021              |


---
## Test-Images
Folder that contains labeled chest X-ray PNG files for the user to easily download and test on the Hugging Face Streamlit app.

---
## project_EDA.ipynb

Jupyter Notebook that conducts Exploratory Data Analysis such as how many different diseases are present in the dataset and the proportion of each disease in the dataset.
---

## Methodology
- **Data Preprocessing:** Resize, normalize, and augment X-ray images
- **Model Selection:** CNN and Transformer variants
- **Training & Validation:** Patient-level splits, loss monitoring
- **Evaluation:** Per-class AUC-ROC and F1 metrics
- **Comparison:** Benchmarks against original CheXNet results

---

## Demo

Try the model here:  
https://huggingface.co/spaces/cfgpp/DACNet

Features:
- image upload  
- multi-label prediction  
- Grad-CAM visualization  

---

## Limitations

- Results rely on NIH labels (not expert annotations)  
- Some implementation details are inferred  
- Performance may vary across environments  
- Dataset path must be manually configured  

---

## Citation

If you use this repository:

An Open-Source Reproduction and Enhancement of CheXNet for Chest X-ray Disease Classification  
https://arxiv.org/abs/2505.06646

---

## Acknowledgments

Rajpurkar et al.  
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning  
https://arxiv.org/abs/1711.05225

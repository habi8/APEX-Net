import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image


DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
SYMMETRIC_DISEASES = {
    "Atelectasis", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax",
}
BOX_LABELS = {"Infiltrate": "Infiltration"}
ANNOTATED_DISEASES = {
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
}
BOX_REFERENCE_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate disease-specific anatomical prior maps from NIH bounding boxes"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="NIH data directory containing BBox_List_2017.csv and images_001 ... images_012",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for .npy maps (default: <data_dir>/prior_maps)",
    )
    return parser.parse_args()


def find_images(data_dir):
    image_paths = {}
    for index in range(1, 13):
        folder = os.path.join(data_dir, f"images_{index:03d}", "images")
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith(".png"):
                    image_paths[filename] = os.path.join(folder, filename)
    return image_paths


def make_mask(boxes, width, height):
    mask = np.zeros((height, width), dtype=np.float32)
    scale_x = width / BOX_REFERENCE_SIZE
    scale_y = height / BOX_REFERENCE_SIZE

    for x, y, box_width, box_height in boxes:
        x0 = int(np.floor(x * scale_x))
        y0 = int(np.floor(y * scale_y))
        x1 = int(np.ceil((x + box_width) * scale_x))
        y1 = int(np.ceil((y + box_height) * scale_y))
        x0, x1 = np.clip([x0, x1], 0, width)
        y0, y1 = np.clip([y0, y1], 0, height)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1.0

    return mask


def generate_maps(data_dir, output_dir):
    bbox_path = os.path.join(data_dir, "BBox_List_2017.csv")
    if not os.path.isfile(bbox_path):
        raise FileNotFoundError(f"Bounding-box CSV not found: {bbox_path}")

    annotations = pd.read_csv(bbox_path)
    if annotations.shape[1] < 6:
        raise ValueError(
            "Expected BBox_List_2017.csv columns: image, label, x, y, width, height"
        )

    image_paths = find_images(data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No NIH PNG images found under: {data_dir}")

    labels = annotations.iloc[:, 1].astype(str).str.strip().replace(BOX_LABELS)
    unknown_labels = sorted(set(labels) - ANNOTATED_DISEASES)
    if unknown_labels:
        raise ValueError(f"Unexpected bounding-box labels: {unknown_labels}")

    coordinates = annotations.iloc[:, 2:6].apply(pd.to_numeric, errors="coerce")
    if coordinates.isna().any().any():
        raise ValueError("Bounding-box rows contain missing or non-numeric coordinates")

    first_image = annotations.iloc[0, 0]
    if first_image not in image_paths:
        raise FileNotFoundError(f"Annotated image not found in image folders: {first_image}")
    with Image.open(image_paths[first_image]) as image:
        width, height = image.size

    counts = {
        disease: np.zeros((height, width), dtype=np.float32)
        for disease in ANNOTATED_DISEASES
    }
    image_disease_boxes = {}
    for row_index, row in annotations.iterrows():
        image_name = row.iloc[0]
        if image_name not in image_paths:
            raise FileNotFoundError(f"Annotated image not found in image folders: {image_name}")
        disease = labels.iloc[row_index]
        image_disease_boxes.setdefault((image_name, disease), []).append(
            coordinates.iloc[row_index].to_numpy(dtype=float)
        )

    for (image_name, disease), boxes in image_disease_boxes.items():
        with Image.open(image_paths[image_name]) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"Image {image_name} has size {image.size}; expected {(width, height)}"
                )
        mask = make_mask(boxes, width, height)
        counts[disease] += mask
        if disease in SYMMETRIC_DISEASES:
            counts[disease] += np.fliplr(mask)

    os.makedirs(output_dir, exist_ok=True)
    for disease in DISEASES:
        if disease in ANNOTATED_DISEASES:
            disease_map = counts[disease]
            maximum = disease_map.max()
            if maximum == 0:
                raise ValueError(f"No valid bounding boxes found for {disease}")
            disease_map = disease_map / maximum
        else:
            disease_map = np.ones((height, width), dtype=np.float32)

        output_path = os.path.join(output_dir, f"{disease}.npy")
        np.save(output_path, disease_map.astype(np.float32, copy=False))

    print(f"Saved {len(DISEASES)} prior maps ({width}x{height}) to {output_dir}")


def main():
    args = parse_args()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(project_dir, "prior_maps")
    generate_maps(args.data_dir, output_dir)


if __name__ == "__main__":
    main()
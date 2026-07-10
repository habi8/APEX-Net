import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import densenet121

disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model(checkpoint_path, device):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(image_path, checkpoint_path, device="cpu"):
    model = load_model(checkpoint_path, device)

    image = Image.open(image_path).convert("RGB")
    tensor = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().tolist()

    results = sorted(zip(disease_list, probs), key=lambda x: x[1], reverse=True)

    print(f"\nPredictions for: {image_path}\n")
    for disease, prob in results:
        bar = "#" * int(prob * 40)
        print(f"{disease:<20} {prob*100:5.1f}%  {bar}")

    return dict(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single chest X-ray image")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mps")
    args = parser.parse_args()

    predict(args.image, args.checkpoint, args.device)
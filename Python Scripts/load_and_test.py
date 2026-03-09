import sns
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import SqueezeNet1_1_Weights

from cotraining import BlumMitchellCoTraining, RGBWithFFTDataset

# Import your classes directly if they’re in another file, e.g.:
# from main_training_script import RGBWithFFTDataset, BlumMitchellCoTraining

# Assuming both classes are in the same file directory as this script,
# you can paste them or import as shown above.


# ==================== Parameters ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_path = "/small_labeled_ultrasound_dataset/test"
batch_size = 16
input_size = (227, 227)
model_rgb_path = "blum_mitchell_rgbV3.pth"
model_fft_path = "blum_mitchell_fftV3.pth"

# ==================== Transforms ====================
rgb_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])

fft_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),
    transforms.Normalize([0.5], [0.2])
])

# ==================== Dataset & Loader ====================
test_dataset = RGBWithFFTDataset(test_path, rgb_transform, fft_transform, labeled=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(test_dataset.classes)

print(f"Loaded test dataset with {len(test_dataset)} samples and {num_classes} classes.")


# ==================== Load Models ====================
# RGB model
model_rgb = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
model_rgb.classifier[1] = nn.Conv2d(model_rgb.classifier[1].in_channels, num_classes, kernel_size=1)
model_rgb.load_state_dict(torch.load(model_rgb_path, map_location=device))
model_rgb = model_rgb.to(device)

# FFT model
model_fft = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
new_layer = nn.Conv2d(1, 64, kernel_size=3, stride=2)
pretrained_weights = model_fft.features[0].weight.data
new_layer.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
model_fft.features[0] = new_layer
model_fft.classifier[1] = nn.Conv2d(model_fft.classifier[1].in_channels, num_classes, kernel_size=1)
model_fft.load_state_dict(torch.load(model_fft_path, map_location=device))
model_fft = model_fft.to(device)

print("Models loaded successfully!")


# ==================== Evaluation using your class ====================
cotrainer = BlumMitchellCoTraining(model_rgb, model_fft, num_classes, device)
rgb_acc, fft_acc, combined_acc = cotrainer.evaluate(test_loader)

print("\n=== Test Results ===")
print(f"RGB Model Accuracy:      {rgb_acc:.4f}")
print(f"FFT Model Accuracy:      {fft_acc:.4f}")
print(f"Combined Model Accuracy: {combined_acc:.4f}")


# ==================== Confusion Matrix for Combined Model ====================
# We re-run evaluation manually to collect labels and predictions
model_rgb.eval()
model_fft.eval()
all_labels, all_preds = [], []

with torch.no_grad():
    for rgb_inputs, fft_inputs, labels in test_loader:
        rgb_inputs, fft_inputs, labels = rgb_inputs.to(device), fft_inputs.to(device), labels.to(device)

        rgb_outputs = model_rgb(rgb_inputs)
        fft_outputs = model_fft(fft_inputs)
        combined = (torch.softmax(rgb_outputs, dim=1) + torch.softmax(fft_outputs, dim=1)) / 2

        _, preds = torch.max(combined, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.title("Confusion Matrix - Combined Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
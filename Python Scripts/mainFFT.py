import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet1_1_Weights
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# --- Evaluation Helper Function ---
def evaluate_model_on_loader(model, loader, device):
    """Evaluates the model on a given DataLoader and returns accuracy, confusion matrix, labels, and predictions."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm, all_labels, all_preds


# -----------------------------------

class FFTTransform(torch.nn.Module):
    def __init__(self, use_magnitude=True, use_phase=False):
        super().__init__()
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase

    def __call__(self, tensor):
        # Ensure tensor is in the right format (C, H, W)
        if tensor.dim() == 4:  # Batch dimension
            batch_fft = []
            for k in range(tensor.size(0)):
                batch_fft.append(self._apply_fft(tensor[k]))
            return torch.stack(batch_fft)
        else:
            return self._apply_fft(tensor)

    def _apply_fft(self, tensor):
        # Apply FFT to each channel (3 channels for RGB input, magnitude output is 3 channels)
        fft_channels = []
        for c in range(tensor.size(0)):
            # Apply 2D FFT
            fft_result = torch.fft.fft2(tensor[c].float())

            channels_to_add = []
            if self.use_magnitude:
                magnitude = torch.abs(fft_result)
                # Apply log transform to compress dynamic range
                magnitude = torch.log(magnitude + 1e-8)
                channels_to_add.append(magnitude)

            if self.use_phase:
                phase = torch.angle(fft_result)
                channels_to_add.append(phase)

            fft_channels.extend(channels_to_add)

        return torch.stack(fft_channels)


def rgb_loader(path):
    """Load RGB image from various formats and convert to PIL Image"""
    if path.endswith('.npy'):
        # Load numpy array
        data = np.load(path, allow_pickle=True)
        data = data.astype(np.float32)

        if data.ndim == 2:
            data = np.stack([data, data, data], axis=-1)
        elif data.ndim == 3 and data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)

        if data.ndim == 3 and data.shape[-1] == 3:
            if data.max() <= 1.0:
                data = (data * 255).astype(np.uint8)
            else:
                data = np.clip(data, 0, 255).astype(np.uint8)
            return Image.fromarray(data, 'RGB')
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


# --- MODEL INITIALIZATION FUNCTION ---
def initialize_squeezenet_fft(num_classes, device, num_input_channels=3):
    """
    Initializes the SqueezeNet1_1 model and modifies the classifier for new class count.
    """
    # Load pre-trained SqueezeNet1_1
    model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

    # Modify the classifier for the new dataset
    model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes

    model = model.to(device)
    print(f"Model: SqueezeNet1_1 adapted for {num_classes} classes.")
    return model


# ---------------------------------------------


# PARAMETERS

print('Program starting...')
# SqueezeNet typically uses 224x224 input size
input_size = (224, 224)
batch_size = 30  # Mini-batch size
num_epochs = 60  # Number of epochs
learning_rate = 1e-4  # Learning rate

# --- ALL DATASET PATHS ---
save_model_paths = [
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredOrganized.pth",
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredLarge.pth",
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredSmall.pth"
]

train_paths = [
    # "D:/Facultate/Disertatie/mainProject/pythonProject1/organized_ultrasound_dataset/labeled_train",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/large_labeled_ultrasound_dataset/labeled_train",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/small_labeled_ultrasound_dataset/labeled_train"
]

test_paths = [
    # "D:/Facultate/Disertatie/mainProject/pythonProject1/organized_ultrasound_dataset/test",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/large_labeled_ultrasound_dataset/test",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/small_labeled_ultrasound_dataset/test"
]

# We will look for a 'validation' folder next to 'labeled_train' for each dataset,
# or use the specific path for the small one as a fallback/check.
VALIDATION_PATH_SMALL = "/small_labeled_ultrasound_dataset/validation"

exp_labels = ["20% labeled"]

# DATASET AND TRANSFORMS
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        FFTTransform(use_magnitude=True, use_phase=False),  # Output 3 channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        FFTTransform(use_magnitude=True, use_phase=False),  # Output 3 channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
}

# ====================================================================================
# --- MAIN EXPERIMENT LOOP (Iterates over all three datasets) ---
# ====================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training device set to {device}...')

for i in range(len(train_paths)):
    print(f"\n=========================================================")
    print(f"=== Starting Experiment {i + 1}: {exp_labels[i]} ===")
    print(f"=========================================================")

    train_data_path = train_paths[i]
    test_data_path = test_paths[i]
    save_model_path = save_model_paths[i]
    exp_label = exp_labels[i]

    # 1. LOAD TRAIN DATA
    full_train_dataset = datasets.DatasetFolder(train_data_path, loader=rgb_loader,
                                                extensions=('.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                                                transform=data_transforms['train'])

    # 2. DETERMINE VALIDATION DATA PATH
    if "small" in train_data_path.lower() and os.path.exists(VALIDATION_PATH_SMALL):
        val_data_path = VALIDATION_PATH_SMALL
    else:
        # Check for a 'validation' subfolder adjacent to the 'labeled_train' folder
        val_data_path = os.path.join(os.path.dirname(train_data_path), "validation")

    # 3. LOAD VALIDATION DATA (with split fallback)
    if os.path.exists(val_data_path):
        val_dataset = datasets.DatasetFolder(val_data_path, loader=rgb_loader,
                                             extensions=('.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                                             transform=data_transforms['val'])
        train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"Validation method: Loaded dedicated set from {val_data_path}")
    else:
        # Fallback to train/val split (70/30) if dedicated folder is not found
        print("Warning: Dedicated validation folder not found. Splitting training data (70/30) for validation.")

        train_indices, val_indices = train_test_split(
            range(len(full_train_dataset)),
            test_size=0.3,
            stratify=full_train_dataset.targets,
            random_state=42,
        )
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get the number of classes
    num_classes = len(full_train_dataset.classes)

    # 4. LOAD TEST DATA
    test_dataset = datasets.DatasetFolder(test_data_path, loader=rgb_loader,
                                          extensions=('.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                                          transform=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Database loaded...')

    # 5. LOAD SQUEEZENET MODEL
    model = initialize_squeezenet_fft(num_classes, device, num_input_channels=3)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 6. TRAINING LOOP
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # PER-EPOCH VALIDATION CHECK
        train_loss = running_loss / len(train_loader)

        accuracy_val_epoch, _, _, _ = evaluate_model_on_loader(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {accuracy_val_epoch:.4f}")

    # Save the model
    torch.save(model.state_dict(), save_model_path)
    print(f"\nModel saved to {save_model_path}")

    # 7. FINAL EVALUATION

    # Validation
    print("Final evaluation on validation data...")
    accuracy_val, cm_val, val_labels, val_preds = evaluate_model_on_loader(model, val_loader, device)
    print(f"Final Validation Accuracy: {accuracy_val:.4f}")

    # Testing
    print("Evaluating on test data...")
    accuracy_test, cm_test, test_labels, test_preds = evaluate_model_on_loader(model, test_loader, device)
    print(f"Test Accuracy: {accuracy_test:.4f}")

    # 8. SAVE RESULTS TO EXCEL
    excel_path = f"../experiment_results_fft_squeezenet.xlsx"
    sheet_name = f"Exp_{i + 1}_{exp_label}_SqueezeNet"

    # Load or create workbook
    try:
        workbook = pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace')
    except FileNotFoundError:
        workbook = pd.ExcelWriter(excel_path, engine='openpyxl')

    # Build DataFrames
    df_summary = pd.DataFrame({
        "Metric": ["Validation Accuracy", "Test Accuracy"],
        "Value": [accuracy_val, accuracy_test]
    })
    df_cm_val = pd.DataFrame(cm_val, index=val_dataset.classes, columns=val_dataset.classes)
    df_cm_val.index.name = "True"
    df_cm_val.columns.name = "Pred"

    df_cm_test = pd.DataFrame(cm_test, index=test_dataset.classes, columns=test_dataset.classes)
    df_cm_test.index.name = "True"
    df_cm_test.columns.name = "Pred"

    # Write sheets
    df_summary.to_excel(workbook, sheet_name=sheet_name, startrow=0, index=False)
    # Start CMs on separate rows
    workbook.sheets[sheet_name].cell(row=6, column=1).value = "Validation Confusion Matrix"
    df_cm_val.to_excel(workbook, sheet_name=sheet_name, startrow=7)

    workbook.sheets[sheet_name].cell(row=7 + len(df_cm_val) + 2, column=1).value = "Test Confusion Matrix"
    df_cm_test.to_excel(workbook, sheet_name=sheet_name, startrow=7 + len(df_cm_val) + 3)

    # Save file
    workbook.close()

    print(f"Saved results for {exp_label} to {excel_path}")
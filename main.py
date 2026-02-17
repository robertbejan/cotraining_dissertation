import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.models import SqueezeNet1_1_Weights
import pandas as pd
import numpy as np


# FUNCTION: create matrix formatted as text block
def format_confusion(cm):
    rows = []
    for r in cm:
        row = " ".join(f"{v:4d}" for v in r)
        rows.append(row)
    return "\n" + "\n".join(rows) + "\n"


# PLACE TO STORE RESULTS FOR EXCEL
results = []

print('Program starting...')
input_size = (227, 227)
batch_size = 30
num_epochs = 60
learning_rate = 1e-4

# Paths
save_model_paths = [
    # "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredOrganized.pth",
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredLarge.pth",
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredSmall.pth"
]

train_paths = [
    # "D:/Facultate/Disertatie/mainProject/pythonProject1/organized_labeled_ultrasound_dataset/labeled_train",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/large_labeled_ultrasound_dataset/labeled_train",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/small_labeled_ultrasound_dataset/labeled_train"
]

test_paths = [
    # "D:/Facultate/Disertatie/mainProject/pythonProject1/organized_labeled_ultrasound_dataset/test",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/large_labeled_ultrasound_dataset/test",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/small_labeled_ultrasound_dataset/test"
]

exp_labels = ["80% labeled", "20% labeled"]


for i in range(3):

    print(f"\n=== Experiment {i+1} ({exp_labels[i]}) ===")

    train_data_path = train_paths[i]
    test_data_path = test_paths[i]
    save_model_path = save_model_paths[i]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        ]),
    }

    # Load the dataset
    image_dataset = datasets.ImageFolder(train_data_path, transform=data_transforms['train'])

    train_indices, val_indices = train_test_split(
        range(len(image_dataset)),
        test_size=0.3,
        stratify=image_dataset.targets,
        random_state=42,
    )
    train_dataset = torch.utils.data.Subset(image_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(image_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(image_dataset.classes)

    test_dataset = datasets.ImageFolder(test_data_path, transform=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Database loaded...')

    # Load pretrained model
    model = models.squeezenet1_1(weights=("pretrained", SqueezeNet1_1_Weights.IMAGENET1K_V1))
    model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1,1))
    model.num_classes = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print('Using device:', device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # TRAINING
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

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), save_model_path)
    print("Model saved:", save_model_path)

    # VALIDATION
    print("Evaluating validation...")
    model.eval()
    val_labels, val_preds = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    acc_val = accuracy_score(val_labels, val_preds)
    cm_val = confusion_matrix(val_labels, val_preds)

    print("Validation accuracy:", acc_val)

    # TEST
    print("Evaluating test...")
    test_labels, test_preds = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    acc_test = accuracy_score(test_labels, test_preds)
    cm_test = confusion_matrix(test_labels, test_preds)

    print("Test accuracy:", acc_test)

    # STORE INTO RESULTS
    results.append({
        "experiment": exp_labels[i],
        "validation_accuracy": acc_val,
        "test_accuracy": acc_test,
        "validation_confusion_matrix": format_confusion(cm_val),
        "test_confusion_matrix": format_confusion(cm_test),
        "classes": str(image_dataset.classes)
    })


# SAVE TO EXCEL
df = pd.DataFrame(results)
df.to_excel("cnn_results.xlsx", index=False)
print("\n=== RESULTS SAVED TO cnn_results.xlsx ===")

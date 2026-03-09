import os
import shutil
from sklearn.model_selection import train_test_split


def create_class_mapping():
    """Define how to map original classes to new classes"""
    return {
        'Fetal abdomen': 'abdomen',
        'Fetal brain-Other': 'brain',
        'Fetal brain-Trans-cerebellum': 'brain',
        'Fetal brain-Trans-thalamic': 'brain',
        'Fetal brain-Trans-ventricular': 'brain',
        'Fetal femur': 'femur',
        'Fetal thorax': 'thorax',
        'Maternal cervix': 'maternal_cervix',
        'Other': 'other'
    }


def reorganize_dataset(source_dir, target_dir, test_ratio=0.2):
    """
    Reorganize dataset with fewer classes and split into train/test only

    :param source_dir: Directory containing original class folders
    :param target_dir: Directory where reorganized dataset will be saved
    :param test_ratio: Proportion of data for test set (default 20%)
    """
    class_map = create_class_mapping()
    os.makedirs(target_dir, exist_ok=True)

    # Create target directories
    for new_class in set(class_map.values()):
        os.makedirs(os.path.join(target_dir, 'train', new_class), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'test', new_class), exist_ok=True)

    # Process each original class
    for orig_class in os.listdir(source_dir):
        if not os.path.isdir(os.path.join(source_dir, orig_class)):
            continue

        new_class = class_map.get(orig_class, 'other')
        print(f"Mapping {orig_class} => {new_class}")

        # Get all image files
        image_files = [
            f for f in os.listdir(os.path.join(source_dir, orig_class))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Split into train/test only
        train_files, test_files = train_test_split(
            image_files, test_size=test_ratio, random_state=42
        )

        # Copy files to new structure
        for file in train_files:
            src = os.path.join(source_dir, orig_class, file)
            dst = os.path.join(target_dir, 'train', new_class, file)
            shutil.copy2(src, dst)

        for file in test_files:
            src = os.path.join(source_dir, orig_class, file)
            dst = os.path.join(target_dir, 'test', new_class, file)
            shutil.copy2(src, dst)


# Configuration
source_train_dir = "D:/Facultate/Disertatie/mainProject/pythonProject1/trainingdata/trainAUX"
source_test_dir = "D:/Facultate/Disertatie/mainProject/pythonProject1/trainingdata/testAUX"
target_dir = "D:/Facultate/Disertatie/mainProject/pythonProject1/consolidated_dataset_simple"

# First process trainAUX (will create train/test splits)
print("Processing trainAUX...")
reorganize_dataset(source_train_dir, target_dir)

# Then process testAUX (all go to test folder)
print("\nProcessing testAUX...")
class_map = create_class_mapping()
os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)

for orig_class in os.listdir(source_test_dir):
    if not os.path.isdir(os.path.join(source_test_dir, orig_class)):
        continue

    new_class = class_map.get(orig_class, 'other')
    print(f"Mapping {orig_class} => {new_class}")

    # Create new class directory if needed
    os.makedirs(os.path.join(target_dir, 'test', new_class), exist_ok=True)

    # Copy all test files
    for file in os.listdir(os.path.join(source_test_dir, orig_class)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(source_test_dir, orig_class, file)
            dst = os.path.join(target_dir, 'test', new_class, file)
            shutil.copy2(src, dst)

print("\nDataset reorganization complete!")
print(f"New dataset structure saved to: {target_dir}")
print("New classes:", sorted(set(create_class_mapping().values())))
print("\nDirectory structure:")
print(f"{target_dir}")
print("├── train/")
print("│   ├── abdomen/")
print("│   ├── brain/")
print("│   ├── femur/")
print("│   ├── thorax/")
print("│   ├── maternal_cervix/")
print("│   └── other/")
print("└── test/")
print("    ├── abdomen/")
print("    ├── brain/")
print("    ├── femur/")
print("    ├── thorax/")
print("    ├── maternal_cervix/")
print("    └── other/")
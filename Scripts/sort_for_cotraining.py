import os
import shutil
import random
from sklearn.model_selection import train_test_split


def create_dataset_structure(base_dir, classes):
    """Create directory structure for the organized dataset"""
    dirs = [
        'labeled_train',
        'unlabeled_train',
        'validation',
        'test'
    ]

    for dir_name in dirs:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
        if dir_name != 'unlabeled_train':
            for class_name in classes:
                os.makedirs(os.path.join(base_dir, dir_name, class_name), exist_ok=True)
        else:
            os.makedirs(os.path.join(base_dir, dir_name, 'unlabeled'), exist_ok=True)


# NEW: Hybrid Sampling Function
def balance_data_by_hybrid_sampling(file_list, min_limit=100, max_limit=300):
    """
    Balances the file list by oversampling classes below min_limit
    and undersampling classes above max_limit.
    """
    class_groups = {}
    for path, class_name in file_list:
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append((path, class_name))

    print(f"Balancing Labeled Train set to be between {min_limit} and {max_limit} samples per class.")

    balanced_list = []

    for class_name, group in class_groups.items():
        current_count = len(group)
        print(f"  - Class '{class_name}' has {current_count} samples.", end='')

        if current_count < min_limit:
            # Oversample: Repeat the group until it reaches min_limit
            target_count = min_limit
            additional_samples_needed = target_count - current_count

            random.shuffle(group)
            oversampled_data = random.choices(group, k=additional_samples_needed)

            balanced_list.extend(group)
            balanced_list.extend(oversampled_data)
            print(f" -> Oversampling to {target_count} samples.")

        elif current_count > max_limit:
            # Undersample: Randomly select max_limit samples
            random.shuffle(group)
            balanced_list.extend(group[:max_limit])
            print(f" -> Undersampling to {max_limit} samples.")

        else:
            # Keep: Class is within the desired range
            balanced_list.extend(group)
            print(f" -> Keeping {current_count} samples (within range).")

    return balanced_list


def organize_ultrasound_dataset(source_dirs, dest_dir, test_size=0.2, val_size=0.15, unlabeled_ratio=0.5,
                                hybrid_balance_labeled_train=False, min_samples=100, max_samples=300):
    """
    Organize ultrasound images into labeled train, unlabeled train, validation, and test sets
    """
    # Collect class names (unchanged)
    class_names = set()
    for source_dir in source_dirs:
        for item in os.listdir(source_dir):
            if os.path.isdir(os.path.join(source_dir, item)):
                class_names.add(item)
    class_names = sorted(class_names)
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

    # Create directory structure and collect image paths (unchanged)
    create_dataset_structure(dest_dir, class_names)
    image_paths = []
    for source_dir in source_dirs:
        for class_name in class_names:
            class_dir = os.path.join(source_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append((os.path.join(class_dir, img_name), class_name))
    print(f"Found {len(image_paths)} total images across all classes")

    # Splitting into Test, Train, Labeled, Unlabeled, Validation (stratified)
    train_paths, test_paths = train_test_split(image_paths, test_size=test_size, stratify=[x[1] for x in image_paths],
                                               random_state=42)
    labeled_paths, unlabeled_paths = train_test_split(train_paths, test_size=unlabeled_ratio,
                                                      stratify=[x[1] for x in train_paths], random_state=42)
    train_labeled_paths, val_paths = train_test_split(labeled_paths, test_size=val_size,
                                                      stratify=[x[1] for x in labeled_paths], random_state=42)

    # NEW: Apply Hybrid Balancing
    original_labeled_train_count = len(train_labeled_paths)
    if hybrid_balance_labeled_train:
        train_labeled_paths = balance_data_by_hybrid_sampling(
            train_labeled_paths,
            min_limit=min_samples,
            max_limit=max_samples
        )
        print(f"Labeled Train set adjusted from {original_labeled_train_count} to {len(train_labeled_paths)} images.")

    # Function to copy files (unchanged)
    def copy_files(file_list, dest_subdir, keep_labels=True):
        for src_path, class_name in file_list:
            if keep_labels:
                dest_path = os.path.join(dest_dir, dest_subdir, class_name, os.path.basename(src_path))
            else:
                dest_path = os.path.join(dest_dir, dest_subdir, 'unlabeled', os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)

    # Copy files (unchanged)
    print("\nOrganizing dataset...")
    copy_files(train_labeled_paths, 'labeled_train')
    copy_files(unlabeled_paths, 'unlabeled_train', keep_labels=False)
    copy_files(val_paths, 'validation')
    copy_files(test_paths, 'test')

    print("\nDataset organization complete!")
    print(f"Labeled Train: {len(train_labeled_paths)} images")
    # ... (rest of print statements unchanged) ...


# Configuration
source_dirs = [
    r"D:\Facultate\Disertatie\mainProject\pythonProject1\consolidated_dataset_simple\train",
    r"D:\Facultate\Disertatie\mainProject\pythonProject1\consolidated_dataset_simple\test"
]

destination_dir = r"/organized_labeled_ultrasound_dataset"

# Parameters
test_size = 0.2
val_size = 0.15
unlabeled_ratio = 0.5

# Target balancing range
MIN_SAMPLES = 0
MAX_SAMPLES = 100000

# Run the organization
if __name__ == "__main__":
    print("Starting ultrasound dataset organization...")
    organize_ultrasound_dataset(
        source_dirs,
        destination_dir,
        test_size,
        val_size,
        unlabeled_ratio,
        hybrid_balance_labeled_train=True,  # Set to True to enable hybrid balancing
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES
    )
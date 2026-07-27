import torch

from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import torch.fft
import hashlib


def compute_sample_hash(grayscale_tensor, fft_tensor, dino_tensor, label):
    """
    Compute a hash value for a sample to detect duplicates
    """
    grayscale_hash = hashlib.md5(grayscale_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
    fft_hash = hashlib.md5(fft_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
    dino_hash = hashlib.md5(dino_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
    return f"{grayscale_hash}_{fft_hash}_{dino_hash}_{label}"


class RGBWithFFTDataset(Dataset):
    """
    Dataset that creates grayscale, FFT, and DINOv2 inputs from source images.
    """
    def __init__(self, root_dir, grayscale_transform=None, fft_transform=None, dino_transform=None, labeled=True):
        self.root_dir = root_dir
        self.grayscale_transform = grayscale_transform
        self.fft_transform = fft_transform
        self.dino_transform = dino_transform
        self.labeled = labeled
        self.classes = sorted(os.listdir(root_dir)) if labeled and os.path.isdir(root_dir) else None
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} if self.classes else None
        self.samples = []
        self.pseudo_samples = []  # Store pseudo-labeled samples separately
        self.pseudo_sample_hashes = set()  # Track pseudo-sample hashes to prevent duplicates
        self.unlabeled_used = set()  # Track which unlabeled samples have been used

        if labeled:
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_dir, file_name),
                            self.class_to_idx[class_name]
                        ))
        else:
            # For unlabeled data
            unlabeled_dir = os.path.join(root_dir, 'unlabeled') if os.path.exists(
                os.path.join(root_dir, 'unlabeled')) else root_dir
            for img_name in os.listdir(unlabeled_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(unlabeled_dir, img_name),
                        -1  # Dummy label for unlabeled
                    ))

    def add_pseudo_samples(self, pseudo_samples):
        """
        Add pseudo-labeled samples to the dataset, avoiding duplicates.
        Each pseudo-sample is a tuple: (grayscale_tensor, fft_tensor, dino_tensor, label)
        """
        added_count = 0
        for grayscale_tensor, fft_tensor, dino_tensor, label in pseudo_samples:
            sample_hash = compute_sample_hash(grayscale_tensor, fft_tensor, dino_tensor, label)
            if sample_hash not in self.pseudo_sample_hashes:
                self.pseudo_sample_hashes.add(sample_hash)
                self.pseudo_samples.append((grayscale_tensor, fft_tensor, dino_tensor, label))
                added_count += 1

        return added_count

    def remove_pseudo_samples(self, pseudo_samples):
        """
        Remove samples based on threshold confidence
        :param pseudo_samples: The samples to be removed
        :return: Number of samples to be removed
        """
        removed_count = 0

        # Build set of hashes to remove
        hashes_to_remove = set()
        for grayscale_tensor, fft_tensor, dino_tensor, label in pseudo_samples:
            sample_hash = compute_sample_hash(grayscale_tensor, fft_tensor, dino_tensor, label)
            if sample_hash in self.pseudo_sample_hashes:
                hashes_to_remove.add(sample_hash)
                removed_count += 1

        # Remove from hash set
        self.pseudo_sample_hashes -= hashes_to_remove

        # Filter pseudo_samples list
        self.pseudo_samples = [
            sample for sample in self.pseudo_samples
            if compute_sample_hash(sample[0], sample[1], sample[2], sample[3]) not in hashes_to_remove
        ]
        return removed_count

    def __len__(self):
        """
        Determines the length of the whole dataset, including labeled/pseudo-labeled samples.
        :return:
        """
        return len(self.samples) + len(self.pseudo_samples)

    def __getitem__(self, idx):
        """
        This method loads and applies transformations to the samples. The spatial images are transformed into Gray scale
        images and FFT domain samples. The latter ones, are transformed using the log of the magnitude and normalized
        accordingly for the SqueezeNet backbone.
        :param idx:
        :return:
        """
        # Check if we're accessing a pseudo-sample or regular sample
        if idx < len(self.samples):
            path, label = self.samples[idx]
            image = Image.open(path).convert('RGB')
        else:
            # This is a pseudo-sample (stored as tensors)
            pseudo_idx = idx - len(self.samples)
            grayscale_tensor, fft_tensor, dino_tensor, label = self.pseudo_samples[pseudo_idx]
            return grayscale_tensor, fft_tensor, dino_tensor, label

        grayscale_image = self.grayscale_transform(image) if self.grayscale_transform else transforms.ToTensor()(image)

        # Compute FFT
        gray_image = transforms.Grayscale()(image)
        gray_tensor = transforms.ToTensor()(gray_image) if not isinstance(gray_image, torch.Tensor) else gray_image

        # Get the complex FFT
        f_complex = torch.fft.fft(gray_tensor.squeeze())

        # Calculate Magnitude and apply Log-Scale
        fft_magnitude = torch.abs(f_complex)
        fft_log = torch.log1p(fft_magnitude)

        # Min-Max Normalization to [0, 1]
        # This ensures the features are in a range SqueezeNet expects
        eps = 1e-8
        fft_min = fft_log.min()
        fft_max = fft_log.max()
        fft_normalized = (fft_log - fft_min) / (fft_max - fft_min + eps)

        # Final Formatting
        fft_vector = fft_normalized.unsqueeze(0).float()

        if self.fft_transform:
            fft_vector = self.fft_transform(fft_vector)

        dino_image = self.dino_transform(image) if self.dino_transform else transforms.ToTensor()(image)

        return grayscale_image, fft_vector, dino_image, label

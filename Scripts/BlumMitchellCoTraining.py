import random

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR


class BlumMitchellCoTraining:
    """Co-training controller for a grayscale CNN and a DINOv2/FFT ensemble."""

    def __init__(self, model_grayscale, model_fft, num_classes, device, checked_number, cotraining_start, k=30,
                 confidence_thresh_fft=0.95, confidence_thresh_grayscale=0.9):
        self.scheduler_grayscale = None
        self.model_grayscale = model_grayscale
        self.model_fft = model_fft
        self.num_classes = num_classes
        self.device = device
        self.checked_number = checked_number
        self.k = k
        self.confidence_thresh_fft = confidence_thresh_fft
        self.confidence_thresh_grayscale = confidence_thresh_grayscale
        self.criterion = nn.CrossEntropyLoss()
        self.cotraining_start = cotraining_start
        self.random_dropout = True
        self.grayscale_dataset = None
        self.fft_dataset = None
        self.unlabeled_dataset = None
        self.used_unlabeled_indices = set()
        self.grayscale_removal_rates = []
        self.fft_removal_rates = []

    def set_datasets(self, grayscale_dataset, fft_dataset, unlabeled_dataset):
        self.grayscale_dataset = grayscale_dataset
        self.fft_dataset = fft_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def init_schedulers(self, optimizer_grayscale, optimizer_fft=None, step_size=5, gamma=0.9):
        """Schedule only the gradient-trained Grayscale model."""
        self.scheduler_grayscale = StepLR(optimizer_grayscale, step_size=step_size, gamma=gamma)
        print(f"Initialized Grayscale StepLR scheduler: step_size={step_size}, gamma={gamma}")

    def train_iteration(self, grayscale_loader, fft_loader, unlabeled_loader, optimizer_grayscale, optimizer_fft,
                        epoch_counter, batch_size, reevaluate_flag):
        # Stage 1 - Train on the labeled dataset
        self.train_on_labeled(grayscale_loader, optimizer_grayscale)
        if epoch_counter > self.cotraining_start:
            # Reevaluation phase with confidence threshold adjustment
            if reevaluate_flag:
                removed_grayscale, removed_fft, checked_samples = self.reevaluate_pseudo_labels()
                self.adjust_confidence_threshold(removed_grayscale, removed_fft, checked_samples)
            # Stage 2 - Label the unnotated data
            grayscale_samples, fft_samples = self.label_unlabeled_data(unlabeled_loader)
            # Add the actual data in the labeled datasets with duplicate avoidance
            if grayscale_samples:
                added = self.grayscale_dataset.add_pseudo_samples(grayscale_samples)
                self.fft_dataset.add_pseudo_samples(fft_samples)
                print(f"Added {added} shared samples to both datasets")
        # Make a LR scheduler step
        if self.scheduler_grayscale is not None:
            self.scheduler_grayscale.step()
            print(f"LR updated. Grayscale LR: {self.scheduler_grayscale.get_last_lr()[0]:.6f}")

    def train_on_labeled(self, grayscale_loader, optimizer_grayscale):
        """Train SqueezeNet, then fit the autoencoder encoder and XGBoost classifier."""
        self.model_grayscale.train()
        losses = []
        for grayscale_images, _, _, labels in grayscale_loader:
            # Make the tensors use the GPU
            labels = labels.to(self.device)
            grayscale_images = grayscale_images.to(self.device)

            # Forward pass
            preds = self.model_grayscale(grayscale_images)

            # Calculate the loss
            loss = self.criterion(preds, labels)

            # Set the optimizer gradients to 0
            optimizer_grayscale.zero_grad()

            # Backpropagation
            loss.backward()

            # Set the gradients to the new values
            optimizer_grayscale.step()

            # Put all losses in a list
            losses.append(loss.item())
        self.model_fft.fit_loader(grayscale_loader)
        print(f"The average Grayscale loss on this epoch is: {sum(losses) / len(losses):.4f}")

    def label_unlabeled_data(self, unlabeled_loader):
        self.model_grayscale.eval()
        self.model_fft.eval()
        candidates = []
        current_idx = 0
        with torch.no_grad():
            for grayscale_inputs, fft_inputs, dino_inputs, _ in unlabeled_loader:
                # Make the tensors use the GPU
                grayscale_inputs = grayscale_inputs.to(self.device)
                # Compute the predictions and probabilities of 
                grayscale_preds = self.model_grayscale(grayscale_inputs)
                grayscale_probs = torch.softmax(grayscale_preds, dim=1)
                # Compute probabilities for the ensemble model
                fft_probs = self.model_fft.predict_proba(dino_inputs, fft_inputs)
                # Get the most trusted annotations
                grayscale_confidences, grayscale_predictions = torch.max(grayscale_probs, dim=1)
                fft_confidences, fft_predictions = torch.max(fft_probs, dim=1)

                for index in range(len(grayscale_inputs)):
                    sample_idx = current_idx + index
                    threshold = (self.confidence_thresh_grayscale + self.confidence_thresh_fft) / 2
                    if (sample_idx not in self.used_unlabeled_indices
                            and grayscale_predictions[index] == fft_predictions[index]
                            and grayscale_confidences[index] > self.confidence_thresh_grayscale
                            and fft_confidences[index] > self.confidence_thresh_fft):
                        candidates.append({
                            "data": (grayscale_inputs[index].cpu(), fft_inputs[index].cpu(),
                                     dino_inputs[index].cpu(), grayscale_predictions[index].item()),
                            "confidence": max(grayscale_confidences[index].item(), fft_confidences[index].item()),
                            "index": sample_idx
                        })
                current_idx += len(grayscale_inputs)
        candidates.sort(key=lambda sample: sample["confidence"], reverse=True)
        samples = []
        for candidate in candidates[:self.k]:
            self.used_unlabeled_indices.add(candidate["index"])
            samples.append(candidate["data"])
        return samples, samples

    def reevaluate_pseudo_labels(self):
        pseudo_samples = self.grayscale_dataset.pseudo_samples
        subset_size = min(self.checked_number, len(pseudo_samples), len(self.fft_dataset.pseudo_samples))
        if subset_size == 0:
            return 0, 0, 0
        indices = random.sample(range(len(pseudo_samples)), subset_size)
        if self.random_dropout:
            to_remove = [pseudo_samples[index] for index in indices]
        else:
            self.model_grayscale.eval()
            self.model_fft.eval()
            to_remove = []
            with torch.no_grad():
                for index in indices:
                    # Exctract the tensors from the pseudo-labeled sample and compute the probs
                    grayscale_tensor, fft_tensor, dino_tensor, pseudo_label = pseudo_samples[index]
                    grayscale_probs = torch.softmax(self.model_grayscale(grayscale_tensor.unsqueeze(0).to(self.device)), dim=1)
                    fft_probs = self.model_fft.predict_proba(dino_tensor.unsqueeze(0), fft_tensor.unsqueeze(0))
                    # Get the max confidence samples
                    grayscale_conf, grayscale_prediction = torch.max(grayscale_probs, dim=1)
                    fft_conf, fft_prediction = torch.max(fft_probs, dim=1)
                    is_invalid = (grayscale_conf < self.confidence_thresh_grayscale or fft_conf < self.confidence_thresh_fft
                                  or grayscale_prediction != fft_prediction or grayscale_prediction != pseudo_label
                                  or fft_prediction != pseudo_label)
                    if is_invalid:
                        to_remove.append(pseudo_samples[index])
        grayscale_removed = self.grayscale_dataset.remove_pseudo_samples(to_remove)
        fft_removed = self.fft_dataset.remove_pseudo_samples(to_remove)
        print(f"Joint reevaluation removed {grayscale_removed} samples from the shared pool.")
        return grayscale_removed, fft_removed, subset_size

    def adjust_confidence_threshold(self, grayscale_removed, fft_removed, batch_size):
        if batch_size <= 0:
            return

        self.grayscale_removal_rates.append(grayscale_removed / batch_size)
        self.fft_removal_rates.append(fft_removed / batch_size)
        self.grayscale_removal_rates = self.grayscale_removal_rates[-3:]
        self.fft_removal_rates = self.fft_removal_rates[-3:]

        grayscale_removal_rate = sum(self.grayscale_removal_rates) / len(self.grayscale_removal_rates)
        fft_removal_rate = sum(self.fft_removal_rates) / len(self.fft_removal_rates)

        self.confidence_thresh_grayscale *= self._threshold_adjustment(grayscale_removal_rate)
        self.confidence_thresh_fft *= self._threshold_adjustment(fft_removal_rate)
        self.confidence_thresh_grayscale = max(self.confidence_thresh_grayscale, 0.75)
        self.confidence_thresh_fft = max(self.confidence_thresh_fft, 0.70)
        print(f"Three-epoch Grayscale removal rate: {grayscale_removal_rate:.2%}")
        print(f"Three-epoch ensemble removal rate: {fft_removal_rate:.2%}")
        print(f"The new confidence threshold for Grayscale model is: {self.confidence_thresh_grayscale}")
        print(f"The new confidence threshold for ensemble model is: {self.confidence_thresh_fft}")

    @staticmethod
    def _threshold_adjustment(removal_rate):
        if removal_rate >= 0.55:
            return 1.04
        if removal_rate >= 0.30:
            return 1.01
        return 0.99

    def evaluate(self, loader):
        self.model_grayscale.eval()
        self.model_fft.eval()
        labels_all, grayscale_all, fft_all, combined_all = [], [], [], []
        with torch.no_grad():
            for grayscale_inputs, fft_inputs, dino_inputs, labels in loader:
                grayscale_inputs = grayscale_inputs.to(self.device)
                grayscale_outputs = self.model_grayscale(grayscale_inputs)
                fft_probs = self.model_fft.predict_proba(dino_inputs, fft_inputs)
                combined_probs = (torch.softmax(grayscale_outputs, dim=1) + fft_probs) / 2
                labels_all.extend(labels.numpy())
                grayscale_all.extend(torch.argmax(grayscale_outputs, dim=1).cpu().numpy())
                fft_all.extend(torch.argmax(fft_probs, dim=1).cpu().numpy())
                combined_all.extend(torch.argmax(combined_probs, dim=1).cpu().numpy())
        return (
            accuracy_score(labels_all, grayscale_all), accuracy_score(labels_all, fft_all),
            accuracy_score(labels_all, combined_all), confusion_matrix(labels_all, grayscale_all),
            confusion_matrix(labels_all, fft_all), confusion_matrix(labels_all, combined_all),
        )
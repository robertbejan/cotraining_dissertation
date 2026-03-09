import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import torch.fft
from torch.optim.lr_scheduler import StepLR
import numpy as np


class BlumMitchellCoTraining:
    """
    This class contains the LR schedulers, models, confidence thresholds, criterions and other information necessary
    to perform a full training of a CoTraining model.
    """

    def __init__(self, model_rgb, model_fft, num_classes, device, checked_number, cotraining_start, k=30,
                 confidence_thresh_fft=0.95,
                 confidence_thresh_rgb=0.9):
        self.scheduler_rgb = None
        self.scheduler_fft = None
        self.model_rgb = model_rgb
        self.model_fft = model_fft
        self.num_classes = num_classes
        self.device = device
        self.checked_number = checked_number
        self.k = k
        self.confidence_thresh_fft = confidence_thresh_fft
        self.confidence_thresh_rgb = confidence_thresh_rgb
        self.criterion = nn.CrossEntropyLoss()
        self.cotraining_start = cotraining_start
        self.base_alpha = 0.5
        self.loss_history = []

        # Keep track of datasets for pseudo-labeling
        self.rgb_dataset = None
        self.fft_dataset = None
        self.unlabeled_dataset = None
        self.used_unlabeled_indices = set()  # Track which unlabeled samples have been used

    def set_datasets(self, rgb_dataset, fft_dataset, unlabeled_dataset):
        """
        Set the datasets for pseudo-labeling
        :param rgb_dataset: The dataset for RGB samples
        :param fft_dataset: The dataset for FFT samples
        :param unlabeled_dataset: The dataset for unlabeled samples
        :return: None
        """
        self.rgb_dataset = rgb_dataset
        self.fft_dataset = fft_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def init_schedulers(self, optimizer_rgb, optimizer_fft, step_size=5, gamma=0.9):
        """Initializes the StepLR schedulers for both optimizers."""
        self.scheduler_rgb = StepLR(optimizer_rgb, step_size=step_size, gamma=gamma)
        self.scheduler_fft = StepLR(optimizer_fft, step_size=step_size, gamma=gamma)
        print(f"Initialized StepLR schedulers: step_size={step_size}, gamma={gamma}")

    def train_iteration(self, rgb_loader, fft_loader, unlabeled_loader, optimizer_rgb, optimizer_fft, epoch_counter,
                        batch_size, reevaluate_flag):
        """
        Executes a training iteration
        :param checked_number: Number of samples to be reevaluated
        :param reevaluate_flag: Flag to indicate if a reevaluation is needed
        :param batch_size: Size of batch to be resampled
        :param rgb_loader: Loader for the RGB samples
        :param fft_loader: Loader for the FFT samples
        :param unlabeled_loader: Loader for the unsampled samples
        :param optimizer_rgb: Optimizer for the RGB model
        :param optimizer_fft: Optimizer for the FFT model
        :param epoch_counter: Actual epoch
        :return: None
        """
        # 1. Train on labeled data
        self.train_on_labeled(rgb_loader, fft_loader, optimizer_rgb, optimizer_fft)

        # 2. Label unlabeled data
        if epoch_counter > self.cotraining_start:
            if reevaluate_flag:
                rgb_removed, fft_removed = self.reevaluate_pseudo_labels(self.checked_number)
                self.adjust_confidence_threshold(rgb_removed, fft_removed, self.checked_number)

            rgb_cons, fft_cons = self.label_unlabeled_data(unlabeled_loader)

            if rgb_cons:
                # Add the same samples to both. This forces both models to
                # agree on the same 'ground truth' for the next epoch.
                added_rgb = self.rgb_dataset.add_pseudo_samples(rgb_cons)
                added_fft = self.fft_dataset.add_pseudo_samples(fft_cons)
                print(f"Added {added_rgb} shared samples to both datasets")

        if hasattr(self, 'scheduler_rgb'):
            self.scheduler_rgb.step()
            self.scheduler_fft.step()
            print(
                f"LR updated. RGB LR: {self.scheduler_rgb.get_last_lr()[0]:.6f}, FFT LR: {self.scheduler_fft.get_last_lr()[0]:.6f}")

        # 4. Evaluate
        # return self.evaluate(labeled_loader)

    def train_on_labeled(self, rgb_loader, fft_loader, optimizer_rgb, optimizer_fft):
        """
        This method performs the training iteration on the labeled/pseudo-labeled data. This uses a special objective
        loss function that uses KL divergence on the FFT model. This pushes the FFT model to have predictions closer to
        the Gray model. The alpha parameter determines the impact of the KL Divergence on the FFT model. Alpha is
        dynamic and updates accordingly to high variations in the loss for the FFT model.
        :param rgb_loader: The dataset for the Gray model
        :param fft_loader: The dataset for the FFT model (deprecated)
        :param optimizer_rgb: The optimizer for the Gray model (C-E)
        :param optimizer_fft: The optimizer for the FFT model (C-E)
        :return: None
        """

        self.model_rgb.train()
        self.model_fft.train()

        if len(self.loss_history) > 0:
            avg_loss = sum(self.loss_history) / len(self.loss_history)
        else:
            avg_loss = None

        epoch_losses = []
        for (rgb_imgs, fft_imgs, labels) in rgb_loader:
            rgb_imgs, fft_imgs, labels = rgb_imgs.to(self.device), fft_imgs.to(self.device), labels.to(self.device)

            temperature = 3.0  # Temperature for softening
            logits_rgb = self.model_rgb(rgb_imgs)
            logits_fft = self.model_fft(fft_imgs)

            # 1. Standard CE Loss
            loss_ce_rgb = self.criterion(logits_rgb, labels)
            loss_ce_fft = self.criterion(logits_fft, labels)

            # 2. Refined Consistency (FFT follows RGB)
            with torch.no_grad():
                # Soften the RGB targets to provide more 'distribution' info
                target_probs = torch.softmax(logits_rgb / temperature, dim=1)

            # Use log_softmax on FFT with the same Temperature
            loss_consistency = nn.KLDivLoss(reduction='batchmean')(
                torch.log_softmax(logits_fft / temperature, dim=1),
                target_probs
            ) * (temperature * temperature)  # Scaling factor for KL with temperature

            current_alpha = self.base_alpha
            if avg_loss and (loss_ce_fft.item() > avg_loss):
                ratio = loss_ce_fft.item() / avg_loss
                current_alpha = self.base_alpha * min(ratio, 2.0)

            current_loss = loss_ce_rgb + loss_ce_fft + (current_alpha * loss_consistency)

            optimizer_rgb.zero_grad()
            optimizer_fft.zero_grad()
            current_loss.backward()
            optimizer_rgb.step()
            optimizer_fft.step()

            epoch_losses.append(current_loss.item())

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        self.loss_history.append(mean_epoch_loss)

        if len(self.loss_history) > 5:
            self.loss_history.pop(0)

        print(f"The value for alpha is now: {current_alpha:.4f}")
        print(f"The average loss on this epoch is: {current_loss:.4f}")


    def label_unlabeled_data(self, unlabeled_loader):
        """
        This method performs pseudo-labeling (label generation). This mechanism is done by taking the highest
        prediction from both models, summing them up and averaging. The highest average wins the title of label on
        the agreement between the two models. The final label will always be what both models have agreed on.
        :param unlabeled_loader: The dataloader for the unlabeled dataset
        :return: The final sample to be added in the member list for the pseudo-labels
        """
        self.model_rgb.eval()
        self.model_fft.eval()

        consensus_samples = []
        current_idx = 0

        with torch.no_grad():
            for rgb_inputs, fft_inputs, _ in unlabeled_loader:
                rgb_inputs = rgb_inputs.to(self.device)
                fft_inputs = fft_inputs.to(self.device)

                rgb_probs = torch.softmax(self.model_rgb(rgb_inputs), dim=1)
                fft_probs = torch.softmax(self.model_fft(fft_inputs), dim=1)

                combined_probs = (rgb_probs + fft_probs) / 2

                max_probs, preds = torch.max(combined_probs, dim=1)

                _, rgb_preds = torch.max(rgb_probs, dim=1)
                _, fft_preds = torch.max(fft_probs, dim=1)

                for i in range(len(rgb_inputs)):
                    sample_idx = current_idx + i

                    if sample_idx not in self.used_unlabeled_indices:
                        agreement = (rgb_preds[i] == fft_preds[i])
                        high_confidence = (max_probs[i] > (self.confidence_thresh_rgb + self.confidence_thresh_fft) / 2)

                        if agreement and high_confidence:
                            consensus_samples.append({
                                'data': (rgb_inputs[i].cpu(), fft_inputs[i].cpu(), preds[i].item()),
                                'conf': max_probs[i].item(),
                                'idx': sample_idx
                            })

                current_idx += len(rgb_inputs)

        consensus_samples.sort(key=lambda x: x['conf'], reverse=True)

        top_k_samples = consensus_samples[:self.k]

        final_samples = []
        for s in top_k_samples:
            self.used_unlabeled_indices.add(s['idx'])
            final_samples.append(s['data'])

        return final_samples, final_samples

    def reevaluate_pseudo_labels(self, batch_size):

        """
        This method reevaluates the pseudo-samples that have been added through co-training.
        If the confidence in the pseudo-samples is lower than the threshold on one of the models OR
        The models have a different prediction on the pseudo-sample OR
        One of the model predicted something difference from the actual pseudo-label THEN
        The sample will be removed from the dataset.
        :param batch_size: Maximum number of samples to be removed
        :return: Returns the number of samples that have been removed
        """

        # We treat pseudo-samples as a shared pool now
        rgb_pseudo = self.rgb_dataset.pseudo_samples if self.rgb_dataset.pseudo_samples else []
        fft_pseudo = self.fft_dataset.pseudo_samples if self.fft_dataset.pseudo_samples else []

        # Take a random subset to check
        subset_size = min(self.checked_number, len(rgb_pseudo), len(fft_pseudo))
        if subset_size == 0:
            return 0, 0

        # Since datasets are synced, we can just sample from one and check both
        indices = random.sample(range(len(rgb_pseudo)), subset_size)

        self.model_rgb.eval()
        self.model_fft.eval()

        samples_to_remove = []

        for idx in indices:
            # Both datasets should have the same sample at the same index
            rgb_tensor, fft_tensor, pseudo_label = rgb_pseudo[idx]

            with torch.no_grad():
                rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
                fft_tensor = fft_tensor.unsqueeze(0).to(self.device)

                # Get predictions from both "eyes"
                rgb_out = self.model_rgb(rgb_tensor)
                fft_out = self.model_fft(fft_tensor)

                rgb_probs = torch.softmax(rgb_out, dim=1)
                fft_probs = torch.softmax(fft_out, dim=1)

                rgb_conf, rgb_pred = torch.max(rgb_probs, dim=1)
                fft_conf, fft_pred = torch.max(fft_probs, dim=1)

            low_confidence = (rgb_conf < self.confidence_thresh_rgb or fft_conf < self.confidence_thresh_fft)
            disagreement = (rgb_pred != fft_pred)
            wrong_label = (rgb_pred != pseudo_label or fft_pred != pseudo_label)

            if low_confidence or disagreement or wrong_label:
                samples_to_remove.append(rgb_pseudo[idx])

        # Remove the bad samples from BOTH datasets to keep them in sync
        rgb_count_removed = self.rgb_dataset.remove_pseudo_samples(samples_to_remove, self.confidence_thresh_rgb)
        fft_count_removed = self.fft_dataset.remove_pseudo_samples(samples_to_remove, self.confidence_thresh_fft)

        print(f"Joint Reevaluation: Removed {rgb_count_removed} inconsistent samples from the shared pool.")

        return rgb_count_removed, fft_count_removed

    def adjust_confidence_threshold(self, rgb_removed, fft_removed, batch_size):
        """
        This functions adjusts the confidence threshold during the reevaluation stage.
        The confidence threshold for both models are modified as followed:
        1. If the removal rate of pseudo-samples is below or equal to 30%, the threshold is augumented by 3%.
        2. If the removal rate of pseudo-samples is below or equal to 55%, the threshold is augumented by 1%.
        3. Else, the threshold is reduced by 4%.
        Also, to help the models create new pseudo-labels, the confidence threshold is set to have the maximum 98.5%

        :param rgb_removed: This contains the samples that have been predicted wrong for the Gray model
        :param fft_removed: This contains the samples that have been predicted wrong for the FFT model
        :param batch_size: The maximum number of samples that could have been removed
        :return: None
        """

        rgb_removal_rate = rgb_removed / batch_size
        fft_removal_rate = fft_removed / batch_size

        if rgb_removal_rate <= 0.3:
            self.confidence_thresh_rgb = self.confidence_thresh_rgb * 1.03
        elif rgb_removal_rate <= 0.55:
            self.confidence_thresh_rgb = self.confidence_thresh_rgb * 1.01
        else:
            self.confidence_thresh_rgb = self.confidence_thresh_rgb * 0.96

        if fft_removal_rate <= 0.3:
            self.confidence_thresh_fft = self.confidence_thresh_fft * 1.03
        elif fft_removal_rate <= 0.55:
            self.confidence_thresh_fft = self.confidence_thresh_fft * 1.01
        else:
            self.confidence_thresh_fft = self.confidence_thresh_fft * 0.96

        minimum_confidence = 0.985

        self.confidence_thresh_rgb = min(max(self.confidence_thresh_rgb, 0.75), minimum_confidence)
        self.confidence_thresh_fft = min(max(self.confidence_thresh_fft, 0.70), minimum_confidence)

        print(f"The removal rate for RGB dataset: {rgb_removal_rate}")
        print(f"The removal rate for FFT dataset: {fft_removal_rate}")
        print(f"The new confidence threshold for RGB model is: {self.confidence_thresh_rgb}")
        print(f"The new confidence threshold for FFT model is: {self.confidence_thresh_fft}")

        # self.checked_number = int(batch_size * (rgb_removal_rate + fft_removal_rate) / 2)

    def evaluate(self, loader):
        """
            Method to evaluate the models (RGB and FFT)
            :param loader: Data loader for any image type
            :return: Accuracy scores
        """
        self.model_rgb.eval()
        self.model_fft.eval()

        all_labels, rgb_preds, fft_preds, combined_preds = [], [], [], []

        with torch.no_grad():
            for rgb_inputs, fft_inputs, labels in loader:
                rgb_inputs, fft_inputs, labels = rgb_inputs.to(self.device), fft_inputs.to(self.device), labels.to(
                    self.device)

                rgb_outputs = self.model_rgb(rgb_inputs)
                fft_outputs = self.model_fft(fft_inputs)

                combined = (torch.softmax(rgb_outputs, dim=1) + torch.softmax(fft_outputs, dim=1)) / 2

                _, rgb_pred = torch.max(rgb_outputs, 1)
                _, fft_pred = torch.max(fft_outputs, 1)
                _, comb_pred = torch.max(combined, 1)

                all_labels.extend(labels.cpu().numpy())
                rgb_preds.extend(rgb_pred.cpu().numpy())
                fft_preds.extend(fft_pred.cpu().numpy())
                combined_preds.extend(comb_pred.cpu().numpy())

        return (
            accuracy_score(all_labels, rgb_preds),
            accuracy_score(all_labels, fft_preds),
            accuracy_score(all_labels, combined_preds),
            confusion_matrix(all_labels, rgb_preds),
            confusion_matrix(all_labels, fft_preds),
            confusion_matrix(all_labels, combined_preds)
        )

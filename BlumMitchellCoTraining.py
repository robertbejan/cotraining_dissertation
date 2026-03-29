import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import torch.fft


class BlumMitchellCoTraining:
    def __init__(self, model_rgb, model_fft, num_classes, device, checked_number, cotraining_start, k=30,
                 confidence_thresh_fft=0.95,
                 confidence_thresh_rgb=0.9):
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
        self.random_dropout = False

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

            rgb_confident_samples, fft_confident_samples = self.label_unlabeled_data(unlabeled_loader,
                                                                                     self.cotraining_start,
                                                                                     epoch_counter, batch_size)

            # RGB model's confident predictions go to FFT dataset
            rgb_added = 0
            if rgb_confident_samples:
                rgb_added = self.fft_dataset.add_pseudo_samples(rgb_confident_samples)
                if rgb_added > 0:
                    print(f"Added {rgb_added} new RGB-labeled samples to FFT dataset")
            # FFT model's confident predictions go to RGB dataset
            fft_added = 0
            if fft_confident_samples:
                fft_added = self.rgb_dataset.add_pseudo_samples(fft_confident_samples)
                if fft_added > 0:
                    print(f"Added {fft_added} new FFT-labeled samples to RGB dataset")

            if rgb_added == 0 and fft_added == 0:
                print("No new pseudo-labeled samples added (all were duplicates)")

        # 4. Evaluate
        # return self.evaluate(labeled_loader)

    def train_on_labeled(self, rgb_loader, fft_loader, optimizer_rgb, optimizer_fft):
        """
        :param rgb_loader: Data loader for RGB samples
        :param fft_loader: Data loader for FFT samples
        :param optimizer_rgb: Optimizer for RGB model
        :param optimizer_fft: Optimizier for FFT model
        :return: None
        """
        self.model_rgb.train()
        self.model_fft.train()

        for rgb_inputs, fft_inputs, labels in rgb_loader:
            rgb_inputs, fft_inputs, labels = rgb_inputs.to(self.device), fft_inputs.to(self.device), labels.to(
                self.device)

            # Train RGB model
            optimizer_rgb.zero_grad()
            rgb_outputs = self.model_rgb(rgb_inputs)
            loss_rgb = self.criterion(rgb_outputs, labels)
            loss_rgb.backward()
            optimizer_rgb.step()

        for rgb_inputs, fft_inputs, labels in fft_loader:
            rgb_inputs, fft_inputs, labels = rgb_inputs.to(self.device), fft_inputs.to(self.device), labels.to(
                self.device)

            # Train FFT model
            optimizer_fft.zero_grad()
            fft_outputs = self.model_fft(fft_inputs)
            loss_fft = self.criterion(fft_outputs, labels)
            loss_fft.backward()
            optimizer_fft.step()

    def label_unlabeled_data(self, unlabeled_loader, cotraining_start, epoch_counter, batch_size):
        """
        Method for labeling the unlabeled data.
        :param unlabeled_loader: Loader for unsampled samples
        :param cotraining_start: Default epoch number for the start of cotraining
        :param epoch_counter: Actual epoch
        :param batch_size: Size of the batch to be sampled
        :return: Returns the RGB and FFT samples
        """
        self.model_rgb.eval()
        self.model_fft.eval()

        rgb_confident = []
        fft_confident = []
        current_idx = 0

        with torch.no_grad():
            for batch_idx, (rgb_inputs, fft_inputs, _) in enumerate(unlabeled_loader):
                rgb_inputs, fft_inputs = rgb_inputs.to(self.device), fft_inputs.to(self.device)

                # Get predictions and confidence
                rgb_probs = torch.softmax(self.model_rgb(rgb_inputs), dim=1)
                fft_probs = torch.softmax(self.model_fft(fft_inputs), dim=1)

                rgb_max_probs, rgb_preds = torch.max(rgb_probs, dim=1)
                fft_max_probs, fft_preds = torch.max(fft_probs, dim=1)

                # Select confident samples that haven't been used yet
                for i in range(len(rgb_inputs)):
                    sample_idx = current_idx + i

                    # Only consider samples that haven't been used yet
                    if sample_idx not in self.used_unlabeled_indices:
                        if rgb_max_probs[i] > self.confidence_thresh_rgb:
                            rgb_confident.append((
                                rgb_inputs[i].cpu(),
                                fft_inputs[i].cpu(),
                                rgb_preds[i].cpu().item(),
                                sample_idx  # Include index for tracking
                            ))

                        if fft_max_probs[i] > self.confidence_thresh_fft:
                            fft_confident.append((
                                rgb_inputs[i].cpu(),
                                fft_inputs[i].cpu(),
                                fft_preds[i].cpu().item(),
                                sample_idx  # Include index for tracking
                            ))

                current_idx += len(rgb_inputs)

        # Sort by confidence and take top k
        rgb_confident.sort(
            key=lambda x: torch.softmax(self.model_rgb(x[0].unsqueeze(0).to(self.device)), dim=1).max().item(),
            reverse=True)
        fft_confident.sort(
            key=lambda x: torch.softmax(self.model_fft(x[1].unsqueeze(0).to(self.device)), dim=1).max().item(),
            reverse=True)

        # Take top k and mark indices as used
        rgb_top_k = rgb_confident[:self.k]
        fft_top_k = fft_confident[:self.k]

        # Mark used indices
        for _, _, _, idx in rgb_top_k:
            self.used_unlabeled_indices.add(idx)
        for _, _, _, idx in fft_top_k:
            self.used_unlabeled_indices.add(idx)

        # Remove index from returned samples
        rgb_samples = [(rgb, fft, label) for rgb, fft, label, _ in rgb_top_k]
        fft_samples = [(rgb, fft, label) for rgb, fft, label, _ in fft_top_k]

        return rgb_samples, fft_samples

    def reevaluate_pseudo_labels(self, batch_size):
        rgb_subset = random.sample(self.rgb_dataset.pseudo_samples,
                                   min(self.checked_number,
                                       len(self.rgb_dataset.pseudo_samples))) if self.rgb_dataset.pseudo_samples else []
        fft_subset = random.sample(self.fft_dataset.pseudo_samples,
                                   min(self.checked_number,
                                       len(self.fft_dataset.pseudo_samples))) if self.fft_dataset.pseudo_samples else []

        self.model_rgb.eval()
        self.model_fft.eval()
        rgb_samples_to_remove = []
        fft_samples_to_remove = []

        for sample in rgb_subset:
            rgb_tensor, fft_tensor, pseudo_label = sample
            with torch.no_grad():
                rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
                prediction = self.model_rgb(rgb_tensor)
                rgb_probs = torch.softmax(prediction, dim=1)
                rgb_max_probs, rgb_preds = torch.max(rgb_probs, dim=1)
            if rgb_max_probs < self.confidence_thresh_rgb or rgb_preds != pseudo_label:
                rgb_samples_to_remove.append(sample)

        for sample in fft_subset:
            rgb_tensor, fft_tensor, pseudo_label = sample
            with torch.no_grad():
                fft_tensor = fft_tensor.unsqueeze(0).to(self.device)
                prediction = self.model_fft(fft_tensor)
                fft_probs = torch.softmax(prediction, dim=1)
                fft_max_probs, fft_preds = torch.max(fft_probs, dim=1)

            if fft_max_probs < self.confidence_thresh_fft or fft_preds != pseudo_label:
                fft_samples_to_remove.append(sample)

        rgb_count_removed = self.rgb_dataset.remove_pseudo_samples(rgb_samples_to_remove, self.confidence_thresh_rgb)
        fft_count_removed = self.fft_dataset.remove_pseudo_samples(fft_samples_to_remove, self.confidence_thresh_fft)

        print(f"Number of removed RGB samples is: {rgb_count_removed}")
        print(f"Number of removed FFT samples is: {fft_count_removed}")

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

        if self.random_dropout:
            self.confidence_thresh_rgb = self.confidence_thresh_rgb * 1.03
            self.confidence_thresh_fft = self.confidence_thresh_fft * 1.03
        else:
            print("Adaptive confidence threshold")
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

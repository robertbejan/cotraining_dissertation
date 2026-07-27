from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


class Autoencoder(nn.Module):
    """Compresses concatenated DINOv2 and FFT features into a shared latent space."""

    def __init__(self, input_dim, latent_dim=128, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, features):
        latent = self.encoder(features)
        return self.decoder(latent), latent

    def encode(self, features):
        return self.encoder(features)


class DinoAdapter(nn.Module):
    """Frozen DINOv2 feature extractor loaded from the official repository."""

    def __init__(self, model_name="dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            return self.model(images)


class FFTEnsembleModel:
    """DINOv2 + FFT autoencoder encoder followed by an XGBoost classifier.

    XGBoost is not differentiable, so this branch is fitted after every labeled
    co-training epoch instead of receiving a PyTorch optimizer step.
    """

    def __init__(self, num_classes, device, latent_dim=128, autoencoder_epochs=60,
                 autoencoder_batch_size=32, xgboost_params=None):
        self.num_classes = num_classes
        self.device = device
        self.latent_dim = latent_dim
        self.autoencoder_epochs = autoencoder_epochs
        self.autoencoder_batch_size = autoencoder_batch_size
        self.dino = DinoAdapter().to(device)
        self.autoencoder = None
        self.classifier = None
        self.xgboost_params = xgboost_params or {}

    def train(self):
        self.dino.eval()
        if self.autoencoder is not None:
            self.autoencoder.train()
        return self

    def eval(self):
        self.dino.eval()
        if self.autoencoder is not None:
            self.autoencoder.eval()
        return self

    def _features(self, dino_images, fft_vectors):
        dino_images = dino_images.to(self.device)
        dino_embeddings = self.dino(dino_images)
        fft_features = fft_vectors.to(self.device).flatten(start_dim=1)
        return torch.cat((dino_embeddings, fft_features), dim=1)

    def _collect_features(self, loader):
        features, labels = [], []
        self.dino.eval()
        with torch.no_grad():
            for _, fft_vectors, dino_images, batch_labels in loader:
                features.append(self._features(dino_images, fft_vectors).cpu())
                labels.append(batch_labels.cpu())
        if not features:
            raise ValueError("Cannot fit the ensemble with an empty dataset.")
        return torch.cat(features), torch.cat(labels)

    def fit_loader(self, loader):
        """Train the autoencoder and fit XGBoost on the resulting latent vectors."""
        features, labels = self._collect_features(loader)
        if self.autoencoder is None:
            self.autoencoder = Autoencoder(features.shape[1], self.latent_dim).to(self.device)

        reconstruction_loader = DataLoader(
            TensorDataset(features), batch_size=self.autoencoder_batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.autoencoder.train()
        for _ in range(self.autoencoder_epochs):
            for (batch_features,) in reconstruction_loader:
                batch_features = batch_features.to(self.device)
                reconstruction, _ = self.autoencoder(batch_features)
                loss = criterion(reconstruction, batch_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.autoencoder.eval()
        with torch.no_grad():
            latent_features = self.autoencoder.encode(features.to(self.device)).cpu().numpy()

        classifier_params = {
            "objective": "multi:softprob",
            "num_class": self.num_classes,
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42,
        }
        classifier_params.update(self.xgboost_params)
        self.classifier = XGBClassifier(**classifier_params)
        self.classifier.fit(latent_features, labels.numpy())

    def predict_proba(self, dino_images, fft_vectors):
        if self.autoencoder is None or self.classifier is None:
            raise RuntimeError("The FFT ensemble must be fitted before prediction.")
        self.eval()
        with torch.no_grad():
            features = self._features(dino_images, fft_vectors)
            latent = self.autoencoder.encode(features).cpu().numpy()
        probabilities = self.classifier.predict_proba(latent)
        full_probabilities = np.zeros((len(latent), self.num_classes), dtype=np.float32)
        full_probabilities[:, self.classifier.classes_] = probabilities
        return torch.from_numpy(full_probabilities).to(self.device)

    def save(self, output_path):
        if self.autoencoder is None or self.classifier is None:
            raise RuntimeError("Cannot save an ensemble that has not been fitted.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "num_classes": self.num_classes,
            "latent_dim": self.latent_dim,
            "autoencoder_input_dim": self.autoencoder.encoder[0].in_features,
            "autoencoder_state_dict": self.autoencoder.state_dict(),
        }, output_path.with_suffix(".pt"))
        self.classifier.save_model(output_path.with_suffix(".json"))
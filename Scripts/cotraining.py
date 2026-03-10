import csv
import os
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet1_1_Weights
import torch.fft
import mlflow
import mlflow.pytorch

from Scripts.RGBWithFFTDataset import RGBWithFFTDataset
from Scripts.BlumMitchellCoTraining import BlumMitchellCoTraining
from Scripts.helper_functions import serialize_confusion_matrix

"""                                                           Questions:
- 
"""
"""                                                            In work:
- De modificat partea de modificat threshold
- Echilibrarea setului de date la inceput si in timpul antrenarii
"""
"""                                                             To do: 
- Change the model to regnet_x_400mf 
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_400mf.html#regnet-x-400mf
- Revalidation 
- try on other preliminary number of epochs (5, 7, 10) 
- 80, 50, 20% unlabeled sample databases
- individual model trainings + combination
- co-training with different thresholds (around 3 threshold)
- Change to adaptive confidence (descending) threshold
"""

"""                                                 What does this script file do
This is an app that implements the training of a Cotraining algorithm using two different models, and different databases.

The script is split into three parts:
1. The declaration and definition of a custom database class named "RGBWithFFTDataset" which inherits the torch.utils.
data.Dataset class. This class is initialized with the root directory of the data, the transformations for RGB and FFT 
databases, and a flag that indicates whether the object contains labeled/unlabeled samples.
2. The class for the model to be trained named "BlumMitchellCoTraining" which defines the methods for:
- training one iteration of the cotraining model, using the labeled data;
- labeling the unlabeled data, by predicting the best k prediction scores
- the evaluation of the model
3. The run section of all of the components which:
- loads the databases and creates loaders for train, test, unlabeled, validation
- executes the training procedure ( includes adding live samples to the databases that are used )
- evaluates the models
"""

# Starting MLFlow API experiment
mlflow.set_experiment("Fetal_Plane_CoTraining")

# Parameters and setup
class ExperimentConfig:
    """
    This class contains the configurations necessary for the training.
    """
    def __init__(self, dataset_type, cotraining_start, conf_rgb, conf_fft):
        self.dataset_type = dataset_type
        self.cotraining_start = cotraining_start
        self.conf_rgb = conf_rgb
        self.conf_fft = conf_fft

        # Fixed parameters
        self.input_size = (227, 227)
        self.batch_size = 30
        self.num_epochs = 60
        self.learning_rate = 1e-4
        self.k = 200
        self.checked_number = 100
        self.cotraining_batch_size = 50

        # Dataset mapping
        # Modify when needed
        dataset_map = {
            "small_80": {
                "base": "small_labeled_ultrasound_dataset",
                "unlabeled_pct": 80
            }
            # "organized_50": {
            #     "base": "organized_labeled_ultrasound_dataset",
            #     "unlabeled_pct": 50
            # },
            # "large_20": {
            #     "base": "large_labeled_ultrasound_dataset",
            #     "unlabeled_pct": 20
            # }
        }

        base_path = "D:/Facultate/Disertatie/mainProject/pythonProject1"
        dataset_info = dataset_map[dataset_type]

        self.labeled_path = os.path.join(base_path, dataset_info["base"], "labeled_train")
        self.unlabeled_path = os.path.join(base_path, dataset_info["base"], "unlabeled_train")
        self.val_path = os.path.join(base_path, dataset_info["base"], "validation")
        self.test_path = os.path.join(base_path, dataset_info["base"], "test")

        self.unlabeled_pct = dataset_info["unlabeled_pct"]
        self.experiment_id = f"{dataset_type}_start{cotraining_start}_rgb{conf_rgb}_fft{conf_fft}"


def initialize_rgb_model(num_classes, device):
    """
    This function initializes the Gray (RGB) model backbone of SqueezeNet with IMAGENET weights.
    :param num_classes: The number of classes
    :param device: The device on which the model will be trained
    :return: The final model
    """
    model_rgb = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    model_rgb.classifier[1] = nn.Conv2d(model_rgb.classifier[1].in_channels, num_classes, kernel_size=1)
    model_rgb.num_classes = num_classes
    model_rgb = model_rgb.to(device)
    return model_rgb


def initialize_fft_model(num_classes, device):
    """
    This function initializes the FFT model backbone of SqueezeNet with IMAGENET weights.
    :param num_classes: The number of classes
    :param device: The device on which the model will be trained
    :return: The final model
    """
    model_fft = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    new_layer = nn.Conv2d(1, 64, kernel_size=3, stride=2)
    pre_trained_weights = model_fft.features[0].weight.data
    new_layer.weight.data = pre_trained_weights.mean(dim=1, keepdim=True)
    model_fft.features[0] = new_layer
    model_fft.classifier[1] = nn.Conv2d(model_fft.classifier[1].in_channels, num_classes, kernel_size=1)
    model_fft.num_classes = num_classes
    model_fft = model_fft.to(device)
    return model_fft


def create_loaders(rgb_data, fft_data, unlabeled_data, val_data, test_data, batch_size):
    """
    This function creates the Dataloaders for all datasets (training, validation and test)
    :param rgb_data: Gray Dataloader
    :param fft_data: FFT Dataloader
    :param unlabeled_data: Unlabeled Dataloader
    :param val_data: Validation Dataloader
    :param test_data: Test Dataloader
    :param batch_size: Mini-batch size
    :return: Returns the Dataloaders
    """
    rgb_loader = DataLoader(rgb_data, batch_size=batch_size, shuffle=True)
    fft_loader = DataLoader(fft_data, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return rgb_loader, fft_loader, unlabeled_loader, val_loader, test_loader


def run_experiment(config):
    """
    This function performs the actual experiment
    :param config: The configurations class with all options
    :return: Returns the results in a dictionary
    """

    with mlflow.start_run(run_name=config.experiment_id):
        print("=" * 80)
        print(f"Starting Experiment: {config.experiment_id}")

        # 1. LOG PARAMETERS
        mlflow.log_params({
            "dataset_type": config.dataset_type,
            "cotraining_start": config.cotraining_start,
            "conf_rgb": config.conf_rgb,
            "conf_fft": config.conf_fft,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "k_samples": config.k
        })

    print("=" * 80)
    print(f"Starting Experiment: {config.experiment_id}")
    print(f"Dataset: {config.dataset_type} ({config.unlabeled_pct}% unlabeled)")
    print(f"Co-training starts: Epoch {config.cotraining_start}")
    print(f"Thresholds - RGB: {config.conf_rgb}, FFT: {config.conf_fft}")
    print("=" * 80)

    # Transforms
    rgb_transform = transforms.Compose([
        transforms.Resize(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ])

    fft_transform = transforms.Compose([
        transforms.Resize(config.input_size),
        transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),
        transforms.Normalize([0.5], [0.2])
    ])

    # Load datasets
    rgb_dataset = RGBWithFFTDataset(config.labeled_path, rgb_transform, fft_transform, labeled=True)
    fft_dataset = RGBWithFFTDataset(config.labeled_path, rgb_transform, fft_transform, labeled=True)
    unlabeled_dataset = RGBWithFFTDataset(config.unlabeled_path, rgb_transform, fft_transform, labeled=False)
    val_dataset = RGBWithFFTDataset(config.val_path, rgb_transform, fft_transform, labeled=True)
    test_dataset = RGBWithFFTDataset(config.test_path, rgb_transform, fft_transform, labeled=True)

    # Device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_classes = len(rgb_dataset.classes)
    print(f"Number of classes: {num_classes}")

    model_rgb = initialize_rgb_model(num_classes, device)
    model_fft = initialize_fft_model(num_classes, device)

    # Co-training setup
    cotrainer = BlumMitchellCoTraining(
        model_rgb, model_fft, num_classes, device,
        cotraining_start=config.cotraining_start,
        k=config.k,
        confidence_thresh_fft=config.conf_fft,
        confidence_thresh_rgb=config.conf_rgb,
        checked_number=config.checked_number
    )
    cotrainer.set_datasets(rgb_dataset, fft_dataset, unlabeled_dataset)

    optimizer_rgb = optim.Adam(model_rgb.parameters(), lr=config.learning_rate)
    optimizer_fft = optim.Adam(model_fft.parameters(), lr=config.learning_rate)

    # NEW - Initialize the LR schedulers - NEW
    cotrainer.init_schedulers(
        optimizer_rgb,
        optimizer_fft,
        step_size=5,
        gamma=0.9
    )

    # Training loop
    print("Starting co-training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        epoch_counter = epoch + 1

        rgb_loader, fft_loader, unlabeled_loader, val_loader, test_loader = create_loaders(
            rgb_dataset, fft_dataset, unlabeled_dataset, val_dataset, test_dataset, config.batch_size
        )

        print(f"RGB dataset size: {len(rgb_dataset)}, FFT dataset size: {len(fft_dataset)}")
        print(f"Used unlabeled samples: {len(cotrainer.used_unlabeled_indices)}/{len(unlabeled_dataset)}")

        reevaluate_flag = (epoch_counter > config.cotraining_start) and (epoch_counter % 4 == 0)
        if reevaluate_flag:
            print("Reevaluation is being performed on this iteration")

        cotrainer.train_iteration(rgb_loader, fft_loader, unlabeled_loader, optimizer_rgb, optimizer_fft,
                                  epoch_counter, config.cotraining_batch_size, reevaluate_flag)

        # Evaluate on validation set
        rgb_acc, fft_acc, combined_acc, _, _, _ = cotrainer.evaluate(val_loader)

        mlflow.log_metric("val_rgb_acc", rgb_acc, step=epoch_counter)
        mlflow.log_metric("val_fft_acc", fft_acc, step=epoch_counter)
        mlflow.log_metric("val_combined_acc", combined_acc, step=epoch_counter)
        mlflow.log_metric("unlabeled_samples_used", len(cotrainer.used_unlabeled_indices), step=epoch_counter)

        print(f"Validation Accuracy - RGB: {rgb_acc:.4f}, FFT: {fft_acc:.4f}, Combined: {combined_acc:.4f}")

    # Final evaluation
    print("\nTesting on test set...")
    _, _, _, val_loader, test_loader = create_loaders(
        rgb_dataset, fft_dataset, unlabeled_dataset, val_dataset, test_dataset, config.batch_size
    )
    rgb_acc, fft_acc, combined_acc, rgb_cm, fft_cm, combined_cm, = cotrainer.evaluate(test_loader)
    print(f"Test Accuracy - RGB: {rgb_acc:.4f}, FFT: {fft_acc:.4f}, Combined: {combined_acc:.4f}")

    mlflow.log_metrics({
        "test_rgb_acc": rgb_acc,
        "test_fft_acc": fft_acc,
        "test_combined_acc": combined_acc
    })

    # 4. LOG MODELS AS ARTIFACTS (Directly to MLflow)
    mlflow.pytorch.log_model(model_rgb, "model_rgb")
    mlflow.pytorch.log_model(model_fft, "model_fft")

    rgb_cm_string = serialize_confusion_matrix(rgb_cm)
    fft_cm_string = serialize_confusion_matrix(fft_cm)
    combined_cm_string = serialize_confusion_matrix(combined_cm)

    # Save models
    os.makedirs("../models", exist_ok=True)
    model_rgb_path = f"models/{config.experiment_id}_rgb2.pth"
    model_fft_path = f"models/{config.experiment_id}_fft2.pth"
    torch.save(model_rgb.state_dict(), model_rgb_path)
    torch.save(model_fft.state_dict(), model_fft_path)
    print(f"Models saved: {model_rgb_path}, {model_fft_path}")

    # Print final statistics
    print(f"\nFinal Statistics:")
    print(f"RGB dataset final size: {len(rgb_dataset)} (original + pseudo-labels)")
    print(f"FFT dataset final size: {len(fft_dataset)} (original + pseudo-labels)")
    print(f"Total unlabeled samples used: {len(cotrainer.used_unlabeled_indices)}/{len(unlabeled_dataset)}")
    print(f"RGB pseudo-samples: {len(rgb_dataset.pseudo_samples)}")
    print(f"FFT pseudo-samples: {len(fft_dataset.pseudo_samples)}")

    # Return results
    results = {
        "experiment_id": config.experiment_id,
        "dataset": config.dataset_type,
        "unlabeled_pct": config.unlabeled_pct,
        "cotraining_start": config.cotraining_start,
        "conf_rgb": config.conf_rgb,
        "conf_fft": config.conf_fft,
        "test_rgb_acc": rgb_acc,
        "test_fft_acc": fft_acc,
        "test_combined_acc": combined_acc,
        "rgb_confusion_matrix": rgb_cm_string,
        "fft_confusion_matrix": fft_cm_string,
        "combined_confusion_matrix": combined_cm_string,
        "num_classes": len(rgb_cm),
        "final_rgb_size": len(rgb_dataset),
        "final_fft_size": len(fft_dataset),
        "unlabeled_used": len(cotrainer.used_unlabeled_indices),
        "rgb_pseudo_samples": len(rgb_dataset.pseudo_samples),
        "fft_pseudo_samples": len(fft_dataset.pseudo_samples),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return results


def save_results_to_csv(results_list, filename="small_80_experiment_results.csv"):
    """
    Save or append results to CSV file
    """
    file_exists = os.path.isfile(filename)

    headers = [
        "experiment_id", "dataset", "unlabeled_pct",
        "cotraining_start", "conf_rgb", "conf_fft",
        "test_rgb_acc", "test_fft_acc", "test_combined_acc",
        "rgb_confusion_matrix", "fft_confusion_matrix", "combined_confusion_matrix",
        "num_classes", "final_rgb_size", "final_fft_size",
        "unlabeled_used", "rgb_pseudo_samples", "fft_pseudo_samples", "timestamp"
    ]

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        for result in results_list:
            writer.writerow(result)


def run_all_experiments():
    """
    Run all experiment combinations
    """
    datasets = ["small_80"]
    # datasets = ["small_80", "organized_50", "large_20"]
    cotraining_starts = [5]  # , 7, 10]
    threshold_configs = [
        {"rgb": 0.95, "fft": 0.90},  # High thresholds
        # {"rgb": 0.90, "fft": 0.85},  # Medium thresholds
        # {"rgb": 0.85, "fft": 0.80}  # Lower thresholds
    ]

    all_results = []
    total_experiments = len(datasets) * len(cotraining_starts) * len(threshold_configs)
    current_experiment = 0

    for dataset_type in datasets:
        for cotraining_start in cotraining_starts:
            for threshold_config in threshold_configs:
                current_experiment += 1
                print(f"\n\n{'=' * 80}")
                print(f"EXPERIMENT {current_experiment}/{total_experiments}")
                print(f"{'=' * 80}\n")

                try:
                    config = ExperimentConfig(
                        dataset_type,
                        cotraining_start,
                        threshold_config["rgb"],
                        threshold_config["fft"]
                    )

                    results = run_experiment(config)
                    all_results.append(results)

                    # Save intermediate results after each experiment
                    save_results_to_csv([results])

                except Exception as e:
                    print(f"Experiment failed with error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print(f"\n\n{'=' * 80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Total successful experiments: {len(all_results)}/{total_experiments}")
    print(f"Results saved to experiment_results.csv")
    print(f"{'=' * 80}")

    return all_results


def run_single_experiment(dataset_type, cotraining_start, conf_rgb, conf_fft):
    """
    Run a single experiment with specified parameters
    """
    config = ExperimentConfig(dataset_type, cotraining_start, conf_rgb, conf_fft)
    results = run_experiment(config)
    save_results_to_csv([results])

    print("\n\nExperiment completed!")
    print(f"Results: {results}")

    return results


# Main execution
if __name__ == "__main__":
    # Choose one of the following:

    # Option 1: Run a single experiment
    # result = run_single_experiment(
    #     dataset_type="small_80",
    #     cotraining_start=5,
    #     conf_rgb=0.95,
    #     conf_fft=0.90
    # )

    # Option 2: Run all experiments (uncomment to use)
    all_results = run_all_experiments()

    # Option 3: Run specific subset (uncomment to use)
    # results = []
    # for start_epoch in [5, 7, 10]:
    #     result = run_single_experiment(
    #         dataset_type="small_80",
    #         cotraining_start=start_epoch,
    #         conf_rgb=0.95,
    #         conf_fft=0.90
    #     )
    #     results.append(result)

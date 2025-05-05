"""
Train script for BreakHis CNN using stratified k-fold cross-validation.
"""

import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import BreakHisClassifier  # using our custom model


# using GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message):
    """
    Helper function to print log messages with timestamps.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def train_and_evaluate(dataset, labels, k_folds=5):
    """
    Trains and evaluates a model using stratified k-fold cross-validation.

    Parameters:
        dataset (str): the full dataset.
        labels (array): class labels corresponding to each data point.
        k_folds (int): number of folds for cross-validation.
                        optional; default = 5
    """
    # keeps the class distribution balanced in each fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_metrics = []

    for fold_index, (train_index, val_index) in enumerate(
        skf.split(np.zeros(len(labels)), labels), 1
    ):
        log(f"---- Fold {fold_index}/{k_folds} ----")

        # dataloaders for this fold
        train_loader = DataLoader(
            Subset(dataset, train_index), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_index), batch_size=32, shuffle=False
        )

        # setting up model, loss, and optimizer for this fold
        model = BreakHisClassifier(num_classes=2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        try:
            for epoch in range(15):
                model.train()
                epoch_loss = 0.0

                # going through the training data
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # evaluating on validation set
                model.eval()
                all_preds, all_targets = [], []

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())

                # calculating metrics for this epoch
                accuracy = accuracy_score(all_targets, all_preds)
                precision = precision_score(all_targets, all_preds)
                recall = recall_score(all_targets, all_preds)
                f1 = f1_score(all_targets, all_preds)

                # logging those metrics for this epoch
                log(
                    f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | "
                    f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}"
                )

        # catching any unexpected crashes
        except Exception as e:
            log(f"Error during training during fold {fold_index}: {str(e)}")
            continue

        # storing this fold's scores
        fold_metrics.append((precision, recall, f1))

    # overall scores for all folds
    avg_precision, avg_recall, avg_f1 = np.mean(fold_metrics, axis=0)
    log(
        f"\nCross-validation results - average precision: {avg_precision:.4f} "
        f"recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}"
    )

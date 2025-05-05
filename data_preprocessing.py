"""
Utility functions for organizing and splitting image data for training and validation.
"""

import os
import shutil
import random
import datetime


def log(message):
    """
    Helper function to print log messages with timestamps.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def organize_data(source_dir, dest_dir):
    """
    Copies images from the source directory into a new destination directory.
    Keeps class folders like 'benign' or 'malignant', but flattens the
    subfolder structure within them.

    Parameters:
        source_dir (str): path to source data with subfolders.
        dest_dir (str): path to destination location.

    Copies images with extensions .png, .jpg, .jpeg, or .tif from source_dir to dest_dir.
    """
    if not os.path.exists(source_dir):
        log(f"Error: Source directory {source_dir} does not exist.")
        return

    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as e:
        log(f"Error creating destination directory: {e}")
        return

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                source = os.path.join(root, filename)
                destination = os.path.join(dest_dir, filename)

                if os.path.exists(destination):
                    log(f"File {filename} already exists in the destination. Skipping.")
                    continue

                try:
                    shutil.copy2(source, destination)
                except OSError as e:
                    log(f"Error copying file {filename}: {e}")


def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Randomly split the dataset into training and validation sets
    based on the given split_ratio.

    Parameters:
        source_dir (str): path to base folder with class subfolders.
        train_dir (str): where to place training images
        val_dir (str): where to place validation images
        split_ratio (float): ratio of training data (default: 0.8)
    """
    if not os.path.exists(source_dir):
        log(f"Error: Source directory {source_dir} does not exist.")
        return

    try:
        os.makedirs(train_dir, exist_ok=True)
    except OSError as e:
        log(f"Error creating training directory: {e}")
        return

    try:
        os.makedirs(val_dir, exist_ok=True)
    except OSError as e:
        log(f"Error creating validation directory: {e}")
        return

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        try:
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
        except OSError as e:
            log(f"Error creating class directories for {class_name}: {e}")
            continue

        for img in train_images:
            source = os.path.join(class_path, img)
            destination = os.path.join(train_class_dir, img)
            shutil.copy(source, destination)

        for img in val_images:
            source = os.path.join(class_path, img)
            destination = os.path.join(val_class_dir, img)
            shutil.copy(source, destination)

    log(
        f"Data split complete with {split_ratio * 100}% for "
        f"training and {100 - split_ratio * 100}% for validation."
    )

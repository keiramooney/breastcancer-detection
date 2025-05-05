# BreakHis CNN Classifier

A PyTorch-based Convolutional Neural Network (CNN) to classify histopathology images as **benign** or **malignant** using the [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis). Includes support for image preprocessing, train/validation splitting, stratified k-fold cross-validation, and detailed evaluation metrics.

---

## Repository Structure

```
project/
│
├── model.py
├── train.py
├── data_preprocessing.py
├── README.md
└── data/
    ├── raw/
    ├── processed/
    │   ├── train/
    │   └── val/
```

---

## Features

- Stratified **k-fold cross-validation**
- CNN with **batch normalization** + **dropout**
- Detailed logs for **accuracy**, **precision**, **recall**, **F1 score**

---

## Example Usage
### 1. Dataset

This project uses the [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis) (Breast Cancer Histopathological Image Classification). Due to licensing restrictions, the dataset is **not included** in this repository.

To use this project:
1. Download the dataset manually from the [Kaggle BreakHis page](https://www.kaggle.com/datasets/ambarish/breakhis).
2. Extract it into a folder named `data/raw/` (or any path you prefer).
3. Run `data_preprocessing.py` to flatten and split the dataset into training and validation sets.

### 2. Organize and split your data

```python
from data_preprocessing import organize_data, split_data

organize_data('path/to/raw_data', 'path/to/flattened_data')
split_data('path/to/flattened_data', 'train_dir', 'val_dir')
```

### 3. Train with cross-validation

```python
from train import train_and_evaluate
from dataset import BreakHisDataset  # custom dataset class
import torch

dataset = BreakHisDataset('train_dir')
labels = dataset.get_labels()  # extract labels as needed

train_and_evaluate(dataset, labels, k_folds=5)
```

---

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- NumPy

Install everything with:

```bash
pip install torch torchvision scikit-learn numpy
```

---

## Sample Results

The model was trained using stratified 5-fold cross-validation. Below are the average results:
- **Average Accuracy:** 0.916
- **Average Recall:** 0.940
- **Average F1 Score:** 0.939

Example training output:
Epoch 12 | Loss: 0.4356 | Accuracy: 91.2% | F1 Score: 0.912
---

## Author

**Keira Mooney**  
MIT '27 — Mathematics and AI + Decision Making  

---

## Acknowledgements

- [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)
- PyTorch, scikit-learn, NumPy

---

## License

MIT License. Free to use and modify.

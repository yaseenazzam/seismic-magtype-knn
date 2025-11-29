# Seismic Magnitude Type Classification with k-NN

This repository contains a machine learning project for classifying earthquake
magnitude types (`magType`) using k-Nearest Neighbours (k-NN). The project was
completed as part of the CURE project in ENDG 319 at the University of Calgary.

## Overview

We build a supervised learning pipeline to predict the magnitude type of an
earthquake using four attributes:

- `mag` (magnitude)
- `longitude`
- `latitude`
- `depth`

The goal is to compare different scalers and values of *k* and then select a
final model based on test performance and generalization behaviour.

## Methods

- **Model:** k-Nearest Neighbours classifier
- **Features:** `mag`, `longitude`, `latitude`, `depth`
- **Target:** `magType`
- **Preprocessing:**
  - Train–test split
  - Feature scaling with:
    - `MinMaxScaler`
    - `StandardScaler`
- **Evaluation:**
  - Accuracy on train and test sets
  - Accuracy vs. k (for k = 1 … 30)
  - Confusion matrix for the final model

## Experiments

### 1. Accuracy vs. k

For both `MinMaxScaler` and `StandardScaler` we plotted:

- Training accuracy as a function of k  
- Test accuracy as a function of k  

This allowed us to see how the model overfits for small k and how performance
stabilizes for larger k.

> Figures:  
> - `figures/accuracy_vs_k_minmax.png`  
> - `figures/accuracy_vs_k_standard.png`

### 2. Comparing scalers

Both scalers were tested over the same range of k values. We observed that
`MinMaxScaler` provided slightly better test accuracy for the best-performing
region of k and was chosen for the final model.

## Final Model

- **Model:** k-NN
- **k:** 8
- **Scaler:** `MinMaxScaler`
- **Training accuracy:** 0.919
- **Test accuracy:** 0.903

These values indicate good generalization from train to test data.

## Confusion Matrix

We computed a 21×21 confusion matrix (rows = true labels, columns = predicted
labels) for all `magType` classes.

Key observations:

- Most correct predictions lie on the diagonal, especially for the most common
  magnitude types.
- Misclassifications occur mainly in rare classes, which is expected given the
  strong class imbalance in the dataset.
- The model performs best on frequent `magType` labels and less reliably on
  underrepresented ones.

## Example Inference

We also tested the final model on a new, unseen instance:

```text
mag = 5.6
longitude = -120.5
latitude = 34.2
depth = 10.0

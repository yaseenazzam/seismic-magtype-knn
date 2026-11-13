# Seismic Magnitude Type Classification using k-NN

A machine learning project that classifies earthquake magnitude types using **k-Nearest Neighbours (k-NN)**, achieving **90.3% test accuracy**.

Developed as part of the Course-Based Undergraduate Research Experience (CURE) in ENDG 319 at the University of Calgary.

## Technologies

**Python · scikit-learn · NumPy · Pandas · Matplotlib · Jupyter Notebook**

## Project Overview

This project explores the use of supervised machine learning to predict earthquake magnitude types (`magType`) from seismic data.

The classification model uses four features:

- **Magnitude (`mag`)**
- **Longitude**
- **Latitude**
- **Depth**

The primary objective was to evaluate how feature scaling and the number of nearest neighbours affect classification accuracy and model generalization.

## Methodology

### 1. Data Preprocessing

- Prepared seismic data for supervised classification.
- Split data into training and testing sets.
- Compared two feature scaling techniques:
  - `MinMaxScaler`
  - `StandardScaler`

### 2. Model Development

Implemented a k-Nearest Neighbours classifier using scikit-learn.

Evaluated values of **k = 1–30** to investigate the relationship between model complexity, overfitting, and classification performance.

### 3. Model Evaluation

Compared training and test accuracy across different values of k and feature scaling methods.

Used a confusion matrix to examine classification performance across individual magnitude types.

## Results

| Metric | Result |
|---|---|
| Algorithm | k-Nearest Neighbours |
| Optimal k selected | 8 |
| Feature scaling | MinMaxScaler |
| Training accuracy | 91.9% |
| Test accuracy | **90.3%** |
| Target classes | 21 |

`MinMaxScaler` produced slightly better test accuracy in the best-performing region of k values compared with `StandardScaler`.

The selected model achieved similar training and test accuracy, although performance varied across individual magnitude types.

### Key Findings

- Feature scaling affected k-NN classification performance.
- Smaller values of k showed a greater tendency toward overfitting.
- More frequently represented magnitude types were classified more reliably.
- Less common classes were more challenging to predict due to class imbalance.

## Example Prediction

The trained model was also evaluated on an example earthquake with the following characteristics:

```python
sample = {
    "mag": 5.6,
    "longitude": -120.5,
    "latitude": 34.2,
    "depth": 10.0
}
```

The sample demonstrates how new earthquake measurements can be passed through the trained classification pipeline.

## Limitations and Future Improvements

While the model achieved 90.3% overall test accuracy, several limitations remain:

- **Class imbalance:** The model performs less consistently on underrepresented magnitude types.
- **Limited feature selection:** Only four seismic attributes were used.
- **Model selection:** Additional cross-validation could improve confidence in hyperparameter selection.

Potential improvements include evaluating class-specific precision and recall, experimenting with additional features, and comparing k-NN against other classification algorithms.

## Acknowledgements

Developed as part of the ENDG 319 CURE project at the University of Calgary.

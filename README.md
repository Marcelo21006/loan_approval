# Loan Risk Analysis with Variational Autoencoder

## Overview

This repository contains a loan risk analysis system that combines unsupervised deep learning with traditional machine learning techniques. The system uses a Variational Autoencoder (VAE) for anomaly detection and feature learning, followed by ensemble classification methods to predict loan default probability.

## Key Features

- **Anomaly Detection**: Identifies unusual loan applications that may require additional scrutiny
- **Risk Categorization**: Classifies loans into low, medium, and high risk categories
- **Ensemble Classification**: Combines multiple models to improve prediction accuracy
- **Comprehensive Evaluation**: Includes detailed performance metrics and visualizations

## Technical Architecture

The solution consists of several key components:

1. **Variational Autoencoder (VAE)**
   - Encoder network with regularization, batch normalization, and dropout
   - Latent space representation (10-dimensional by default)
   - Decoder network for reconstruction
   - Custom loss function combining reconstruction loss and KL divergence

2. **Data Preprocessing Pipeline**
   - Standardization for numerical features
   - One-hot encoding for categorical features
   - Structured feature handling using ColumnTransformer

3. **Classification Models**
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - AdaBoost Classifier
   - Bagging Classifier
   - Voting Classifier
   - Deep Neural Network

## Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib
- Seaborn

## Usage

### Loading and Preparing Data

```python
import pandas as pd
from preprocessing import prepare_data

# Load your loan data
df = pd.read_csv('loan_data.csv')

# Prepare data for the VAE
X = prepare_data(df)
```

### Training the VAE Model

```python
from vae_model import VariationalAutoencoder

# Initialize and train VAE
vae = VariationalAutoencoder(input_dim=X.shape[1], latent_dim=10)
history = vae.fit(X, epochs=50, batch_size=256)
```

### Detecting Anomalies and Categorizing Risk

```python
# Detect anomalies
is_anomaly, reconstruction_errors = vae.detect_anomalies(X)

# Add results to dataframe
df['is_anomaly'] = is_anomaly
df['reconstruction_error'] = reconstruction_errors

# Categorize risk
df['risk_category'] = df['reconstruction_error'].apply(categorize_risk)
```

### Training Classification Models

```python
from classification import advanced_loan_classification

# Prepare features and target
X_ = preprocessor.fit_transform(X)
y = df['loan_status']

# Train and evaluate models
results = advanced_loan_classification(X_, y)
```

## Dataset

The model expects loan data with the following features:

### Numerical Features
- `person_age`: Age of the loan applicant
- `person_income`: Annual income of the applicant
- `person_emp_exp`: Employment experience in years
- `loan_amnt`: Loan amount requested
- `loan_int_rate`: Interest rate of the loan
- `loan_percent_income`: Loan amount as a percentage of income
- `cb_person_cred_hist_length`: Credit history length
- `credit_score`: Credit score of the applicant

### Categorical Features
- `person_gender`: Gender of the applicant
- `person_education`: Education level
- `person_home_ownership`: Home ownership status
- `loan_intent`: Purpose of the loan
- `previous_loan_defaults_on_file`: Previous defaults

## Model Evaluation

The repository includes evaluation tools:

- Classification reports (precision, recall, F1-score)
- Confusion matrices
- ROC curves and AUC scores
- Cross-validation metrics

## Customization

You can customize the VAE architecture and classification models by adjusting parameters:

- `latent_dim`: Dimension of the latent space (default: 10)
- `threshold_percentile`: Percentile threshold for anomaly detection (default: 95)
- Classification model parameters (e.g., n_estimators, max_depth)

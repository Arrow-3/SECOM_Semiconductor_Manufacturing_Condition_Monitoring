import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(data_path, label_path):

    # Load data
    X = pd.read_csv(data_path, sep=' ', header=None)
    y_data = pd.read_csv(label_path, sep=' ', header=None)

    # Extract labels
    y = y_data.iloc[:, 0]

    # Convert labels: -1 → 0, 1 → 1
    y = y.replace(-1, 0)

    # Handle missing values (NaN → mean)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Remove constant features
    variance = np.var(X_imputed, axis=0)
    X_filtered = X_imputed[:, variance > 0]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    return X_scaled, y

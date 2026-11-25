"""
PhishGuard – Training Data Feature Pipeline
============================================
Applies the feature extraction module across the full cleaned URL dataset
and exports the engineered feature dataset for machine learning.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(__file__))
from feature_extraction import (
    FEATURE_COLUMNS,
    extract_features_from_url
)

def main():
    cleaned_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_urls.csv')
    output_features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_dataset.csv')
    
    print(f"Loading cleaned data from: {cleaned_data_path}")
    df = pd.read_csv(cleaned_data_path)
    total_records = len(df)
    print(f"Total URL records to process: {total_records:,}")
    
    t0 = time.time()
    print("Extracting features from all URLs (in-memory vectorized processing)...")
    
    # Process features in list comprehension for optimal performance
    urls = df['URL'].tolist()
    feature_records = [extract_features_from_url(u) for u in urls]
    
    # Create DataFrame strictly with FEATURE_COLUMNS
    features_df = pd.DataFrame(feature_records, columns=FEATURE_COLUMNS)
    
    # Retain the ground-truth target label
    features_df['Label'] = df['Label'].values
    
    t1 = time.time()
    print(f"Feature extraction completed in: {t1 - t0:.2f} seconds.")
    
    # Quality & Integrity Validation
    print("\n=== FEATURE MATRIX VALIDATION ===")
    print(f"Feature Matrix Shape: {features_df.shape[0]:,} rows, {features_df.shape[1]} columns")
    print(f"Numerical Feature Columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Target Label Column: 'Label'")
    
    # Check for missing values
    null_counts = features_df[FEATURE_COLUMNS].isnull().sum()
    print(f"\nMissing Values in Numerical Features:")
    print(null_counts)
    
    # Check for infinite values
    inf_counts = np.isinf(features_df[FEATURE_COLUMNS].values).sum()
    print(f"\nInfinite Values in Numerical Features: {inf_counts}")
    
    # Feature summary statistics
    print("\n=== SUMMARY STATISTICS OF EXTRACTED FEATURES ===")
    print(features_df[FEATURE_COLUMNS].describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    # Save the feature dataset
    print(f"\nSaving feature dataset to: {output_features_path}...")
    features_df.to_csv(output_features_path, index=False)
    file_size_mb = os.path.getsize(output_features_path) / (1024 * 1024)
    print(f"Successfully saved features dataset ({file_size_mb:.2f} MB).")

if __name__ == '__main__':
    main()

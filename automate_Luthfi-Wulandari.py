# Ini komentar untuk trigger workflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(path):
    print("Loading dataset...")
    return pd.read_csv(path)

def clean_data(df):
    print("Cleaning data...")
    df.fillna(df.median(), inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def encode_features(df, categorical_cols):
    print("Encoding categorical features...")
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    return df

def scale_features(df, numerical_cols):
    print("Scaling numerical features...")
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def save_data(df, output_path):
    print(f"Saving processed data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    input_path = 'namadataset_raw/heart.csv'
    output_path = 'preprocessing/namadataset_preprocessing/heart_clean.csv'

    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    df = load_data(input_path)
    df = clean_data(df)
    df = encode_features(df, categorical_cols)
    df = scale_features(df, numerical_cols)
    save_data(df, output_path)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

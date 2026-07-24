import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse

def preprocess_data(raw_data_path, output_dir):
    print(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print("Initial shape:", df.shape)
    
    if 'gameId' in df.columns:
        df = df.drop(columns=['gameId'])
        print("Dropped 'gameId' column.")
        
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Handling {missing} missing values by dropping rows.")
        df = df.dropna()
        
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Dropping {duplicates} duplicated rows.")
        df = df.drop_duplicates()
        
    print("Standardizing features...")
    X = df.drop(columns=['blueWins'])
    y = df['blueWins']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    df_clean = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    print("Cleaned shape:", df_clean.shape)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_clean.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Preprocessing")
    parser.add_argument("--input", type=str, default="../league_of_legends_raw/high_diamond_ranked_10min.csv", help="Path to raw dataset")
    parser.add_argument("--output_dir", type=str, default="../league_of_legends_preprocessing", help="Directory to save preprocessed data")
    
    args = parser.parse_args()
    preprocess_data(args.input, args.output_dir)

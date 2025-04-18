import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(df):
    # Drop non-numeric or irrelevant columns
    df = df.drop(columns=["CustomerID"], errors='ignore')
    
    # Convert categorical data
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    
    return numeric_df, scaled

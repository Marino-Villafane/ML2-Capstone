import pandas as pd
import numpy as np

def preprocess_data(df):
    """Clean and preprocess input data"""
    df = df.copy()
    
    # Handle temporal features
    temporal_cols = ['Month', 'DayofMonth', 'DayOfWeek']
    for col in temporal_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Validate ranges
    df['Month'] = df['Month'].clip(1, 12)
    df['DayofMonth'] = df['DayofMonth'].clip(1, 31)
    df['DayOfWeek'] = df['DayOfWeek'].clip(1, 7)
    
    # Convert other features
    df['DepTime'] = pd.to_numeric(df['DepTime'])
    df['Distance'] = pd.to_numeric(df['Distance'])
    
    # Handle categorical features
    categorical_cols = ['UniqueCarrier', 'Origin', 'Dest']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df

def engineer_features(df):
    """Engineer features for prediction"""
    df = df.copy()
    
    # Extract hour from DepTime
    df['Hour'] = df['DepTime'] // 100
    
    # Create cyclical features
    def create_cyclical_features(values, max_value):
        values = 2 * np.pi * values / max_value
        return np.sin(values), np.cos(values)
    
    # Apply cyclical transformations
    df['Month_sin'], df['Month_cos'] = create_cyclical_features(df['Month'], 12)
    df['DayOfWeek_sin'], df['DayOfWeek_cos'] = create_cyclical_features(df['DayOfWeek'], 7)
    df['Hour_sin'], df['Hour_cos'] = create_cyclical_features(df['Hour'], 24)
    
    # Create distance bins and time categories
    df['Distance_Bin'] = pd.qcut(df['Distance'], q=5, labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long'])
    df['TimeOfDay'] = pd.cut(df['Hour'], 
                            bins=[-np.inf, 6, 12, 18, np.inf], 
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    # Drop processed columns
    df = df.drop(['DepTime', 'Hour'], axis=1)
    
    return df

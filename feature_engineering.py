import pandas as pd
import os

def feature_engineering():
    input_file = 'cleaned_data.csv'
    output_file = 'featured_data.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found. Please run clean_data.py first.")
        return

    # 1. Load the dataset named 'cleaned_data.csv' and ensure the 'Date' column is set as the datetime index
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Sort chronologically to make sure shifts and rolling calculations are correct
    df.sort_index(inplace=True)
    
    # 2. Create lag features for the 'Children in HHS Care' column: t-1, t-7, and t-14
    # (These represent the 1st, 7th, and 14th previous reporting periods)
    df['HHS_Care_Lag_1'] = df['Children in HHS Care'].shift(1)
    df['HHS_Care_Lag_7'] = df['Children in HHS Care'].shift(7)
    df['HHS_Care_Lag_14'] = df['Children in HHS Care'].shift(14)
    
    # 3. Create 7-period and 14-period rolling averages for the 'Children in HHS Care' column
    df['HHS_Care_Rolling_7'] = df['Children in HHS Care'].rolling(window=7).mean()
    df['HHS_Care_Rolling_14'] = df['Children in HHS Care'].rolling(window=14).mean()
    
    # 4. Create flow-based signal column 'Net_Pressure' = 'Children transferred' - 'Children discharged'
    df['Net_Pressure'] = df['Children transferred'] - df['Children discharged']
    
    # 5. Drop any rows with NaN (empty) values created during the shifting/rolling process
    # This will remove the first 14 rows where lag_14 and rolling_14 are NaN.
    df_featured = df.dropna()
    
    # Save this final, updated dataset as a new file named 'featured_data.csv'
    df_featured.to_csv(output_file)
    
    print(f"Feature engineering completed successfully! Saved to '{output_file}'.")
    print(f"Original dataset shape: {df.shape}")
    print(f"Featured dataset shape (after dropping NaNs): {df_featured.shape}")
    print("\nColumns in featured dataset:")
    print(df_featured.columns.tolist())
    print("\nFirst 5 rows of featured data:")
    print(df_featured.head())
    print("\nLast 5 rows of featured data:")
    print(df_featured.tail())

if __name__ == '__main__':
    feature_engineering()

import pandas as pd
import os

def clean_data():
    input_file = 'raw_data.tsv'
    output_file = 'cleaned_data.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Read the data (tab-separated)
    # Use keep_default_na=False to prevent empty strings from becoming NaN prematurely
    df = pd.read_csv(input_file, sep='\t', keep_default_na=False)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Rename columns to match requested names
    rename_dict = {
        'Children apprehended and placed in CBP custody*': 'Children apprehended',
        'Children transferred out of CBP custody': 'Children transferred',
        'Children discharged from HHS Care': 'Children discharged'
    }
    df = df.rename(columns=rename_dict)
    
    # Define target columns in order
    columns_order = [
        'Date',
        'Children apprehended',
        'Children in CBP custody',
        'Children transferred',
        'Children in HHS Care',
        'Children discharged'
    ]
    
    # Keep only the requested columns
    df = df[columns_order]
    
    # Filter out empty or header repeating rows
    df = df[df['Date'].astype(str).str.strip() != '']
    df = df[df['Date'].astype(str).str.strip() != 'Date']
    
    # Clean numeric columns (remove commas and convert to integer)
    for col in columns_order[1:]:
        # Remove commas, strip spaces, convert to numeric
        df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
    # Clean Date column (strip spaces)
    df['Date'] = df['Date'].astype(str).str.strip()
    
    # Parse dates to YYYY-MM-DD format
    df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y', errors='coerce')
    
    # Drop rows with unparseable dates
    df = df.dropna(subset=['Date'])
    
    # Sort chronologically (earliest date first) for time series forecasting
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Format Date column back to YYYY-MM-DD string representation in CSV
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Write to CSV
    df.to_csv(output_file, index=False)
    print(f"Data cleaning completed successfully. Cleaned dataset saved to '{output_file}'.")
    print(f"Total rows cleaned: {len(df)}")
    print("\nFirst 5 rows of cleaned data:")
    print(df.head())
    print("\nLast 5 rows of cleaned data:")
    print(df.tail())

if __name__ == '__main__':
    clean_data()

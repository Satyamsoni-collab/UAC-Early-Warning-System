import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

def decompose():
    input_file = 'cleaned_data.csv'
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found. Please run clean_data.py first.")
        return

    # Load cleaned data
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # The data is collected every few days (irregular). Let's resample it to daily frequency
    # using linear interpolation to fill the gaps for decomposition.
    df_daily = df.resample('D').interpolate(method='linear')
    
    # We will decompose the 'Children in HHS Care' time series
    # Using an additive model, with a weekly period (7 days) and yearly period (365 days)
    # to understand both short-term weekly fluctuations and long-term trends.
    print("Performing seasonal decomposition on 'Children in HHS Care'...")
    
    # 1. Weekly seasonality decomposition (period = 7 days)
    result_weekly = seasonal_decompose(df_daily['Children in HHS Care'], model='additive', period=7)
    
    # 2. Yearly seasonality decomposition (period = 365 days)
    result_yearly = seasonal_decompose(df_daily['Children in HHS Care'], model='additive', period=365)
    
    # Save the yearly decomposition plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    result_yearly.observed.plot(ax=axes[0], legend=False, color='blue')
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Yearly Seasonal Decomposition - Children in HHS Care')
    
    result_yearly.trend.plot(ax=axes[1], legend=False, color='red')
    axes[1].set_ylabel('Trend')
    
    result_yearly.seasonal.plot(ax=axes[2], legend=False, color='green')
    axes[2].set_ylabel('Seasonality')
    
    result_yearly.resid.plot(ax=axes[3], legend=False, color='purple')
    axes[3].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig('decomposition_plot_yearly.png')
    plt.close()
    
    # Save the weekly decomposition plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    result_weekly.observed.plot(ax=axes[0], legend=False, color='blue')
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Weekly Seasonal Decomposition - Children in HHS Care')
    
    result_weekly.trend.plot(ax=axes[1], legend=False, color='red')
    axes[1].set_ylabel('Trend')
    
    result_weekly.seasonal.plot(ax=axes[2], legend=False, color='green')
    axes[2].set_ylabel('Seasonality')
    
    result_weekly.resid.plot(ax=axes[3], legend=False, color='purple')
    axes[3].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig('decomposition_plot_weekly.png')
    plt.close()
    
    # Let's print some descriptive stats about Trend and Seasonality
    trend = result_yearly.trend.dropna()
    seasonal = result_yearly.seasonal.dropna()
    residual = result_yearly.resid.dropna()
    
    print("\nDecomposition completed successfully!")
    print(f"Data range: {df_daily.index.min().strftime('%Y-%m-%d')} to {df_daily.index.max().strftime('%Y-%m-%d')}")
    print(f"Trend range: Min={trend.min():.2f}, Max={trend.max():.2f}")
    print(f"Yearly Seasonality peak-to-peak amplitude: {seasonal.max() - seasonal.min():.2f}")
    print(f"Weekly Seasonality peak-to-peak amplitude: {(result_weekly.seasonal.max() - result_weekly.seasonal.min()):.2f}")
    print("Plots saved as 'decomposition_plot_yearly.png' and 'decomposition_plot_weekly.png'.")

if __name__ == '__main__':
    decompose()

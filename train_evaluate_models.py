import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def evaluate_models():
    input_file = 'featured_data.csv'
    predictions_file = 'model_predictions.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found. Please run feature_engineering.py first.")
        return

    # 1. Load the dataset and set the 'Date' column as the datetime index
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    target_col = 'Children in HHS Care'
    
    # 2. Perform a strict time-based train-test split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train period: {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    print(f"Test period:  {test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')} ({len(test_df)} rows)")
    
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # Define features for Machine Learning model
    features = [
        'HHS_Care_Lag_1', 'HHS_Care_Lag_7', 'HHS_Care_Lag_14',
        'HHS_Care_Rolling_7', 'HHS_Care_Rolling_14', 'Net_Pressure'
    ]
    X_train = train_df[features]
    X_test = test_df[features]

    # --- Model 1: Baseline Model (Moving Average forecast) ---
    # We use the 7-day rolling average feature as the baseline forecast.
    # Specifically, the 7-day rolling average at t-1 (using HHS_Care_Rolling_7 shifted by 1)
    # is a standard moving average baseline for predicting the next value.
    y_pred_baseline = test_df['HHS_Care_Rolling_7']

    # --- Model 2: Statistical Model (ARIMA) ---
    # We fit a univariate ARIMA(2, 1, 2) model on y_train
    print("\nTraining ARIMA model...")
    try:
        # We suppress warnings to keep the output clean
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', UserWarning)
        
        # Fit ARIMA
        model_arima = ARIMA(y_train, order=(2, 1, 2))
        res_arima = model_arima.fit()
        
        # Perform dynamic forecasting over the test set horizon
        y_pred_arima = res_arima.forecast(steps=len(test_df))
        # Ensure dates match test_df index
        y_pred_arima.index = test_df.index
    except Exception as e:
        print(f"ARIMA training failed: {e}. Falling back to ARIMA(1, 1, 1)...")
        model_arima = ARIMA(y_train, order=(1, 1, 1))
        res_arima = model_arima.fit()
        y_pred_arima = res_arima.forecast(steps=len(test_df))
        y_pred_arima.index = test_df.index

    # --- Model 3: Machine Learning Model (Random Forest Regressor) ---
    print("Training Random Forest Regressor...")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    y_pred_rf = pd.Series(y_pred_rf, index=test_df.index)

    # --- Evaluation ---
    models_preds = {
        'Baseline (Moving Average)': y_pred_baseline,
        'ARIMA': y_pred_arima,
        'Random Forest': y_pred_rf
    }
    
    results = {}
    for name, preds in models_preds.items():
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds) * 100 # percentage
        results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    # Print evaluation metrics side-by-side
    results_df = pd.DataFrame(results).T
    print("\n" + "="*50)
    print("             MODEL EVALUATION METRICS")
    print("="*50)
    print(results_df.round(2))
    print("="*50)

    # Determine the best model based on MAE
    best_model_name = results_df['MAE'].idxmin()
    best_preds = models_preds[best_model_name]
    print(f"\nBest Model identified by lowest MAE: {best_model_name}")

    # Save the predictions to a new file named 'model_predictions.csv'
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Baseline_Predictions': y_pred_baseline,
        'ARIMA_Predictions': y_pred_arima,
        'Random_Forest_Predictions': y_pred_rf
    }, index=test_df.index)
    
    predictions_df.to_csv(predictions_file)
    print(f"All predictions saved to '{predictions_file}'.")

    # Save a comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index[-60:], y_train.iloc[-60:], label='Historical Train (Last 60 Days)', color='black')
    plt.plot(test_df.index, y_test, label='Actual Test', color='blue', linewidth=2)
    plt.plot(test_df.index, y_pred_baseline, label='Baseline Forecast', color='gray', linestyle='--')
    plt.plot(test_df.index, y_pred_arima, label='ARIMA Forecast', color='red', linestyle=':')
    plt.plot(test_df.index, y_pred_rf, label='Random Forest Forecast', color='green', alpha=0.8)
    plt.title('HHS Care Load Forecasting: Model Comparison')
    plt.ylabel('Children in HHS Care')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison_plot.png')
    plt.close()
    print("Saved comparison plot as 'model_comparison_plot.png'.")

if __name__ == '__main__':
    evaluate_models()

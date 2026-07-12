# Predictive Forecasting of Care Load & Placement Demand (UAC Early Warning System)

An interactive, end-to-end data science and forecasting pipeline to monitor, analyze, and predict the care load and placement (discharge) demand for **Unaccompanied Alien Children (UAC)** under HHS care. 

This platform leverages statistical and machine learning models to help government stakeholders anticipate capacity constraints and manage operational workflows proactively.

---

## 🚀 Features

*   **Standardized Data Preparation**: Cleans irregular reporting dates, handles numerical comma formatting, and formats data chronologically.
*   **Time-Series Decomposition**: Isolates long-term structural trends, yearly seasonality cycles, and weekly administrative workflow cycles using additive decomposition.
*   **Feature Engineering**: Generates rolling averages, time-based lag variables ($t-1$, $t-7$, $t-14$), and custom system-pressure flow signals (`Net_Pressure`).
*   **Forecasting Model Suite**: Trains, evaluates, and compares:
    *   *Baseline Model*: 7-day Rolling Moving Average
    *   *Statistical Model*: Univariate ARIMA(2,1,2)
    *   *Machine Learning Model*: Random Forest Regressor
*   **Interactive Streamlit Dashboard**: A professional, responsive web interface utilizing Plotly for dynamic charts, $95\%$ confidence interval shading, alert threshold indicators, and inflow/outflow balance sheets.

---

## 📁 Repository Structure

```
├── clean_data.py               # Raw data ingestion & cleaning script
├── decompose_time_series.py    # Seasonal decomposition analysis & plotting
├── feature_engineering.py      # Generates lags, rolling averages, and pressure metrics
├── train_evaluate_models.py    # Trains and evaluates MA baseline, ARIMA, and Random Forest
├── app.py                      # Streamlit interactive web dashboard application
├── raw_data.tsv                # Irregular raw tab-separated dataset
├── cleaned_data.csv            # Standardized chronological daily dataset
├── featured_data.csv           # Engineered dataset with lags and rolling features
├── model_predictions.csv       # Test set predictions for side-by-side comparison
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/uac-forecasting-dashboard.git
    cd uac-forecasting-dashboard
    ```

2.  **Install Dependencies**:
    Make sure you have Python 3.9+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Running the Pipeline

You can run each module step-by-step or launch the dashboard directly (the dashboard will automatically read pre-generated data files).

### Step 1: Clean Raw Data
Parses raw data and outputs a standardized CSV:
```bash
python clean_data.py
```

### Step 2: Run Seasonal Decomposition Analysis
Analyzes trends and seasonal fluctuations (outputs `decomposition_plot_yearly.png` and `decomposition_plot_weekly.png`):
```bash
python decompose_time_series.py
```

### Step 3: Engineer Time-Series Features
Generates lag variables, rolling means, and net-pressure features:
```bash
python feature_engineering.py
```

### Step 4: Train and Evaluate Models
Trains the models, compares test performance, and writes predictions to file:
```bash
python train_evaluate_models.py
```

### Step 5: Launch the Streamlit Dashboard
Launches the interactive dashboard in your browser:
```bash
streamlit run app.py
```

---

## 📊 Model Evaluation Summary

Models were trained on historical records from **Feb 9, 2023, to May 15, 2025**, and evaluated on a holdout test set from **May 18, 2025, to Dec 21, 2025 (142 periods)**.

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute Percentage Error (MAPE) |
| :--- | :---: | :---: | :---: |
| **Baseline (Moving Average)** | **25.07** | **30.88** | **1.12%** |
| **Random Forest Regressor** | 60.85 | 83.33 | 2.82% |
| **ARIMA(2, 1, 2)** | 234.66 | 284.06 | 11.06% |

*Note: ARIMA error reflects dynamic recursive forecasting over the full 142-day test set without mid-period correction, while Random Forest and the Baseline operate on 1-step-ahead forecasts.*

---

## 🛡️ Dashboard Features for Stakeholders
*   **Forecast Horizon Slider**: Look ahead 7, 14, 21, or 30 days.
*   **Model Selector**: Swap forecasting approaches in real-time.
*   **Capacity Warning Panels**: Dynamically turns yellow when the model projects care load to exceed target thresholds (e.g., $3,500$ children).
*   **Flow Comparison Graphs**: Visually compares UAC inflow (`Children transferred` into care) against outflow (`Children discharged`).

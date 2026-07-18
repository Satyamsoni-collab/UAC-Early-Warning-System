# 🛡️ UAC Early Warning System (Predictive Forecasting)

🌐 Live Demo: https://uac-early-warning-system-aafymefjynpec9ympafffq.streamlit.app/

Hi everyone! 👋 Welcome to my machine learning project.

I built this end-to-end forecasting pipeline and interactive dashboard to solve a real-world problem: helping government stakeholders (like HHS and CBP) manage care facilities for Unaccompanied Children (UAC).

Usually, these systems are reactive—they look at past data to see what happened. I wanted to see if I could use historical data to build a proactive Early Warning System that predicts care loads and placement demands 7 to 30 days in advance, helping to prevent bed shortages and understaffing.

## 🚀 What I Built (Features)

*   **Data Cleaning & Prep**: Took raw, irregular daily data, fixed the missing dates, handled weird formatting, and created a clean chronological dataset.
*   **Exploratory Analysis**: Decomposed the time-series data to find real patterns (like the huge peak in late December and mid-week workflow spikes).
*   **Feature Engineering**: I didn't just feed raw numbers into the model. I created custom features like 7-day/14-day rolling averages, lag features, and a custom 'Net Pressure' signal (Transfers minus Discharges) to measure system stress.
*   **Model Training**: Trained and compared three different models:
    *   A 7-day Moving Average (as my baseline)
    *   An ARIMA(2,1,2) statistical model
    *   A Random Forest Regressor (Machine Learning)
*   **Interactive Dashboard**: Wrapped all the predictions into a live Streamlit web app using Plotly for some really cool, interactive charts.

---

## 📁 How I Organized the Code

Here's a quick look at the project structure:

```
├── clean_data.py               # Script to ingest and clean the raw data
├── decompose_time_series.py    # Analyzes trends & seasonality (generates plots)
├── feature_engineering.py      # Creates lag features and my custom pressure metrics
├── train_evaluate_models.py    # Trains the ML models and calculates error metrics
├── app.py                      # The main Streamlit dashboard application!
├── raw_data.tsv                # The original messy data
├── cleaned_data.csv            # Cleaned up data
├── featured_data.csv           # Data ready for the ML models
├── model_predictions.csv       # Saved predictions to display on the dashboard
├── requirements.txt            # All the Python libraries you need to run this
└── README.md                   # You are here!
```

---

## 🛠️ Want to run this locally?

If you want to test out the code on your own machine, just follow these steps:

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/your-username/uac-forecasting-dashboard.git
    cd uac-forecasting-dashboard
    ```

2.  **Install the dependencies**:  
    Make sure you have Python 3.9+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**:  
    Because I already saved the processed data and predictions in the repo, you can jump straight to launching the app!
    ```bash
    streamlit run app.py
    ```

*Note: If you want to see how the data was processed from scratch, you can run the python scripts in order (`clean_data.py` -> `feature_engineering.py` -> `train_evaluate_models.py`).*

---

## 📊 How did the models do?

I trained the models on data from Feb 2023 to May 2025, and tested them on a hidden holdout set from May to Dec 2025 (142 days). Here are the results:

| Model | MAE | RMSE | MAPE |
| :--- | :---: | :---: | :---: |
| **Baseline (Moving Average)** | **25.07** | **30.88** | **1.12%** |
| **Random Forest Regressor** | 60.85 | 83.33 | 2.82% |
| **ARIMA(2, 1, 2)** | 234.66 | 284.06 | 11.06% |

*Note: The Baseline and Random Forest were evaluated on short-term 1-step-ahead forecasts. I pushed the ARIMA model much harder by making it do a dynamic recursive forecast over the entire 142-day test set without any mid-period corrections, which is why the compound error is higher but represents a true long-term planning scenario!*

---

## 🛡️ Inside the Dashboard

If you run the Streamlit app, you'll see a few cool features I added for the end-users:

*   **Forecast Horizon Slider**: Users can drag a slider to look 7, 14, 21, or 30 days into the future.
*   **Model Toggle**: A dropdown to instantly switch between the Baseline, ARIMA, and ML predictions.
*   **Visual Alerts**: I hardcoded a safe operational limit (3,500 children). If the model predicts we will cross this line, the UI throws a prominent yellow warning to alert the staff.
*   **Confidence Intervals**: Shaded areas on the graphs to show the worst-case and best-case scenarios.

---

Feel free to reach out or open an issue if you have any questions about the code!

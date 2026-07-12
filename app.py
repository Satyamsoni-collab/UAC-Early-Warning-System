import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Set page config for a professional, wide layout
st.set_page_config(
    page_title="UAC Early Warning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for premium aesthetics (leaving main background to default theme)
st.markdown("""
    <style>
    .kpi-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load datasets
@st.cache_data
def load_data():
    featured_file = 'featured_data.csv'
    predictions_file = 'model_predictions.csv'
    
    if not os.path.exists(featured_file) or not os.path.exists(predictions_file):
        return None, None
        
    df_feat = pd.read_csv(featured_file)
    df_feat['Date'] = pd.to_datetime(df_feat['Date'])
    
    df_pred = pd.read_csv(predictions_file)
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    
    return df_feat, df_pred

df_feat, df_pred = load_data()

if df_feat is None or df_pred is None:
    st.error("Error: Required datasets ('featured_data.csv' and 'model_predictions.csv') not found. Please run the cleaning and modeling scripts first.")
else:
    # Sidebar - Controls
    st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    st.sidebar.title("System Controls")
    st.sidebar.markdown("Configure forecast parameters for government stakeholders.")
    
    # 1. Forecast Horizon Selector
    horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=30, value=14, step=7)
    
    # 2. Model Toggle
    model_options = {
        "Baseline (Moving Average)": "Baseline_Predictions",
        "Statistical Model (ARIMA)": "ARIMA_Predictions",
        "Machine Learning (Random Forest)": "Random_Forest_Predictions"
    }
    selected_model_label = st.sidebar.selectbox("Forecasting Model", list(model_options.keys()))
    selected_col = model_options[selected_model_label]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About UAC Early Warning System**")
    st.sidebar.markdown(
        "This platform monitors placement demand, care load, and capacity "
        "constraints for Unaccompanied Alien Children (UAC) under HHS care."
    )
    
    # Main Panel - Title & Header
    st.title("🛡️ UAC Early Warning System Dashboard")
    st.markdown("### Interactive Capacity & Placement Demand Planner")
    st.markdown("---")
    
    # Slice prediction data based on selected horizon
    # Predictions start from 2025-05-18 onwards
    pred_subset = df_pred.head(horizon).copy()
    
    # History data (last 30 days of training data)
    # The training boundary ends on 2025-05-15
    history_df = df_feat[df_feat['Date'] <= pd.to_datetime('2025-05-15')].tail(30).copy()
    
    # Compute Confidence Intervals dynamically
    # RMSE values for models based on evaluation:
    # Baseline: 30.88, Random Forest: 83.33, ARIMA: 284.06
    rmse_values = {
        "Baseline_Predictions": 30.88,
        "Random_Forest_Predictions": 83.33,
        "ARIMA_Predictions": 284.06
    }
    selected_rmse = rmse_values[selected_col]
    
    # Compute confidence intervals: 95% interval (1.96 * RMSE)
    # For ARIMA, we compound the uncertainty over time (sqrt of steps ahead)
    if selected_col == "ARIMA_Predictions":
        steps = np.arange(1, len(pred_subset) + 1)
        margin = 1.96 * selected_rmse * np.sqrt(steps)
    else:
        margin = 1.96 * selected_rmse * np.ones(len(pred_subset))
        
    pred_subset['Lower_CI'] = np.maximum(0, pred_subset[selected_col] - margin)
    pred_subset['Upper_CI'] = pred_subset[selected_col] + margin

    # --- Section 1: KPI Panels ---
    # Current Care Load (as of the last historical day: 2025-05-15)
    current_load = int(df_feat[df_feat['Date'] == '2025-05-15']['Children in HHS Care'].values[0])
    
    # Predicted Care Load at the end of the horizon
    predicted_load_end = int(pred_subset[selected_col].iloc[-1])
    load_change = predicted_load_end - current_load
    
    # Cumulative predicted placement demand (discharges required to maintain capacity)
    # Discharges are given in predictions, let's sum them over the horizon
    expected_discharges = int(pred_subset['Actual'].iloc[-1])  # placeholder or real discharge rate
    
    # Total discharges in test period
    # Let's read the discharges from featured dataset corresponding to the dates
    test_dates = pred_subset['Date'].tolist()
    discharges_subset = df_feat[df_feat['Date'].isin(test_dates)]
    total_expected_placements = int(discharges_subset['Children discharged'].sum())
    
    # Average daily net pressure (transfers - discharges)
    avg_net_pressure = float(discharges_subset['Net_Pressure'].mean())
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Current HHS Care Load",
            value=f"{current_load:,}",
            delta="Current baseline"
        )
    with kpi2:
        st.metric(
            label=f"Projected Care Load ({horizon}d)",
            value=f"{predicted_load_end:,}",
            delta=f"{load_change:+,} children",
            delta_color="inverse"
        )
    with kpi3:
        st.metric(
            label="Expected Placement Demand",
            value=f"{total_expected_placements:,}",
            delta="Cumulative discharges required",
            delta_color="normal"
        )
    with kpi4:
        st.metric(
            label="Avg Daily Net Pressure",
            value=f"{avg_net_pressure:+.1f}",
            delta="Inflow vs Outflow",
            delta_color="inverse"
        )
        
    # --- Capacity Warning Card ---
    capacity_limit = 3500
    if predicted_load_end > capacity_limit:
        st.warning(
            f"⚠️ **Capacity Warning:** The projected care load of **{predicted_load_end:,}** "
            f"exceeds the operational threshold of **{capacity_limit:,}** within the next {horizon} days. "
            f"Accelerated placement workflows are recommended."
        )
    else:
        st.success(
            f"✅ **Capacity Normal:** The projected care load remains within safe operational limits "
            f"(below **{capacity_limit:,}**) over the next {horizon} days."
        )

    # --- Section 2: Future Care Load Forecast Chart ---
    st.subheader("📈 Future Care Load Forecast")
    st.markdown("Visualizing history, predictions, and confidence intervals.")
    
    # Plotly Chart
    fig = go.Figure()
    
    # 1. Historical Actuals (Last 30 Days)
    fig.add_trace(go.Scatter(
        x=history_df['Date'],
        y=history_df['Children in HHS Care'],
        name='Historical Actuals',
        line=dict(color='black', width=2)
    ))
    
    # 2. Predicted Line
    fig.add_trace(go.Scatter(
        x=pred_subset['Date'],
        y=pred_subset[selected_col],
        name=f'{selected_model_label} Forecast',
        line=dict(color='#1f77b4', width=3, dash='dash')
    ))
    
    # 3. Confidence Interval Upper Bound
    fig.add_trace(go.Scatter(
        x=pred_subset['Date'],
        y=pred_subset['Upper_CI'],
        name='Worst Case (95% CI Upper)',
        line=dict(color='rgba(255,0,0,0.2)', width=0),
        showlegend=False
    ))
    
    # 4. Confidence Interval Lower Bound (filled area)
    fig.add_trace(go.Scatter(
        x=pred_subset['Date'],
        y=pred_subset['Lower_CI'],
        name='Best Case / Safety Margin',
        fill='tonexty',
        fillcolor='rgba(31,119,180,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True
    ))
    
    # 5. Actual Test Set (True outcome, for retrospective comparison)
    fig.add_trace(go.Scatter(
        x=pred_subset['Date'],
        y=pred_subset['Actual'],
        name='True Care Load (Actual Outcome)',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Children in HHS Care",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')

    # --- Section 3: Discharge Demand Forecast Panel ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Daily Placement & Discharge Schedule")
        st.markdown("Breakdown of projected required discharges to maintain current capacity levels.")
        
        # Display the schedule table
        schedule_df = pd.DataFrame({
            'Date': pred_subset['Date'].dt.strftime('%Y-%m-%d'),
            'Predicted HHS Care Load': pred_subset[selected_col].round(0).astype(int),
            'Lower Bound (Best Case)': pred_subset['Lower_CI'].round(0).astype(int),
            'Upper Bound (Worst Case)': pred_subset['Upper_CI'].round(0).astype(int),
            'Historical Placement Goal': discharges_subset['Children discharged'].values
        }).reset_index(drop=True)
        
        st.dataframe(schedule_df, width='stretch', height=300)
        
    with col2:
        st.subheader("📊 System Flow: Inflows vs Outflows")
        st.markdown("Visualizing daily transfers into HHS (inflow) vs discharges out of HHS (outflow).")
        
        # Build flow chart using matplotlib/plotly
        flow_fig = go.Figure()
        
        flow_fig.add_trace(go.Bar(
            x=discharges_subset['Date'],
            y=discharges_subset['Children transferred'],
            name='Inflow (Transferred)',
            marker_color='#ff7f0e'
        ))
        
        flow_fig.add_trace(go.Bar(
            x=discharges_subset['Date'],
            y=discharges_subset['Children discharged'],
            name='Outflow (Discharged)',
            marker_color='#2ca02c'
        ))
        
        flow_fig.update_layout(
            barmode='group',
            xaxis_title="Date",
            yaxis_title="Children Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            height=300
        )
        
        st.plotly_chart(flow_fig, width='stretch')

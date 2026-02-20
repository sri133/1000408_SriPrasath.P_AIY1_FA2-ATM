# =====================================================
# ATM INTELLIGENCE DEMAND FORECASTING - ADVANCED FA2
# With Clustering + Anomaly + Neural Network Forecast
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="ATM Intelligence Dashboard", layout="wide")

st.title("🏧 ATM Intelligence Demand Forecasting Dashboard")

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_AIY1_FA2-ATM/main/atm_cash_management_dataset.csv"
    return pd.read_csv(url)

df = load_data()

# -----------------------------------------------------
# Preprocessing
# -----------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

# Encode categorical columns
cat_cols = ["Day_of_Week", "Time_of_Day", "Location_Type", "Weather_Condition"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes

# -----------------------------------------------------
# Sidebar Filter
# -----------------------------------------------------
st.sidebar.header("Filters")
locations = st.sidebar.multiselect(
    "Location Type",
    df["Location_Type"].unique(),
    default=df["Location_Type"].unique()
)

filtered_df = df[df["Location_Type"].isin(locations)]

# -----------------------------------------------------
# KPI Metrics
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total ATMs", df["ATM_ID"].nunique())
col2.metric("Avg Withdrawal", round(df["Total_Withdrawals"].mean(), 2))
col3.metric("Records", len(df))

# =====================================================
# STAGE 3 - EDA
# =====================================================

st.header("📊 Exploratory Data Analysis")

fig1 = px.line(filtered_df, x="Date", y="Total_Withdrawals",
               title="Withdrawals Over Time", template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(filtered_df, x="Total_Withdrawals",
                    nbins=40, template="plotly_white",
                    title="Withdrawal Distribution")
st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# STAGE 4 - CLUSTERING (KMEANS)
# =====================================================

st.header("📍 ATM Clustering")

cluster_features = ["Total_Withdrawals", "Total_Deposits", "Location_Type"]
X_cluster = filtered_df[cluster_features]

scaler_cluster = StandardScaler()
X_scaled_cluster = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
filtered_df["Cluster"] = kmeans.fit_predict(X_scaled_cluster)

fig_cluster = px.scatter(
    filtered_df,
    x="Total_Withdrawals",
    y="Total_Deposits",
    color="Cluster",
    size="Total_Withdrawals",
    hover_data=["ATM_ID"],
    template="plotly_white",
    title="Clustered ATMs"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# =====================================================
# STAGE 5 - ANOMALY DETECTION (IQR)
# =====================================================

st.header("⚠️ Anomaly Detection")

Q1 = filtered_df["Total_Withdrawals"].quantile(0.25)
Q3 = filtered_df["Total_Withdrawals"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

filtered_df["Anomaly"] = (
    (filtered_df["Total_Withdrawals"] < lower) |
    (filtered_df["Total_Withdrawals"] > upper)
)

fig_anomaly = go.Figure()

fig_anomaly.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["Total_Withdrawals"],
    mode="lines",
    name="Normal"
))

fig_anomaly.add_trace(go.Scatter(
    x=filtered_df[filtered_df["Anomaly"]]["Date"],
    y=filtered_df[filtered_df["Anomaly"]]["Total_Withdrawals"],
    mode="markers",
    marker=dict(color="red", size=8),
    name="Anomaly"
))

fig_anomaly.update_layout(title="Withdrawal Anomaly Detection",
                          template="plotly_white")

st.plotly_chart(fig_anomaly, use_container_width=True)

# =====================================================
# STAGE 6 - FORECASTING (MLP REGRESSOR)
# =====================================================

st.header("🤖 Neural Network Demand Forecasting")

if "Cash_Demand_Next_Day" in df.columns:

    features = [
        "Total_Withdrawals",
        "Total_Deposits",
        "Location_Type",
        "Month",
        "Week_Number"
    ]

    X = df[features]
    y = df["Cash_Demand_Next_Day"]

    scaler_nn = MinMaxScaler()
    X_scaled = scaler_nn.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    colA, colB = st.columns(2)
    colA.metric("MAE", round(mae, 2))
    colB.metric("R² Score", round(r2, 3))

    fig_pred = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title="Actual vs Predicted Cash Demand",
        template="plotly_white"
    )

    st.plotly_chart(fig_pred, use_container_width=True)

else:
    st.warning("Cash_Demand_Next_Day column not found in dataset.")

st.success("✅ Advanced ATM Intelligence Dashboard Running Successfully")

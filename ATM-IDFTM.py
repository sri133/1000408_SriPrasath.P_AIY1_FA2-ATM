# ==============================================
# ATM INTELLIGENCE DEMAND FORECASTING - FA2
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ATM Intelligence Dashboard", layout="wide")

st.title("🏧 ATM Intelligence Demand Forecasting Dashboard")

# ==============================================
# LOAD DATA
# ==============================================

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_AIY1_FA2-ATM/main/atm_cash_management_dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ==============================================
# DATA PREPROCESSING
# ==============================================

# Convert Date column safely
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Fill missing values safely
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# Extract date features
df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week
df["Day"] = df["Date"].dt.day

# Encode categorical columns
categorical_cols = ["Day_of_Week", "Time_of_Day", "Location_Type", "Weather_Condition"]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes

# ==============================================
# SIDEBAR FILTERS
# ==============================================

st.sidebar.header("🔎 Filters")

selected_location = st.sidebar.multiselect(
    "Select Location Type",
    df["Location_Type"].unique(),
    default=df["Location_Type"].unique()
)

filtered_df = df[df["Location_Type"].isin(selected_location)]

# ==============================================
# STAGE 3 - EDA
# ==============================================

st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Withdrawals Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["Total_Withdrawals"], bins=30)
    st.pyplot(fig)

with col2:
    st.subheader("Deposits Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["Total_Deposits"], bins=30)
    st.pyplot(fig)

# Time Trend
st.subheader("📈 Withdrawals Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["Total_Withdrawals"])
plt.xticks(rotation=45)
st.pyplot(fig)

# Holiday Impact
if "Holiday_Flag" in filtered_df.columns:
    st.subheader("🎉 Holiday Impact on Withdrawals")
    fig, ax = plt.subplots()
    sns.boxplot(x=filtered_df["Holiday_Flag"], y=filtered_df["Total_Withdrawals"], ax=ax)
    st.pyplot(fig)

# ==============================================
# CORRELATION HEATMAP (SAFE FIXED VERSION)
# ==============================================

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ==============================================
# STAGE 4 - CLUSTERING
# ==============================================

st.header("📍 ATM Clustering")

cluster_features = ["Total_Withdrawals", "Total_Deposits", "Location_Type"]

X = filtered_df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

st.subheader("Elbow Method")
fig, ax = plt.subplots()
ax.plot(range(1, 6), inertia, marker="o")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Inertia")
st.pyplot(fig)

# Final Clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
filtered_df["Cluster"] = kmeans.fit_predict(X_scaled)

st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(
    filtered_df["Total_Withdrawals"],
    filtered_df["Total_Deposits"],
    c=filtered_df["Cluster"]
)
st.pyplot(fig)

# ==============================================
# STAGE 5 - ANOMALY DETECTION (IQR METHOD)
# ==============================================

st.header("⚠️ Anomaly Detection")

Q1 = filtered_df["Total_Withdrawals"].quantile(0.25)
Q3 = filtered_df["Total_Withdrawals"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_df["Anomaly"] = (
    (filtered_df["Total_Withdrawals"] < lower_bound) |
    (filtered_df["Total_Withdrawals"] > upper_bound)
)

st.subheader("Withdrawal Anomalies Highlighted")

fig, ax = plt.subplots()
ax.scatter(filtered_df["Date"], filtered_df["Total_Withdrawals"])
ax.scatter(
    filtered_df[filtered_df["Anomaly"]]["Date"],
    filtered_df[filtered_df["Anomaly"]]["Total_Withdrawals"]
)
plt.xticks(rotation=45)
st.pyplot(fig)

# ==============================================
# SUMMARY INSIGHTS
# ==============================================

st.header("📌 Key Insights")

st.write("• Clusters group ATMs by demand behavior.")
st.write("• High withdrawal variance indicates demand spikes.")
st.write("• Holiday flags show significant impact on cash demand.")
st.write("• Anomalies help detect unusual spikes or shortages.")

st.success("✅ App Running Successfully - FA2 Complete")

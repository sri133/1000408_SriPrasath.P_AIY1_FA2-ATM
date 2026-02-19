# -------------------------------
# ATM Demand Forecasting Script
# FA‑2 (Data Mining)
# -------------------------------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# For anomaly detection
from scipy import stats

# Improve plot styles
plt.style.use('seaborn')
sns.set_context('talk')

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
url = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_AIY1_FA2-ATM/main/atm_cash_management_dataset.csv"
df = pd.read_csv(url)

print("\nDataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# -------------------------------
# 2️⃣ Data Preprocessing
# -------------------------------

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract useful features
df['Month'] = df['Date'].dt.month
df['Week_Number'] = df['Date'].dt.isocalendar().week
df['Day_of_Week'] = df['Date'].dt.day_name()

# Encode categorical features
df['Day_of_Week_encoded'] = df['Day_of_Week'].map({
    'Monday': 1,'Tuesday': 2,'Wednesday': 3,
    'Thursday': 4,'Friday': 5,'Saturday': 6,'Sunday': 7
})

# Example Time_of_Day mapping
df['Time_of_Day_encoded'] = df['Time_of_Day'].map({
    'Morning': 1,'Afternoon':2,'Evening':3,'Night':4
})

# Encode flags
df['Holiday_Flag'] = df['Holiday_Flag'].astype(int)
df['Special_Event_Flag'] = df['Special_Event_Flag'].astype(int)

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Print clean summary
print("\nAfter Preprocessing:")
print(df.describe())

# -------------------------------
# 3️⃣ Exploratory Data Analysis (EDA)
# -------------------------------

# Distribution Analysis
plt.figure(figsize=(10, 5))
sns.histplot(df['Total_Withdrawals'], kde=True, color='blue')
plt.title("Distribution of Total Withdrawals")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['Total_Deposits'], kde=True, color='green')
plt.title("Distribution of Total Deposits")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Total_Withdrawals'])
plt.title("Boxplot: Withdrawals Outliers")
plt.show()

print("Observation: Histogram shows central tendency and spread of withdrawals.")

# Time-Based Trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Total_Withdrawals')
plt.title("Withdrawals Over Time")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Day_of_Week', y='Total_Withdrawals', data=df)
plt.title("Avg Withdrawals By Day of Week")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Time_of_Day', y='Total_Withdrawals', data=df)
plt.title("Avg Withdrawals By Time of Day")
plt.show()

# Holiday & Event Impact
plt.figure(figsize=(8,5))
sns.barplot(x='Holiday_Flag', y='Total_Withdrawals', data=df)
plt.title("Withdrawals on Holiday vs Normal")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Special_Event_Flag', y='Total_Withdrawals', data=df)
plt.title("Withdrawals on Special Event Days")
plt.show()

# External Factors
plt.figure(figsize=(8,5))
sns.boxplot(x='Weather_Condition', y='Total_Withdrawals', data=df)
plt.title("Withdrawals vs Weather Condition")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Nearby_Competitor_ATMs', y='Total_Withdrawals', data=df)
plt.title("Effect of Nearby Competitor ATMs on Withdrawals")
plt.show()

# Relationship Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Previous_Day_Cash_Level', y='Cash_Demand_Next_Day', data=df)
plt.title("Previous Day Cash vs Next Day Demand")
plt.show()

# -------------------------------
# 4️⃣ Clustering Analysis
# -------------------------------

features = df[['Total_Withdrawals','Total_Deposits','Location_Type','Nearby_Competitor_ATMs']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow method to decide clusters
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(2,7), inertia, marker='o')
plt.title("Elbow Method for K‑Means")
plt.show()

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled_features)
df['Cluster_Label'] = kmeans.labels_

# Silhouette Score
print("Silhouette Score:", silhouette_score(scaled_features, df['Cluster_Label']))

sns.pairplot(df, hue='Cluster_Label', vars=['Total_Withdrawals','Total_Deposits'])
plt.suptitle("Cluster visualization")
plt.show()

# Assign cluster meaning
df['Cluster_Type'] = df['Cluster_Label'].map({
    0: 'Steady-Demand', 1:'High-Demand', 2:'Low-Demand'
})

# -------------------------------
# 5️⃣ Anomaly Detection
# -------------------------------

# Z-score method
df['zscore_withdrawals'] = np.abs(stats.zscore(df['Total_Withdrawals']))
anomalies_z = df[df['zscore_withdrawals'] > 3]

# IQR method
Q1 = df['Total_Withdrawals'].quantile(0.25)
Q3 = df['Total_Withdrawals'].quantile(0.75)
IQR = Q3 - Q1
anomalies_iqr = df[(df['Total_Withdrawals'] < Q1 - 1.5*IQR) | (df['Total_Withdrawals'] > Q3 + 1.5*IQR)]

print("Z-score Anomalies Count:", anomalies_z.shape[0])
print("IQR Anomalies Count:", anomalies_iqr.shape[0])

# Visualize anomalies
plt.figure(figsize=(12,6))
plt.scatter(df['Date'], df['Total_Withdrawals'], label='Normal')
plt.scatter(anomalies_iqr['Date'], anomalies_iqr['Total_Withdrawals'], color='red', label='Anomaly')
plt.title("Anomalies in Withdrawals (IQR)")
plt.legend()
plt.show()

# -------------------------------
# 6️⃣ Interactive Planner / Filter Examples
# -------------------------------

# Example: Weekends
print("\nFilter Example: ATMs on Weekends")
weekend_data = df[df['Day_of_Week'].isin(['Saturday','Sunday'])]
print(weekend_data[['Date','Total_Withdrawals','Cluster_Type']].head())

# Example: Urban ATMs only
print("\nFilter Example: Urban ATMs only")
urban_data = df[df['Location_Type']==1]
print(urban_data[['Date','Total_Withdrawals','Cluster_Type']].head())

print("\nScript execution completed!")

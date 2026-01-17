import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import preprocess_data

# Load + basic cleaning (DateTime merge, -200 -> NaN, decimals)
# For EDA we keep NaNs so we can visualise missingness.
df = preprocess_data(
    path="air+quality/AirQualityUCI.xlsx",
    missing_strategy="none",
    add_time_features_flag=False,
    normalize=False,
)

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe().T)

# Missing values heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df.isna(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

pollutants = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

# Time-series of pollutant concentrations
plt.figure(figsize=(14, 6))
for p in pollutants:
    plt.plot(df['DateTime'], df[p], label=p)
plt.legend()
plt.title("Pollutant Concentrations Over Time")
plt.xlabel("Date")
plt.ylabel("Concentration (µg/m³)")
plt.show()

# Hour-of-day patterns
df['hour'] = df['DateTime'].dt.hour
hourly_means = df.groupby('hour')[pollutants].mean()

plt.figure(figsize=(10, 6))
sns.lineplot(data=hourly_means)
plt.title("Average Hourly Concentrations")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Concentration")
plt.show()

# Weekday patterns (use names for nicer plots)
df['weekday'] = df['DateTime'].dt.day_name()
weekly_means = df.groupby('weekday')[pollutants].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

weekly_means.plot(kind='bar', figsize=(10, 5))
plt.title("Average Pollutant Levels by Day of Week")
plt.ylabel("Mean Concentration")
plt.xticks(rotation=45)
plt.show()

# Correlation between pollutants and meteorology
corr = df[pollutants + ['T', 'RH', 'AH']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Pollutants and Meteorological Variables")
plt.show()

# Pairwise relationships
sns.pairplot(
    df[['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']].dropna()
)
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

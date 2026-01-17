# Air Quality Forecasting Project  
COMP9417 • Term 3, 2025  
**Authors:** Aayush Veturi • Arnav Raina • Neha Gajendra • Vansh Kalra

---

## Overview
This project develops machine-learning models to forecast air pollutant concentrations (regression) and CO-level categories (classification) using the **UCI Air Quality Dataset**.  
The pipeline includes:

- Exploratory data analysis (EDA)
- Temporal feature engineering
- Multihorizon forecasting (1 h, 6 h, 12 h, 24 h)
- Regression + classification models
- Baseline comparison
- Full evaluation and interpretation

The dataset contains **9,358 hourly measurements** (2004–2005) covering pollutants (CO, NO₂, NOx, C6H6), sensor responses, and meteorological variables.

---

## Project Objectives
- Understand pollutant dynamics using statistical EDA.
- Engineer features capturing temporal structure (lags, rolling means, cyclical encodings).
- Train and compare models for short- and long-horizon forecasting.
- Evaluate regression models using **RMSE** and classification models using **Accuracy + F1**.
- Benchmark all models against a **naïve persistence baseline**.
- Identify limitations and propose improvements.


## Data Preprocessing
- Sensor error markers (−200) replaced with `NaN`.
- Missing values filled using **time-based interpolation**.
- Combined `Date` + `Time` into a **DateTime index**.
- Extracted hour, weekday, and month.
- Anomaly analysis performed, but anomalies were **not removed**.

---

## Feature Engineering

### **1. Cyclical Time Features**
Sine/cosine encodings of hour and month to preserve periodicity observed in the EDA  
(diurnal and seasonal cycles).

### **2. Lag Features**
1 h, 2 h, and 24 h lags for pollutants + meteorological variables  
(capture short-term persistence and daily recurrence).

### **3. Rolling Statistics**
6 h, 12 h, and 24 h rolling averages  
(reduce noise + capture short-term trends).

---

## Modelling

### **Regression Models**
- Linear Regression (LR)
- Gradient Boosting Regressor (GBR)

**Evaluation Metric:** RMSE  
**Baseline:** Naïve persistence model (predict next = current)

### **Classification Models**
- Logistic Regression (multinomial, L2-regularised)
- Random Forest Classifier

**Evaluation Metrics:** Accuracy, Weighted/Macro F1  
**Outputs:** Confusion matrices for t+1, t+6, t+12, t+24  
**Baseline:** Persistence of CO class

---

## Key Results

### **Regression**
- Naïve performs best at **1 h** due to strong autocorrelation.
- At **6–24 h**, naïve performance drops sharply.
- **GBR outperforms LR** in most pollutant–horizon combinations (9/16 wins).
- LR struggles with multicollinearity and nonlinear pollutant spikes.

### **Classification**
- Naïve best at **1 h** (Accuracy: 0.748).
- **Random Forest strongest at 6–12 h** (0.577 and 0.535).
- Logistic Regression collapses toward “Low” at long horizons.
- At 24 h, both models approach the naïve baseline.
- Unfiltered anomalies likely increased Medium/High confusion.

---

## Discussion Highlights
- Short-horizon forecasts reflect strong pollutant persistence.
- Nonlinear models (GBR, RF) capture spikes + interactions more effectively.
- Medium/long horizons show degradation due to weaker temporal signal.
- EDA anomaly clusters contributed to instability in classification boundaries.
- Missing environmental drivers (wind, traffic, regional movement) limit accuracy for 12–24 h forecasting.

---

## Limitations
- NMHC excluded due to severe missingness.
- No anomaly filtering applied.
- Missing external features (wind, traffic, humidity variation).
- Linear models affected by multicollinearity.

---

## Future Work
- Apply anomaly filtering.
- Integrate weather + traffic datasets.
- Use dimensionality reduction or regularised linear models

---

## How to Run the Models

### **Install Dependencies**
pip install numpy pandas scikit-learn matplotlib seaborn joblib

### **Run Regression Models**
python src/model_linear_regression.py

python src/model_gradient_boosting.py

### **Run Classification Models**
python src/model_logistic_regression.py

python src/model_random_forest.py

**Full Report:**  
[COMP9417 Group Assignment — Final Report](./Report.pdf)

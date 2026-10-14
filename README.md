# Inventory Demand Forecasting using Machine Learning

An end-to-end Machine Learning project designed to predict stock requirements for various products across multiple stores, helping vendors optimize inventory levels, prevent shortages, and avoid excess storage costs.

---

## Project Overview

Maintaining the right stock levels is crucial for retail businesses to satisfy customer demand without overstocking. This project utilizes historical sales data spanning 10 stores and 50 products over five years to build, train, and evaluate a robust demand forecasting pipeline in Python.

---

## Project Workflow

### 1. Importing Libraries & Dataset

* **Libraries Used:** Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, and the `holidays` library.
* **Dataset:** Historical sales data (`StoreDemand.csv`) containing 913,000 records across 4 initial columns with no missing values.

### 2. Exploratory Data Analysis (EDA) & Inspection

* Inspected structure (`df.info()`) and descriptive statistics (`df.describe()`).
* Identified 10 unique stores and 50 unique products.
* Analyzed sales distributions using distribution and box plots, discovering a right-skewed target distribution and outliers (filtered out sales $> 140$).

### 3. Feature Engineering

Extracted meaningful temporal features from the raw date column to significantly boost model accuracy:

* **Date Parsing:** Extracted `year`, `month`, and `day`.
* **Calendar Flags:** Created `weekend` and `weekday` indicators.
* **Holiday Mapping:** Integrated country-specific holiday flags using the `holidays` Python package.
* **Cyclical Encoding:** Added sine and cosine transformations (`m1`, `m2`) to capture seasonal monthly trends.
* Cleaned up the raw `date` column after feature extraction.

### 4. Data Visualization & Insights

* **Bar Charts:** Visualized average sales trends grouped by stores, years, months, and weekdays.
* **Line Plots:** Tracked stock variation across days of the month.
* **Moving Averages:** Implemented a 30-day Simple Moving Average (SMA) to smooth short-term fluctuations and highlight long-term trends.
* **Correlation Heatmaps:** Inspected feature relationships using a correlation matrix filter.

### 5. Model Training & Evaluation

* **Data Splitting:** Separated features and target variables, splitting the dataset into training (95%) and validation (5%) sets.
* **Normalization:** Applied `StandardScaler` on features for stable and fast model convergence.
* **Models Tested:**
* Linear Regression
* Lasso Regression
* Ridge Regression
* XGBoost Regressor


* **Performance Metric:** Evaluated models using Mean Absolute Error (MAE) on both training and validation sets.

---

## Key Results

* **Best Performing Model:** **XGBoost Regressor** achieved the lowest validation error.
* **Conclusion:** Ensemble tree-based methods like XGBoost are highly effective at capturing complex, non-linear patterns in retail sales data compared to traditional linear models.

---

## Getting Started

### Prerequisites & Dependencies

Install the required Python libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost holidays

```

### Running the Project

1. Place the dataset (`StoreDemand.csv`) in your project root directory.
2. Run the preprocessing, feature engineering, and model training script sequentially to replicate the pipeline and evaluate performance metrics.
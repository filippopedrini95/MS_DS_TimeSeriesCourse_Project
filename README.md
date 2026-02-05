# MS_DS_TimeSeriesCourse_Project

# 1. Project Description
This project focuses on forecasting daily grocery sales demand using real-world retail data from Ecuador. The goal is to build, evaluate, and compare multiple time series forecasting models to predict future unit sales at a regional level.

The project mirrors a realistic retail forecasting scenario, similar to those used in inventory planning, demand management, and operational decision-making. It follows a complete end-to-end workflow, from data exploration and feature engineering to model evaluation and comparison.

**Business Problem**\
Retail organizations need accurate demand forecasts to optimize inventory levels, reduce waste and stockouts, and improve operational efficiency. In this project, the task is to predict daily unit sales aggregated across all stores and products in a specific region, using historical sales data.

# 2. Problem Context
**Why Sales Forecasting Matters**\
Accurate sales forecasting is a critical capability for retail businesses. Reliable demand predictions enable better inventory planning, reduce excess stock and lost sales due to stockouts, and support decisions related to promotions, staffing, and logistics.

**Business Impact**\
Poor forecasts can lead to overstocking, increased waste, missed revenue opportunities, and inefficient operations. Conversely, accurate forecasts improve supply chain efficiency, reduce operational costs, and support more informed business decisions.


# 3. Data Overview

**Dataset Description**\
The analysis is based on the Corporación Favorita Grocery Sales Forecasting dataset. The original dataset contains several million records, with each row representing the daily quantity sold of a specific item in a specific store across Ecuador between 2013 and 2017.
Due to the size of the original dataset, a pre-filtered and prepared version was provided for this project (train2.csv). The filtering process was performed prior to the analysis and includes the following constraints:\
- Region: Guayas only
- Product families: Top 3 families by number of unique items (GROCERY I, BEVERAGES, CLEANING)

**Time Range**\
January 1, 2013 – March 31, 2014 (inclusive)

**Target Variable**\
unit_sales: daily quantity of products sold\
For this project, sales were aggregated, without distinguishing between individual stores or product families. The forecasting task therefore focuses on predicting total daily demand across the entire region and selected product categories.


**Additional Datasets**\
In addition to the main sales dataset, several auxiliary datasets were provided to enrich the analysis with contextual and external information relevant for demand forecasting. 
The additional datasets include:
- holidays_events.csv contains detailed information on national and local holidays and events across Ecuador.
- items.csv provides metadata about the products sold, such as product family and item-level characteristics.
- oil.csv contains the daily oil price in Ecuador.
- stores.csv includes descriptive information about individual stores, such as location and store type
- transactions.csv reports the total number of daily transactions per store.


# 4. Methodology

**Overall Approach**\
The project was developed using the Databricks platform, where all datasets were loaded from CSV files and stored as Spark (Delta) tables. The analysis followed a structured time series forecasting workflow:
- Initial exploratory data analysis (EDA) to understand overall demand patterns
- Visualization of daily and weekly sales dynamics
- Assessment of calendar effects (weekdays, weekends, holidays)
- Training and evaluation of multiple forecasting models
- Comparison of model performance using consistent evaluation metrics


Sales data were aggregated at the daily level, across all stores and product families within the selected region.

**Exploratory Data Analysis**\
The EDA phase focused on understanding the temporal behavior of daily unit sales:
- Visualization of unit sales over time, both overall and split by the top 3 product families
- Weekly sales heatmap to highlight recurring weekly patterns and seasonality
- Preliminary analysis of holiday impact by comparing average unit sales across:
Holidays,
Weekdays,
Weekends,

## **Models Tested**
Three different forecasting approaches were implemented and evaluated, each in a separate Databricks notebook:
### 1. Exponential Smoothing (Holt-Winters)
Classical time series modeling was performed using the statsmodels library:\
. Models tested: Holt-Winters (triple exponential smoothing)\
. A seasonal period of 7 days was used to capture weekly seasonality\
. Both additive and multiplicative seasonality were evaluated\
. Models were tested with and without trend damping


While additive and multiplicative seasonality showed similar performance, trend damping consistently improved forecast accuracy and was therefore retained. Additive seasonality was eventually chosen in the final model.

### 2. Prophet
The Prophet model was implemented following its required data format and conventions.
The modeling process followed a stepwise approach:\
. The model was first evaluated using Prophet’s default configuration, which includes built-in weekly and yearly seasonalities.\
. Seasonalities were then defined manually, testing different combinations of:
Weekly seasonality,
Monthly seasonality,
Yearly seasonality,
. For each seasonal component, multiple Fourier orders were evaluated in order to control model flexibility and avoid overfitting.

Through this process, the best-performing configuration was found to include weekly and monthly seasonalities, while the yearly component did not lead to further improvements.

Holiday effects were also evaluated by incorporating calendar information from the holidays_events dataset. However, models including holidays consistently showed worse performance on the test set. As a result, holiday effects were excluded from the final Prophet model.

Model selection was performed by training on the training set and evaluating forecasts on a three-month holdout test set. Final parameters were chosen based on comparative performance across both training and test data.


### 3. XGBoost (Recursive Forecasting)
A machine learning approach was implemented using XGBoost in combination with the skforecast library, which enables recursive multi-step forecasting.
Key aspects of the approach include:
- Automatic handling of:
Lagged features, 
Rolling statistics
- Recursive forecasting to update lag and rolling features during prediction
- Lag selection optimized during hyperparameter tuning

The best-performing lag configuration was:
- Lags: [1, 7, 30, 90, 120]
- Rolling statistics: rolling mean (7), rolling standard deviation (7)


Additional time-based features were manually engineered, including:
- Day index
- Cyclical encodings using sine and cosine transformations for:
Month,
Weekday,
Day of year
- Binary weekend indicator
- Temporal index


Daily store transactions were tested as an external regressor. However, since their inclusion degraded performance on the test set, they were excluded from the final model.
Hyperparameter tuning was performed without cross-validation, by training the model on the training set and evaluating its performance on both the training and test sets. During this phase, true historical values were used to compute lagged features and rolling statistics for the test period. This choice allowed the model to generate predictions for both the training and test sets under comparable conditions, enabling a fair and transparent comparison of model performance and isolating the model’s ability to learn underlying patterns from the compounding effects of recursive errors.
Once the optimal hyperparameters and lag configuration were identified, the final model was evaluated under a fully realistic forecasting setup, using recursive multi-step forecasting on the test set. In this final evaluation, lagged and rolling features were generated exclusively from the model’s own predictions, reflecting a true deployment scenario.

### Evaluation Metrics
Model performance was evaluated using multiple complementary metrics:
- R2 – goodness of fit
- MAE (Mean Absolute Error) – average absolute prediction error
- RMSE (Root Mean Squared Error) – penalizes larger errors more strongly


These metrics allowed for a balanced comparison between overall fit and forecast accuracy, particularly on the holdout test set.


# 5. Model Comparison

**Performance Summary**\
The table below summarizes the performance of the three forecasting models evaluated in this project, using consistent metrics and the same train/test split.\

Model_______ R²______ RMSE_____MAE______Training Time__Prediction Time\
Holt-Winters_ 0.4333__ 9086.09__ 6509.36___ 79.12ms______4.00ms\
Prophet_____ 0.5036__ 8504.32__ 5585.90___ 208.81ms_____73.28ms\
XGBoost_____0.2169__ 10680.99__7813.88___ 54.59ms______40.94ms

**Model-Level Observations**\
The three models exhibit markedly different trade-offs in terms of accuracy, computational cost, and modeling assumptions.
- **XGBoost** achieved the weakest performance across all evaluation metrics. Despite its flexibility and strong performance in many regression problems, the recursive multi-step setup and limited data horizon likely made it difficult for the model to capture the underlying temporal structure effectively. This highlights the challenges of applying general-purpose machine learning models to pure time series forecasting tasks without extensive feature engineering or longer histories.

- **Holt-Winters (Exponential Smoothing)** delivered solid performance with extremely fast training and prediction times. Its simplicity and efficiency make it well-suited for scenarios where computational speed and interpretability are prioritized. However, its limited flexibility restricts its ability to capture more complex temporal patterns beyond trend and seasonality.

- **Prophet** achieved the strongest overall performance in terms of accuracy, at the cost of longer training and tuning times. Its ability to model multiple seasonal components explicitly and incorporate calendar effects makes it particularly effective for retail demand forecasting.


**Best Performing Model: Prophet**\
Based on the evaluation results, Prophet was selected as the best-performing model for this forecasting task.
Key reasons for this choice include:
- Best performance across all evaluation metrics (R², RMSE, MAE)
- A time-series–specific modeling framework that is:
simpler to use than machine learning approaches such as XGBoost, 
more flexible than classical exponential smoothing methods
- Rich and interpretable outputs, including:
decomposed trend and seasonal components,
prediction intervals and uncertainty estimates

Overall, Prophet offers the most favorable balance between predictive accuracy, interpretability, and modeling flexibility for this specific problem setting.



# 6. Setup Instructions
Development Environment
This project was developed and executed using the Databricks platform, leveraging Apache Spark for data handling and scalable computation. All analyses and model training were performed within Databricks notebooks.

**Repository Setup**\
To clone the repository locally:\
git clone \<repository-url>\
cd \<repository-name>

**Dependencies**\
All required Python libraries and their corresponding versions are listed in the requirements.txt file.\
Dependencies can be installed using:\
pip install -r requirements.txt

**Data Access and Format**\
All datasets used in this project are located in the data/ directory and are provided in CSV format.
These datasets include:
- Sales data
- Holiday and event information
- Product metadata
- Store-level information
- Daily transaction counts

**Data Loading in Databricks**\
Although the datasets are stored in the repository as CSV files, the project workflow does not operate directly on CSV files.
At the beginning of the analysis, all datasets were:
- Loaded from CSV files
- Saved as Spark tables / Delta tables
- Subsequently accessed exclusively from these tables throughout the project
This approach ensures better performance, consistency, and reproducibility when working within the Databricks environment.

All subsequent data processing, feature engineering, and modeling steps rely on these Spark/Delta tables rather than reloading raw CSV files.

**Running the Notebooks**\
The project is organized into multiple Databricks notebooks, each corresponding to a specific stage of the workflow (EDA, model training, evaluation).
Notebooks can be executed sequentially within the Databricks workspace after ensuring that:
- All required libraries are installed
- The Spark tables have been created successfully


# 7. How to Use
**Loading the Trained Models**\
All trained models were saved using the .joblib format. To load and use them, the joblib library must be installed and imported:\
import joblib\
Each model can then be loaded and used independently, as described below.

### Holt-Winters (Exponential Smoothing)
hw_model = joblib.load("holt_winters_model.joblib")\
pred = hw_model.forecast(steps=90)

steps: number of future time steps to forecast.
In this case, steps=90 generates predictions for the next 90 days after the end of the training data.\
The output pred is a sequence of forecasted daily unit sales values.

### Prophet
prophet_model = joblib.load("prophet_model.joblib")\
future = prophet_model.make_future_dataframe(periods=90, freq="D")\
forecast = prophet_model.predict(future)

periods: number of future periods to forecast (e.g. 90 days)\
freq: frequency of the time series\
"D" indicates daily frequency

The forecast object is a DataFrame containing:
- point forecasts (yhat)
- uncertainty intervals (yhat_lower, yhat_upper)
- decomposed components such as trend and seasonality

### XGBoost (Recursive Forecasting)
forecaster = joblib.load("xgb_forecaster.joblib")\
pred = forecaster.predict(steps=90, exog=exog_future)

steps: number of future time steps to predict\
exog: DataFrame containing future values of exogenous features

The XGBoost model uses a recursive multi-step forecasting strategy, meaning that predictions are generated step by step and fed back into the model to compute future lagged and rolling features.

**Exogenous Features for XGBoost**\
The exog_future DataFrame must:
- Have dates as index
- Contain the same engineered features used during training (described in the Methodology section)

Example structure of exog_future:

\<class 'pandas.core.frame.DataFrame'>\
DatetimeIndex: 454 entries, 2013-01-02 to 2014-03-31\
Data columns (total 10 columns):\
   Column        Non-Null Count  Dtype\
---  ------        --------------  ----- \
 0   day           454 non-null    int32\
 1   month_sin     454 non-null    float64\
 2   month_cos     454 non-null    float64\
 3   weekday_sin   454 non-null    float64\
 4   weekday_cos   454 non-null    float64\
 5   doy_sin       454 non-null    float64\
 6   doy_cos       454 non-null    float64\
 7   temporal_idx  454 non-null    int64\
 8   is_weekend    454 non-null    int64\
 9   transactions  454 non-null    float64\
dtypes: float64(7), int32(1), int64(2)

These features include calendar-based encodings (cyclical sine/cosine transformations), temporal indices, and optional external regressors. Only features present during training should be provided at prediction time.

**Expected Outputs**
- Holt-Winters: array-like structure of point forecasts
- Prophet: DataFrame with forecasts, confidence intervals, and components
- XGBoost: array-like structure of predicted values generated recursively

All outputs represent daily unit sales forecasts for the specified forecast horizon.


# 8. Results and Findings
The analysis highlights clear differences between the tested forecasting approaches. Prophet emerges as the best-performing model across all evaluation metrics, while Holt-Winters offers a strong trade-off between simplicity and computational efficiency. XGBoost, despite its flexibility, underperforms in this time-series setting.

Overall, time-series–specific models prove more effective for this task. Based on these results, Prophet is recommended as the final model when accuracy and interpretability are the primary objectives, while Holt-Winters may be preferred in scenarios where speed and simplicity are critical.

# 9. Future Improvements
Several directions could be explored to further improve the forecasting performance.

For Prophet, additional work could focus on a deeper analysis of the trend component. Prophet models the trend as a piecewise linear or logistic function with automatically detected changepoints; tuning the flexibility of this component (for example by adjusting the number or impact of changepoints) could lead to better adaptation to structural changes in the data. It would also be worth testing multiplicative seasonality, which may better capture cases where seasonal effects scale with the level of the series. Finally, further optimization of the prior scale parameters (such as trend, seasonality, and changepoint prior scales) could help balance model flexibility and overfitting.

Regarding XGBoost, performance could potentially be improved by engineering additional exogenous features and testing alternative feature representations. Beyond this, a more extensive hyperparameter search and different lag configurations could be explored to better capture temporal dependencies.

More generally, other forecasting approaches could be investigated, such as ARIMA-family models for a classical statistical baseline, or neural network–based models like RNNs or LSTMs, which may be better suited to learning complex non-linear patterns in longer time series.


# 10. Repository Structure
The project repository is organized as follows:
- notebooks/\
Contains all Databricks notebooks used for data exploration, preprocessing, modeling, and evaluation.
- visualizations/\
Stores the main plots and figures generated during the analysis, saved in .png format.
- models/\
Contains the trained forecasting models saved in .joblib format, ready to be loaded and reused for inference.
- data/\
Includes the datasets used in the project, stored as .csv files.

At the root level of the repository:
- README.md\
Provides an overview of the project, its structure, and instructions for usage.
- requirements.txt\
Lists all required Python libraries and their versions, ensuring reproducibility of the environment.

# 11. Contact
Author: Filippo Pedrini\
Date Completed: February 5, 2026\
Contact: filippopedrini95@gmail.com


# Groundwater Level Prediction: MLR vs. ANN

This project conducts a comparative analysis between a Multiple Linear Regression (MLR) model and an Artificial Neural Network (ANN) to predict monthly groundwater levels (GWL) in the districts of a specific region. The analysis finds that the ANN model significantly outperforms the traditional MLR model in accuracy and predictive power.

-----

## Key Findings 📈

  * **ANN is More Accurate:** The Artificial Neural Network (ANN) demonstrated superior performance across all evaluation metrics, achieving an **R² of 0.56** compared to the MLR model's **R² of 0.36**.
  * **Significant Improvement:** The ANN's lower prediction error (RMSE of 4.25 vs. MLR's 5.11) is statistically significant, confirmed by a paired t-test ($p < 0.001$).
  * **Important Predictors:** Both models identified the **time index** and the **previous month's groundwater level** (`GWL_lag1`) as the most influential factors in predicting current GWL.
  * **MLR Limitations:** The dataset violates several key assumptions of linear regression, including linearity and homoscedasticity, making the ANN a more suitable choice.

-----

## Dataset 💾

The analysis uses the `final_dataset.csv`, which contains time-series data aggregated by district and month.

  * **Dependent Variable**: `GWL_average` (Average Groundwater Level).
  * **Predictors (Covariates)**:
      * **Climate**: `RF_average` (Rainfall), `t2m_value` (Temperature), `tp_value` (Total Precipitation).
      * **Soil**: `swvl1_value` (Soil Moisture Volumetric Layer 1), `swvl2_value` (Soil Moisture Volumetric Layer 2), `sand_pct`, `floamy_pct`, `f_clayey_pct`.
      * **Temporal**: `time_index` (a monthly counter) and `GWL_lag1` (GWL from the previous month).

-----

## Methodology ⚙️

The project is divided into two main modeling approaches.

### 1\. Multiple Linear Regression (MLR)

A traditional statistical approach was used to establish a baseline linear model.

1.  **Exploratory Data Analysis (EDA)**: Summary statistics and a correlation heatmap were generated to understand the data. The heatmap revealed high multicollinearity between soil moisture (`swvl1`, `swvl2`) and climate variables.
2.  **Assumption Testing**: The model was tested for MLR assumptions. The results indicated significant violations:
      * **Multicollinearity**: Extremely high Variance Inflation Factor (VIF) scores for soil moisture variables (`swvl1_value` \> 200, `swvl2_value` \> 170) confirmed severe multicollinearity.
      * **Homoscedasticity**: The Breusch-Pagan test yielded a very low p-value, rejecting the assumption of constant error variance.
      * **Normality of Residuals**: The Jarque-Bera test also had a low p-value, indicating that the model's errors were not normally distributed.
3.  **Model Selection**: Three different MLR models were compared using AIC/BIC scores. The "full model" using all predictors was found to be the best.
4.  **Prediction**: The final model was trained and evaluated on a test set, yielding an R² of **0.36**.

### 2\. Artificial Neural Network (ANN)

A Multi-Layer Perceptron (MLP) was built to capture more complex, non-linear relationships in the data.

1.  **Data Scaling**: Predictor variables were scaled to a range of [0, 1] using `MinMaxScaler`, which is crucial for ANN performance.
2.  **Model Architecture**: A sequential model was constructed with two hidden `Dense` layers (128 and 64 neurons) using the 'relu' activation function. `Dropout` layers (rate=0.2) were added for regularization to prevent overfitting.
3.  **Training**: The model was trained using the `Adam` optimizer to minimize Mean Squared Error (MSE). `EarlyStopping` was used to prevent overfitting by stopping the training when validation loss stopped improving.
4.  **Explainability**: To understand the "black box" ANN model, **Permutation Importance** and **SHAP (SHapley Additive exPlanations)** analyses were conducted. Both methods confirmed that `time_index` and `GWL_lag1` were the most significant predictors.

-----

## Results and Comparison 📊

The ANN model provided a substantial improvement in predictive accuracy compared to the MLR model.

| Model | RMSE | MAE | R² | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **MLR** | 5.1102 | 3.7764 | 0.3630 | - |
| **ANN (MLP)** | 4.2502 | 2.3619 | 0.5593 | 2.34 |

A **paired t-test** on the absolute errors of the two models resulted in a p-value of **0.0000**, confirming that the ANN's superior performance is statistically significant.

### Model Explainability (SHAP)

The SHAP summary plot provides insights into the ANN's decision-making.

  * **`time_index`**: Has the largest impact. High values (later dates, shown in red) have a strong negative influence on the predicted GWL, confirming a declining trend over time.
  * **`GWL_lag1`**: The second most important feature. High values from the previous month (red) have a positive impact on the current month's prediction.
  * **Soil Composition**: `floamy_pct` and `sand_pct` also show a significant negative impact on GWL predictions.

-----

## How to Use 🚀

1.  **Prerequisites**: Ensure you have Python installed with the necessary libraries. You can install them using pip:
    ```bash
    pip install numpy pandas matplotlib seaborn statsmodels scikit-learn tensorflow shap
    ```
2.  **Dataset**: Place your `final_dataset.csv` file in the same directory as the notebook.
3.  **Run Notebook**: Open and run the `MLR_vs_MLP_GLW.ipynb` notebook in a Jupyter or Colab environment. The notebook is structured sequentially and will execute the entire analysis from data loading to model comparison.

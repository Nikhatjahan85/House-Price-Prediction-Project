import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import os

# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

# Replace with your dataset path
df = pd.read_csv("data/housing.csv")

print("\nDataset Loaded Successfully\n")

# =====================================================
# DATASET PREVIEW
# =====================================================

print("First 5 Rows:\n")
print(df.head())

# Save dataset preview as image
plt.figure(figsize=(10, 4))
plt.axis('off')
table = plt.table(
    cellText=df.head().values,
    colLabels=df.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)

plt.savefig("images/dataset_preview.png", bbox_inches='tight')
plt.close()

# =====================================================
# DATA CLEANING
# =====================================================

# Fill missing numerical values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col]= df[col].fillna(df[col].median())

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

print("\nData Cleaning Completed\n")

# =====================================================
# CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.savefig("images/heatmap.png")
plt.close()

print("Heatmap Saved")

# =====================================================
# TARGET VARIABLE
# =====================================================

# Replace 'price' with your actual target column
target_column = "price"

X = df.drop(target_column, axis=1)
y = df[target_column]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Test Split Completed\n")

# =====================================================
# LINEAR REGRESSION MODEL
# =====================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(name, y_true, predictions):

    mae = mean_absolute_error(y_true, predictions)

    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    r2 = r2_score(y_true, predictions)

    print(f"\n{name} Results")
    print("-" * 30)

    print(f"MAE  : {mae}")
    print(f"RMSE : {rmse}")
    print(f"R2   : {r2}")

    return mae, rmse, r2

# =====================================================
# EVALUATE MODELS
# =====================================================

linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

# =====================================================
# SAVE EVALUATION RESULTS
# =====================================================

with open("outputs/evaluation_results.txt", "w") as file:

    file.write("MODEL EVALUATION RESULTS\n\n")

    file.write("Linear Regression\n")
    file.write(f"MAE  : {linear_results[0]}\n")
    file.write(f"RMSE : {linear_results[1]}\n")
    file.write(f"R2   : {linear_results[2]}\n\n")

    file.write("Random Forest\n")
    file.write(f"MAE  : {rf_results[0]}\n")
    file.write(f"RMSE : {rf_results[1]}\n")
    file.write(f"R2   : {rf_results[2]}\n")

print("\nEvaluation Results Saved")

# =====================================================
# ACTUAL VS PREDICTED GRAPH
# =====================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, rf_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.savefig("outputs/prediction_graph.png")
plt.savefig("images/actual_vs_predicted.png")

plt.close()

print("Prediction Graph Saved")

# =====================================================
# FEATURE IMPORTANCE GRAPH
# =====================================================

importance = rf_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_features["Importance"],
    y=top_features["Feature"]
)

plt.title("Top 10 Important Features")

plt.savefig("images/feature_importance.png")

plt.close()

print("Feature Importance Graph Saved")

# =====================================================
# PRICE DISTRIBUTION GRAPH
# =====================================================

plt.figure(figsize=(8, 5))

sns.histplot(y, kde=True)

plt.title("House Price Distribution")

plt.savefig("images/price_distribution.png")

plt.close()

print("Price Distribution Graph Saved")

# =====================================================
# SAMPLE PREDICTION
# =====================================================

sample_house = X.iloc[[0]]

sample_prediction = rf_model.predict(sample_house)

print("\nSample House Predicted Price:")
print(f"₹ {sample_prediction[0]:,.2f}")

# =====================================================
# TERMINAL OUTPUT IMAGE
# =====================================================

output_text = f"""
MODEL RESULTS

Linear Regression
MAE  : {linear_results[0]}
RMSE : {linear_results[1]}
R2   : {linear_results[2]}

Random Forest
MAE  : {rf_results[0]}
RMSE : {rf_results[1]}
R2   : {rf_results[2]}

Sample Prediction:
₹ {sample_prediction[0]:,.2f}
"""

plt.figure(figsize=(10, 5))
plt.text(0.01, 0.5, output_text, fontsize=12)
plt.axis('off')

plt.savefig("images/terminal_output.png")

plt.close()

print("\nTerminal Output Image Saved")

# =====================================================
# PROJECT COMPLETED
# =====================================================

print("\nAll Images Generated Successfully")
print("\nCheck:")
print("1. outputs/ folder")
print("2. images/ folder")
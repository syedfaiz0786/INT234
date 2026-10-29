# ============================================================
# AIR QUALITY ANALYSIS AND POLLUTANT LEVEL PREDICTION
# USING MACHINE LEARNING
# ============================================================

# 1. IMPORT LIBRARIES

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 2. PROJECT TITLE AND OBJECTIVES
# ============================================================

print("=" * 70)
print("AIR QUALITY ANALYSIS AND POLLUTANT LEVEL PREDICTION")
print("USING MACHINE LEARNING")
print("=" * 70)

objectives = [
    "1. Analyze air pollution using exploratory data analysis.",
    "2. Clean the dataset and handle missing and invalid values.",
    "3. Visualize pollutant distributions and locations.",
    "4. Predict average pollutant measurements.",
    "5. Compare five regression algorithms.",
    "6. Identify the best-performing model.",
    "7. Evaluate results and summarize project findings."
]

print("\nPROJECT OBJECTIVES")
for objective in objectives:
    print(objective)


# ============================================================
# 3. LOAD DATASET
# ============================================================

# Change this path if your dataset is stored elsewhere.

file_path = (
    "/Users/syedfaizhusain/Documents/Python/"
    "5th sem/project/dataset.csv"
)

df = pd.read_csv(file_path)

print("\nDATASET OVERVIEW")
print("=" * 70)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

original_row_count = len(df)

# Remove duplicate records
df = df.drop_duplicates().copy()

# Required columns for prediction
required_columns = [
    "pollutant_min",
    "pollutant_max",
    "pollutant_avg"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

# Convert required columns to numeric
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinite values with missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows with missing model values
df.dropna(subset=required_columns, inplace=True)

# Remove negative measurements
df = df[
    (df["pollutant_min"] >= 0)
    & (df["pollutant_max"] >= 0)
    & (df["pollutant_avg"] >= 0)
].copy()

if df.empty:
    raise ValueError(
        "No valid records remain after cleaning. "
        "Please check your dataset."
    )

print("\nDATA CLEANING SUMMARY")
print("=" * 70)

print("Original rows:", original_row_count)
print("Rows after cleaning:", len(df))
print("Rows removed:", original_row_count - len(df))

print("\nMissing values after cleaning:")
print(df[required_columns].isnull().sum())

print("\nDuplicate rows after cleaning:", df.duplicated().sum())


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

sns.set_theme(style="whitegrid")

print("\nEXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(df.describe(include="all"))


# ------------------------------------------------------------
# EDA 1: NUMBER OF RECORDS BY POLLUTANT TYPE
# ------------------------------------------------------------

if "pollutant_id" in df.columns:

    pollutant_counts = df["pollutant_id"].value_counts()

    print("\nRecords by pollutant type:")
    print(pollutant_counts)

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        y="pollutant_id",
        order=pollutant_counts.index
    )

    plt.title("Number of Records by Pollutant Type")
    plt.xlabel("Number of Records")
    plt.ylabel("Pollutant Type")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# EDA 2: AVERAGE VALUES BY POLLUTANT TYPE
# ------------------------------------------------------------

if "pollutant_id" in df.columns:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="pollutant_id",
        y="pollutant_avg"
    )

    plt.title("Distribution of Average Values by Pollutant Type")
    plt.xlabel("Pollutant Type")
    plt.ylabel("Average Pollutant Measurement")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# EDA 3: TOP 10 CITIES BY MEAN POLLUTANT MEASUREMENT
# ------------------------------------------------------------

if "city" in df.columns:

    city_summary = (
        df.groupby("city")["pollutant_avg"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop 10 cities by mean pollutant measurement:")
    print(city_summary)

    plt.figure(figsize=(12, 6))

    city_summary.sort_values().plot(kind="barh")

    plt.title("Top 10 Cities by Mean Pollutant Measurement")
    plt.xlabel("Mean Pollutant Measurement")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# EDA 4: TOP 10 STATES BY MEAN POLLUTANT MEASUREMENT
# ------------------------------------------------------------

if "state" in df.columns:

    state_summary = (
        df.groupby("state")["pollutant_avg"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop 10 states by mean pollutant measurement:")
    print(state_summary)

    plt.figure(figsize=(12, 6))

    state_summary.sort_values().plot(kind="barh")

    plt.title("Top 10 States by Mean Pollutant Measurement")
    plt.xlabel("Mean Pollutant Measurement")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# EDA 5: DISTRIBUTION OF AVERAGE POLLUTANT MEASUREMENTS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="pollutant_avg",
    bins=30,
    kde=True
)

plt.title("Distribution of Average Pollutant Measurements")
plt.xlabel("Average Pollutant Measurement")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 6. FEATURE SELECTION
# ============================================================

# Input features
X = df[["pollutant_min", "pollutant_max"]]

# Target variable
y = df["pollutant_avg"]

print("\nFEATURE SELECTION")
print("=" * 70)

print("Input features:", X.columns.tolist())
print("Target variable: pollutant_avg")

print(
    "\nThe models estimate average pollutant measurements "
    "using minimum and maximum pollutant measurements."
)

print(
    "This is a regression task, not a future air-quality "
    "forecasting task."
)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 8. DEFINE MACHINE-LEARNING MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Polynomial Regression": make_pipeline(
        PolynomialFeatures(
            degree=2,
            include_bias=False
        ),
        StandardScaler(),
        LinearRegression()
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Support Vector Regression": make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf")
    ),

    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5)
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE ALL MODELS
# ============================================================

print("\nMODEL TRAINING AND EVALUATION")
print("=" * 70)

results = []
predictions = {}

# Classification metrics are calculated by converting continuous
# regression outputs into Low (0) and High (1) classes.
# The cutoff is calculated from the training target only.
classification_cutoff = float(y_train.median())

actual_classes = (
    np.asarray(y_test) >= classification_cutoff
).astype(int)

for model_name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Store predictions for visualizations
    predictions[model_name] = y_pred

    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Convert regression predictions into Low (0) / High (1) classes
    predicted_classes = (
        np.asarray(y_pred) >= classification_cutoff
    ).astype(int)

    # Classification metrics for this algorithm
    accuracy = accuracy_score(actual_classes, predicted_classes)
    precision = precision_score(
        actual_classes, predicted_classes, zero_division=0
    )
    recall = recall_score(
        actual_classes, predicted_classes, zero_division=0
    )
    f1 = f1_score(
        actual_classes, predicted_classes, zero_division=0
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\nModel: {model_name}")
    print("-" * 50)
    print(f"MAE       : {mae:.4f}")
    print(f"MSE       : {mse:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print(f"R2 Score  : {r2:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 10. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "Rank",
    range(1, len(results_df) + 1)
)

print("\nMODEL COMPARISON TABLE")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format="%.4f"
    )
)


# ------------------------------------------------------------
# VISUAL 1: R-SQUARED COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

sns.barplot(
    data=results_df,
    x="R2 Score",
    y="Model"
)

plt.title("Model Comparison: R-Squared Score")
plt.xlabel("R-Squared Score (Higher is Better)")
plt.ylabel("Machine Learning Model")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# VISUAL 2: RMSE COMPARISON
# ------------------------------------------------------------

rmse_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=rmse_df,
    x="RMSE",
    y="Model"
)

plt.title("Model Comparison: Root Mean Squared Error")
plt.xlabel("RMSE (Lower is Better)")
plt.ylabel("Machine Learning Model")
plt.tight_layout()
plt.show()


# ============================================================
# 11. SELECT THE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]
best_metrics = results_df.iloc[0]

print("\nBEST MODEL")
print("=" * 70)

print("Best model:", best_model_name)
print(f"MAE      : {best_metrics['MAE']:.4f}")
print(f"MSE      : {best_metrics['MSE']:.4f}")
print(f"RMSE     : {best_metrics['RMSE']:.4f}")
print(f"R2 Score : {best_metrics['R2 Score']:.4f}")


# Display actual and predicted values
comparison_df = pd.DataFrame({
    "Actual Value": np.asarray(y_test),
    "Predicted Value": np.asarray(best_predictions)
})

print("\nFirst 10 actual and predicted values:")
print(comparison_df.head(10).to_string(index=False))


# ============================================================
# 12. ACTUAL VS PREDICTED VISUALIZATION FOR ALL ALGORITHMS
# ============================================================

# Every algorithm gets its own graph.
# The diagonal line represents perfect predictions.

actual_values = np.asarray(y_test)

for model_name, y_pred in predictions.items():

    predicted_values = np.asarray(y_pred)

    min_value = min(
        float(np.min(actual_values)),
        float(np.min(predicted_values))
    )

    max_value = max(
        float(np.max(actual_values)),
        float(np.max(predicted_values))
    )

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=actual_values,
        y=predicted_values,
        alpha=0.6
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect Prediction Reference"
    )

    plt.title(f"Actual vs Predicted Values: {model_name}")
    plt.xlabel("Actual Pollutant Average")
    plt.ylabel("Predicted Pollutant Average")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 13. FIGURE: ACTUAL VS PREDICTED VALUES FOR THE BEST MODEL
# ============================================================

actual_values = np.asarray(y_test)
best_predicted_values = np.asarray(best_predictions)

min_value = min(
    float(np.min(actual_values)),
    float(np.min(best_predicted_values))
)

max_value = max(
    float(np.max(actual_values)),
    float(np.max(best_predicted_values))
)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=actual_values,
    y=best_predicted_values,
    alpha=0.6
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction Reference"
)

plt.title(
    f"Actual vs Predicted Values for Best Model: {best_model_name}"
)

plt.xlabel("Actual Pollutant Average")
plt.ylabel("Predicted Pollutant Average")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. FIGURE: DECISION TREE STRUCTURE (TOP THREE LEVELS)
# ============================================================

# The Decision Tree was already fitted in the model-training loop.
# max_depth=2 displays the root node and the next two levels.

decision_tree_model = models["Decision Tree"]

plt.figure(figsize=(20, 10))

plot_tree(
    decision_tree_model,
    feature_names=X.columns.tolist(),
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=10,
    proportion=False,
    precision=2
)

plt.title(
    "Structure of the Fitted Decision Tree (Top Three Levels)"
)

plt.tight_layout()
plt.show()


# ============================================================
# 15. LOW/HIGH CLASSIFICATION ANALYSIS
# ============================================================

print("\nLOW/HIGH PREDICTION CLASSIFICATION")
print("=" * 70)

# Use training-set median as the cutoff for low/high values.
classification_cutoff = float(y_train.median())

actual_classes = (
    np.asarray(y_test) >= classification_cutoff
).astype(int)

predicted_classes = (
    np.asarray(best_predictions) >= classification_cutoff
).astype(int)

cm = confusion_matrix(
    actual_classes,
    predicted_classes,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low", "Actual High"],
    columns=["Predicted Low", "Predicted High"]
)

print("\nConfusion Matrix:")
print(cm_df)

accuracy = accuracy_score(
    actual_classes,
    predicted_classes
)

precision = precision_score(
    actual_classes,
    predicted_classes,
    zero_division=0
)

recall = recall_score(
    actual_classes,
    predicted_classes,
    zero_division=0
)

f1 = f1_score(
    actual_classes,
    predicted_classes,
    zero_division=0
)

print("\nClassification Metrics")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ------------------------------------------------------------
# VISUAL: CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix: Low vs High Pollutant Average")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()


# ============================================================
# 16. AUTOMATIC PROJECT CONCLUSION
# ============================================================

print("\nPROJECT CONCLUSION")
print("=" * 70)

best_r2 = float(best_metrics["R2 Score"])
best_mae = float(best_metrics["MAE"])
best_rmse = float(best_metrics["RMSE"])

lowest_mae_row = results_df.loc[
    results_df["MAE"].idxmin()
]

lowest_rmse_row = results_df.loc[
    results_df["RMSE"].idxmin()
]

print(
    f"\n1. A total of {len(df)} valid records remained "
    "after data cleaning."
)

print(
    f"\n2. The dataset was split into {len(X_train)} training "
    f"records and {len(X_test)} testing records."
)

print(
    f"\n3. The best model based on R-squared was {best_model_name}, "
    f"with R2 = {best_r2:.4f}."
)

print(
    f"\n4. The selected model achieved MAE = {best_mae:.4f} "
    f"and RMSE = {best_rmse:.4f}."
)

print(
    f"\n5. The lowest MAE was obtained by {lowest_mae_row['Model']} "
    f"(MAE = {lowest_mae_row['MAE']:.4f})."
)

print(
    f"\n6. The lowest RMSE was obtained by {lowest_rmse_row['Model']} "
    f"(RMSE = {lowest_rmse_row['RMSE']:.4f})."
)

if best_r2 > 0.8:
    print(
        "\n7. The selected model explains a high proportion of "
        "the variation in the target values on this test split."
    )

elif best_r2 > 0.5:
    print(
        "\n7. The selected model shows moderate explanatory "
        "performance on this test split."
    )

elif best_r2 >= 0:
    print(
        "\n7. The selected model has limited explanatory "
        "performance on this test split."
    )

else:
    print(
        "\n7. The selected model has a negative R-squared score "
        "and performs worse than predicting the test-set mean "
        "under the standard R-squared comparison."
    )

print(
    "\n8. The EDA graphs summarize pollutant distributions and "
    "differences across locations or pollutant types when "
    "the relevant dataset columns are available."
)

print(
    "\n9. The models estimate pollutant_avg from pollutant_min "
    "and pollutant_max. They do not forecast future air quality "
    "or provide official AQI classifications."
)

print(
    "\n10. Future improvements could include adding weather "
    "variables, using more informative predictors, evaluating "
    "on a separate time period, and testing on unseen locations."
)

print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED")
print("=" * 70)

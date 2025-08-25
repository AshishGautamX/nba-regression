# predict.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Target and features
target = "PTS"
features = ["MP", "FGA", "3PA", "FTA", "TRB", "AST", "STL", "BLK"]

results = {}

for feat in features:
    X = df[[feat]]
    y = df[target]

    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load model
    model = joblib.load(f"model_{feat}.pkl")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[feat] = {"R²": r2, "MAE": mae, "RMSE": rmse}

    # Visualization
    plt.figure(figsize=(6,4))
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.plot(X_test, y_pred, color="black", linewidth=2)
    plt.title(f"Regression: {feat} vs {target}\nR²={r2:.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.xlabel(feat)
    plt.ylabel(target)
    plt.legend()
    plt.show()

# Print leaderboard
print("Feature -> R² | MAE | RMSE")
for k, v in sorted(results.items(), key=lambda x: x[1]["R²"], reverse=True):
    print(f"{k:5} -> {v['R²']:.3f} | {v['MAE']:.2f} | {v['RMSE']:.2f}")

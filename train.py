# train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("nba.csv")  # replace with actual file

# Target and features
target = "PTS"
features = ["MP", "FGA", "3PA", "FTA", "TRB", "AST", "STL", "BLK"]

models = {}

for feat in features:
    X = df[[feat]]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f"model_{feat}.pkl")
    print(f"✅ Model trained and saved: model_{feat}.pkl")

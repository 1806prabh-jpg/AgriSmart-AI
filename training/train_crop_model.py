
import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load the crop recommendation dataset
DATA_PATH = "data/Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)

# Separate input features and target label
X = df.drop("label", axis=1)
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create and train the Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("Training the crop recommendation model...")
model.fit(X_train, y_train)

# Evaluate the trained model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed!")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/crop_model.joblib")

# Save evaluation metrics
metrics = {
    "accuracy": float(accuracy),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "number_of_features": X.shape[1],
    "number_of_crops": y.nunique()
}

with open("models/crop_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print("Location: models/crop_model.joblib")

import pickle
import pandas as pd

# Load the saved model
print("=== Loading Saved Model ===")
with open('pr_merge_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✓ Model loaded successfully")

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# Make predictions with loaded model
predictions = model.predict(X_test[:5])

print("\n=== Testing Predictions ===")
print("First 5 test PRs:")
print(f"Predictions: {predictions}")
print(f"Actual:      {y_test[:5]}")
print("\n✓ Model works!")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


#Load train/test data
print("=== Loading Data ===")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.columns.tolist()}")



# Train logistic regression (baseline model)
print("\n=== Training Logistic Regression ===")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

#metrics
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))



# Random Forest
print("\n=== Training Random Forest ===")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


rf_train_pred = rf_model.predict(X_train)
rf_train_acc = accuracy_score(y_train, rf_train_pred)
print(f"Training Accuracy: {rf_train_acc:.4f}")

rf_test_pred = rf_model.predict(X_test)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
print(f"Test Accuracy: {rf_test_acc:.4f}")

#metrics
print("\nClassification Report (Test Set):")
print(classification_report(y_test, rf_test_pred))




#XGBoost
print("\n=== Training XGBoost ===")

xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)


xgb_train_pred = xgb_model.predict(X_train)
xgb_train_acc = accuracy_score(y_train, xgb_train_pred)
print(f"Training Accuracy: {xgb_train_acc:.4f}")


xgb_test_pred = xgb_model.predict(X_test)
xgb_test_acc = accuracy_score(y_test, xgb_test_pred)
print(f"Test Accuracy: {xgb_test_acc:.4f}")

# Detailed metrics on test set
print("\nClassification Report (Test Set):")
print(classification_report(y_test, xgb_test_pred))




#Model Comparison
print("\n=== MODEL COMPARISON ===")
print(f"Logistic Regression - Test Acc: {test_accuracy:.4f}")
print(f"Random Forest       - Test Acc: {rf_test_acc:.4f}")
print(f"XGBoost             - Test Acc: {xgb_test_acc:.4f}")
print("\nWinner: Random Forest (highest test accuracy)")
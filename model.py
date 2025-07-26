import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Create model directory
os.makedirs("model1", exist_ok=True)

# Load dataset
df = pd.read_csv("data/adult 3.csv")

# Replace '?' with NaN and drop missing
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Clean column names
df.columns = df.columns.str.strip()

# Convert income to binary
df['income'] = df['income'].str.strip()
df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)

# Rename columns for consistency
df = df.rename(columns={
    'sex': 'gender',
    'education-num': 'educational-num',
    'marital-status': 'marital-status'
})

# Split features and labels
X = df.drop("income", axis=1)
y = df["income"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns

# Encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Save encoders using pickle
with open("model1/encoder.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Save columns
with open("model1/columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model using pickle
with open("model1/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print(classification_report(y_test, y_pred))

# Feature importance plot
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Top Features Influencing Income Prediction")
plt.tight_layout()
plt.savefig("model1/feature_importance.png")

# Actual vs Predicted plot
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.figure(figsize=(6, 4))
sns.histplot(data=comparison_df, x="Actual", color="blue", label="Actual", kde=True, stat="density", bins=2)
sns.histplot(data=comparison_df, x="Predicted", color="orange", label="Predicted", kde=True, stat="density", bins=2)
plt.legend()
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("model1/actual_vs_predicted.png")

# import pandas as pd
# import numpy as np
# import os
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Create model directory if not exists
# os.makedirs("model", exist_ok=True)

# # Load dataset
# df = pd.read_csv("data/adult 3.csv")

# # Replace missing values marked as '?' with NaN and drop such rows
# df.replace("?", np.nan, inplace=True)
# df.dropna(inplace=True)

# # Strip spaces in column names
# df.columns = df.columns.str.strip()

# # Convert target column
# df['income'] = df['income'].str.strip()
# df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)

# # Separate features and target
# X = df.drop("income", axis=1)
# y = df["income"]

# # Identify categorical and numerical columns
# categorical_cols = X.select_dtypes(include='object').columns
# numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# # Encode categorical features
# encoders = {}
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     encoders[col] = le

# # Scale numerical features
# scaler = StandardScaler()
# X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Train model
# model = RandomForestClassifier(n_estimators=150, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Validation Accuracy: {accuracy:.4f}")
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Plot confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("model/accuracy_plot.png")
# plt.close()

# # Save files
# joblib.dump(model, "model/model.pkl")
# joblib.dump(encoders, "model/encoder.pkl")
# joblib.dump(scaler, "model/scaler.pkl")

# print("Model training complete and files saved to /model/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create model directory if not exists
os.makedirs("model1", exist_ok=True)

# Load dataset
df = pd.read_csv("data/adult 3.csv")

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Drop rows with missing values and strip whitespaces
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Target column
target_column = 'income'

# Encode categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns.drop(target_column)

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target column
target_encoder = LabelEncoder()
df[target_column] = target_encoder.fit_transform(df[target_column])
label_encoders[target_column] = target_encoder

# Train/test split
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Save model
with open("model1/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
with open("model1/encoder.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Save column names
with open("model1/columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# Feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("model1/feature_importance.png")

# Actual vs Predicted plot (for classification, use scatter of probabilities)
probs = model.predict_proba(X_test)[:,1]
plt.figure(figsize=(8,6))
plt.scatter(y_test, probs, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted Probability")
plt.title("Actual vs Predicted Probability")
plt.tight_layout()
plt.savefig("model1/actual_vs_predicted.png")

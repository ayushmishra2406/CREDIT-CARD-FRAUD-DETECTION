import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# Load dataset
df = pd.read_csv("Dataset/creditcard.csv/creditcard.csv")
# If you later fix the folder structure, change the above line to:
# df = pd.read_csv("Dataset/creditcard.csv")

print("Dataset loaded successfully!\n")

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n")
print(df.isnull().sum())


# Exploratory Data Analysis


print("\nClass Distribution:\n")
print(df["Class"].value_counts())

print("\nPercentage Distribution:\n")
print(df["Class"].value_counts(normalize=True) * 100)

# Class Distribution

plt.figure(figsize=(6,4))
df["Class"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Transactions")
plt.savefig("screenshots/class_distribution.png")
plt.close()

# Amount Distribution

plt.figure(figsize=(8,5))
plt.hist(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.savefig("screenshots/amount_distribution.png")
plt.close()

# Time Distribution

plt.figure(figsize=(8,5))
plt.hist(df["Time"], bins=50)
plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.savefig("screenshots/time_distribution.png")
plt.close()

# Fraud vs Normal Amount

plt.figure(figsize=(8,5))
plt.boxplot(
    [
        df[df["Class"] == 0]["Amount"],
        df[df["Class"] == 1]["Amount"]
    ],
    labels=["Normal", "Fraud"]
)
plt.title("Fraud vs Normal Transaction Amount")
plt.ylabel("Amount")
plt.savefig("screenshots/fraud_vs_normal_amount.png")
plt.close()

# Correlation Heatmap

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.close()

# Amount vs Time

plt.figure(figsize=(8,5))
plt.scatter(df["Time"], df["Amount"], s=1)
plt.title("Amount vs Time")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.savefig("screenshots/amount_vs_time.png")
plt.close()

# Fraud Percentage

counts = df["Class"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    counts,
    labels=["Normal", "Fraud"],
    autopct="%1.2f%%"
)
plt.title("Fraud Percentage")
plt.savefig("screenshots/fraud_percentage.png")
plt.close()


# Prepare the data


X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# Train Model


model = RandomForestClassifier(
    n_estimators=30,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Predictions


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.4f}")

# Classification Report

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")
print(cm)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.savefig("screenshots/confusion_matrix.png")
plt.close()


# Feature Importance


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(12,6))
importance.head(15).plot(kind="bar")
plt.title("Top 15 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.savefig("screenshots/feature_importance.png")
plt.close()


# ROC Curve


fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("screenshots/roc_curve.png")
plt.close()


# Precision Recall Curve


precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("screenshots/precision_recall_curve.png")
plt.close()

print("\nProject completed successfully!")
print(f"Final Accuracy : {accuracy:.4f}")
print(f"ROC AUC Score : {roc_auc:.4f}")
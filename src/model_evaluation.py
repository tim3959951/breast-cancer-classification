import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import joblib

# Load test data
df = pd.read_csv("cleaned_breast_cancer_data.csv")
X_test = df.drop(columns=['Class'])
y_test = df['Class']

# Load models
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
}

# Evaluate models
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results.append({"Model": name, "Accuracy": accuracy, "AUC": roc_auc})
    
    # Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"visualizations/conf_matrix_{name.replace(' ', '_').lower()}.png")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("model_performance.csv", index=False)

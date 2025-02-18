import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load test data
df = pd.read_csv("cleaned_breast_cancer_data.csv")
X_test = df.drop(columns=['Class'])

# Load best model (Random Forest)
model = joblib.load("random_forest_model.pkl")

# SHAP analysis
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Summary Plot
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test)
plt.savefig("visualizations/shap_summary_plot.png")

# Feature Importance
shap_values_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
mean_abs_shap = shap_values_df.abs().mean().sort_values(ascending=False)
mean_abs_shap.plot(kind="barh", figsize=(8,5), title="Feature Importance (SHAP)")
plt.savefig("visualizations/shap_feature_importance.png")

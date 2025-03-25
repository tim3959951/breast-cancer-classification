# **Breast Cancer Classification using Machine Learning**

## 📌 Project Overview  
This project applies various machine learning models to classify **breast cancer tumors** as **malignant (cancerous) or benign (non-cancerous)** using a publicly available dataset. The focus is on **model performance, feature importance, and explainability** using SHAP (SHapley Additive exPlanations).

## 🏆 Key Highlights  
✅ **Models Used:** Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, MLP (Neural Network)  
✅ **Best Model:** Logistic Regression (AUC = 0.9927)  
✅ **Explainability:** SHAP Analysis to interpret feature importance  
✅ **Dataset:** Breast cancer cell data with 8 key features  
✅ **Evaluation Metrics:** Accuracy, AUC, ROC Curve, Confusion Matrix  

---

## ⚙️ Machine Learning Pipeline  
1️⃣ **Data Preprocessing**  
   - Handled missing values (`Bare_Nuclei` contained `?` values)  
   - Standardized numerical features  
   - Split dataset into **training (80%) and testing (20%)**  

2️⃣ **Model Training & Tuning**  
   - Used **GridSearchCV** for hyperparameter tuning  
   - Evaluated models using **Accuracy & AUC**  

3️⃣ **Model Evaluation**  
   - Compared performance using **ROC Curves & Confusion Matrices**  
   - Identified best-performing model: **Logistic Regression**  

---

## 📊 Dataset Information  
- The dataset consists of **breast cancer cell features** extracted from microscopic images.  
- Features include:  
  - `Bare_Nuclei` – Most influential feature in predicting malignancy  
  - `Clump_Thickness`, `Uniformity_Of_Cell_Size`, `Bland_Chromatin`, etc.  
- **Target Variable:**  
  - `0` → Benign Tumor  
  - `1` → Malignant Tumor  

---

## 📈 Model Performance  
| **Model**             | **Accuracy** | **AUC**  | **Precision** | **Recall** | **F1-score** |
|------------------------|--------------|----------|----------------|-------------|--------------|
| Logistic Regression    | 96.35%        | 99.27%   | 92.16%         | 97.92%      | 94.95%       |
| Random Forest          | 96.35%        | 98.50%   | 92.00%         | 95.83%      | 93.88%       |
| SVM                    | 95.62%        | 97.00%   | 92.00%         | 95.83%      | 93.88%       |
| **XGBoost**            | **97.08%**    | 98.74%   | 92.31%         | **100.0%**  | **96.00%**   |
| LightGBM               | 95.62%        | 98.64%   | **93.75%**     | 93.75%      | 93.75%       |
| MLP                    | 94.89%        | **98.99%** | 91.84%         | 93.75%      | 92.78%       |


🔹 **ROC Curve Comparison** (Higher AUC means better classification):  
<img src="visualizations/final_roc_curve.png" width="600">

📊 **F1-score vs AUC Comparison**
> F1-score measures balance between precision and recall. AUC reflects overall classification power.

<img src="visualizations/f1_vs_auc_barplot.png" width="600">

- Based on F1-score and recall, XGBoost was selected as the final model due to its perfect identification of malignant cases.

---

### **Confusion Matrices** (Compare Model Performance)  
<table>
  <tr>
    <td align="center"><b>Logistic Regression</b></td>
    <td align="center"><b>Random Forest</b></td>
  </tr>
  <tr>
    <td><img src="visualizations/tuned_conf_matrix_logreg.png" width="400"></td>
    <td><img src="visualizations/tuned_conf_matrix_rf.png" width="400"></td>
  </tr>
</table>

---

## 🛠️ SHAP Analysis - Feature Importance (Malignant Tumors)  
📌 **SHAP analysis helps explain how each feature impacts the model's decision, ensuring AI transparency in healthcare applications.**  

### **🔍 Feature Importance & Dependence**  
<table>
  <tr>
    <td align="center"><b>SHAP Feature Importance</b></td>
    <td align="center"><b>SHAP Dependence - Bare Nuclei</b></td>
  </tr>
  <tr>
    <td><img src="visualizations/shap_feature_importance_malignant.png" width="400"></td>
    <td><img src="visualizations/shap_dependence_bare_nuclei_malignant.png" width="400"></td>
  </tr>
</table>

📌 **Key Insights:**  
- `Bare_Nuclei` is the strongest predictor of malignancy.  
- The **dependence plot** shows that higher `Bare_Nuclei` values increase malignancy probability, interacting with `Uniformity_Of_Cell_Size`.  

---

### **🔍 SHAP Summary Plot**  
📌 **This plot ranks features based on their impact on malignancy predictions.**  
<img src="visualizations/shap_summary_plot_malignant.png" width="600">

📌 **Key Takeaways:**  
- `Bare_Nuclei` contributes most to model predictions.  
- Higher values of `Uniformity_Of_Cell_Size` also increase malignancy probability.  
- Feature interactions (e.g., `Bare_Nuclei` & `Uniformity_Of_Cell_Size`) play a key role.  

---

## 🚀 Next Steps / Potential Improvements  
- **Add more advanced models** (e.g., Transformer-based models for tabular data)  
- **Use feature engineering techniques** (e.g., PCA, additional domain knowledge features)  
- **Deploy as a web app** (Streamlit / Flask API for real-time predictions)  

---

## 🏆 Key Takeaways  
✅ **Logistic Regression is the best-performing model (AUC = 0.9927).**  
✅ **SHAP analysis confirms `Bare_Nuclei` as the most critical feature.**  
✅ **Feature interactions (e.g., `Bare_Nuclei` & `Uniformity_Of_Cell_Size`) are important for prediction.**  
  

---
## 📂 Repository Structure  
| File/Folder             | Description                                      |
|-------------------------|--------------------------------------------------|
| 📂 src                 | Contains all core scripts                        |
| 📂 visualizations      | Stores generated plots & SHAP analysis images    |
| 📂 experiments                 | Model experiments & evaluation |
| 📄 requirements.txt    | Python dependencies                              |
| 📄 README.md           | Project documentation                           |
| 📄 .gitignore          | Ignore unnecessary files                         |
| 📄 breast_cancer_classification_with_shap.ipynb | Main Jupyter Notebook |

---
## 🚀 How to Run
Run the full pipeline with:
```bash
python src/main.py
```

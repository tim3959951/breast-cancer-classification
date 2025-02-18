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

## 📊 Dataset Information  
- The dataset consists of **breast cancer cell features** extracted from microscopic images.  
- Features include:  
  - `Bare_Nuclei` – Most influential feature in predicting malignancy  
  - `Clump_Thickness`, `Uniformity_Of_Cell_Size`, `Bland_Chromatin`, etc.  
- **Target Variable:**  
  - `0` → Benign Tumor  
  - `1` → Malignant Tumor  

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

4️⃣ **Explainability with SHAP**  
   - Identified key features driving predictions  
   - Explored **feature interactions** (e.g., `Bare_Nuclei` & `Uniformity_Of_Cell_Size`)  

---

## 📈 Model Performance  
| **Model** | **Accuracy** | **AUC** |  
|-----------|-------------|---------|  
| Logistic Regression | **96.35%** | **0.9927** |  
| Random Forest | 96.35% | 0.9850 |  
| SVM | 95.62% | 0.9700 |  
| XGBoost | **97.08%** | 0.9874 |  
| LightGBM | 95.62% | 0.9864 |  
| MLP | 94.89% | **0.9899** |  

🔹 **ROC Curve Comparison:**  
![ROC Curve](visualizations/final_roc_curve.png)  

🔹 **Confusion Matrices:**  
- **Logistic Regression:**  
  ![Confusion Matrix - Logistic Regression](visualizations/tuned_conf_matrix_logreg.png)  
- **Random Forest:**  
  ![Confusion Matrix - Random Forest](visualizations/tuned_conf_matrix_rf.png)  

---

## 🛠️ SHAP Analysis - Feature Importance (Malignant Tumors)  
### **🔍 Summary Plot**  
![SHAP Summary Plot](visualizations/shap_summary_plot_malignant.png)  

### **🔍 Feature Importance Ranking**  
![SHAP Feature Importance](visualizations/shap_feature_importance_malignant.png)  

### **🔍 Dependence Plot - `Bare_Nuclei` Impact on Malignancy**  
![SHAP Dependence - Bare Nuclei](visualizations/shap_dependence_bare_nuclei_malignant.png)  

---

## 🚀 Next Steps / Potential Improvements  
- **Add more advanced models** (e.g., Transformer-based models for tabular data)  
- **Use feature engineering techniques** (e.g., PCA, additional domain knowledge features)  
- **Deploy as a web app** (Streamlit / Flask API for real-time predictions)  

---

## 📂 Repository Structure  


| File/Folder             | Description                                      |
|-------------------------|--------------------------------------------------|
| 📂 src                 | Contains all core scripts                        |
| 📂 visualizations      | Stores generated plots & SHAP      |
| 📄 requirements.txt    | Python dependencies                              |
| 📄 README.md           | Project documentation                           |
| 📄 .gitignore          | Ignore unnecessary files                         |
| 📄 2019_IMBD_1D_CNN.ipynb | Main Jupyter Notebook |


---
## 🏆 Key Takeaways  
✅ **Logistic Regression is the best-performing model (AUC = 0.9927).**  
✅ **SHAP analysis confirms `Bare_Nuclei` as the most critical feature.**  
✅ **Feature interactions (e.g., `Bare_Nuclei` & `Uniformity_Of_Cell_Size`) are important for prediction.**  

📌 **This project is an excellent demonstration of AI & Machine Learning in healthcare!** 🚀  




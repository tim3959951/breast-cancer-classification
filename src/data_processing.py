import pandas as pd
import numpy as np

# Load dataset
file_path = "breast_cancer_data.csv"  # Ensure this file is in the correct directory
columns = ['Sample_code_number','Clump_Thickness','Uniformity_Of_Cell_Size',
           'Uniformity of Cell Shape','Marginal Adhesion','Single_Epithelial_Cell_Size',
           'Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses','Class']
df = pd.read_csv(file_path, names=columns)

# Remove ID column
df.drop('Sample_code_number', axis=1, inplace=True)

# Convert "?" in 'Bare_Nuclei' to NaN and change dtype to float
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Convert target variable (2 -> 0 for benign, 4 -> 1 for malignant)
df['Class'] = df['Class'].map({2: 0, 4: 1}).astype(int)

# Feature correlation filtering
correlation_matrix = df.corr()
correlation_threshold = 0.9
high_correlation_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            high_correlation_features.add(colname)

df.drop(columns=high_correlation_features, inplace=True)

# Split data into features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Save cleaned dataset for further processing
df.to_csv("cleaned_breast_cancer_data.csv", index=False)

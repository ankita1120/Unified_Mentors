# 🧠 OCD Patient Dataset: Demographics & Clinical Data Analysis

This project explores and analyzes a dataset of 1,500 patients diagnosed with Obsessive-Compulsive Disorder (OCD), using exploratory data analysis (EDA) and machine learning (ML) techniques to uncover insights and build predictive models.

---

## 📁 Dataset Overview

The dataset contains demographic and clinical information including:

- Age, Gender, Ethnicity, Marital Status, Education Level
- Duration of OCD Symptoms
- Types of Obsessions and Compulsions
- Y-BOCS Scores (Obsessions and Compulsions)
- Co-occurring Diagnoses (Depression, Anxiety)
- Medications
- Family History and Previous Diagnoses

📦 [Download Dataset](https://drive.google.com/file/d/1q0kasDpGbhIeOdnCEyBHxRRGdzpkS7mQ/view)

---

## 🎯 Objectives

- Perform exploratory data analysis (EDA) on demographic and clinical variables
- Preprocess and clean the dataset
- Encode categorical variables and scale features
- Train multiple machine learning models
- Predict prescribed medication types based on clinical and demographic features
- Visualize key relationships and model performance

---

## 🔧 Tools & Technologies

- **Languages:** Python
- **Libraries:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Models:** Random Forest, XGBoost, LightGBM, Logistic Regression, Voting Ensemble

---

## 🧪 Key Steps

### 1. Exploratory Data Analysis
- Distribution plots of Age, Gender, and Ethnicity
- Boxplots of Y-BOCS scores across Gender
- Heatmaps of correlations between numeric variables

### 2. Data Preprocessing
- Handling missing values using `SimpleImputer`
- Label Encoding of categorical variables
- Scaling with `MinMaxScaler`
- Train-test split

### 3. Machine Learning Modeling
- Model training with Random Forest, XGBoost, LightGBM, and Voting Classifier
- Evaluation using accuracy and confusion matrix
- Feature importance analysis

---

## 📊 Results

- Models achieved ~30-35% accuracy due to class imbalance and medication overlaps.
- Voting classifier combines strengths of various models for better generalization.
- Key influencing features: OCD duration, Y-BOCS scores, Depression/Anxiety diagnosis.

---

## 📁 Folder Structure

📂 OCD-ML-Project/
├── 📄 README.md
├── 📄 ocd_patient_dataset.csv
├── 📄 ocd_analysis.ipynb # Main notebook
├── 📁 plots/ # All EDA and result plots
├── 📁 models/ # Saved models (if applicable)

yaml
Copy
Edit

---

## 📌 Future Improvements

- Hyperparameter tuning (GridSearchCV, Optuna)
- Class balancing techniques (SMOTE, class weighting)
- Model interpretability (SHAP, LIME)
- Deployment via Streamlit or Flask

---

## 🙌 Acknowledgements

Dataset provided by the OCD research portal [link above].  
Inspired by mental health informatics and data-driven psychiatry projects.


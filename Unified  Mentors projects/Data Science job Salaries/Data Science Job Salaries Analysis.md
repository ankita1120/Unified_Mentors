Data Science Job Salaries Analysis


readme_content = """
# 💼 Data Science Job Salaries Analysis

This repository contains a Jupyter Notebook analyzing salary trends in the data science industry using a real-world dataset. It explores how various factors — including experience level, remote work, and company size — influence salaries.

## 📁 Files

- `Data Science Job Salaries.ipynb`: Jupyter Notebook containing data analysis and visualization.
- `Data Science Job Salaries.csv`: Dataset with salary information for various data-related job roles.

## 📊 Dataset Overview

The dataset includes:
- `work_year`, `experience_level`, `employment_type`, `job_title`
- `salary`, `salary_currency`, `salary_in_usd`
- `employee_residence`, `remote_ratio`, `company_location`, `company_size`

## 📌 Key Analysis Steps

1. **Data Loading & Cleaning**
   - Handle missing values
   - Standardize categorical data
   - Drop rows with critical null values
![alt text](file:///Users/ankita/Downloads/img1.png)

2. **Feature Engineering**
   - Ordinal encoding for `experience_level` and `employment_type`
   - Derived features like `salary_ratio`
   - One-hot encoding for `company_size`
![alt text](file:///Users/ankita/Downloads/img2.png)

3. **Exploratory Data Analysis (EDA)**
   - Salary distribution (histogram)
   - Correlation heatmap
   - Boxplots for salary vs experience level
   - Bar plots for salary vs remote ratio

![alt text](file:///Users/ankita/Downloads/img3.png)
![alt text](file:///Users/ankita/Downloads/img4.png)

4. **Machine Learning**
   - Linear Regression to predict `salary_in_usd`
   - Feature selection and model evaluation

5. **Interactive Dashboard (Streamlit)**
   - Summary statistics
   - Line chart of salary trends
   - Selectable job title filter with bar chart of salaries

## 📈 Example Visuals

- 📊 Salary Distribution
- 🔥 Correlation Heatmap
- 📦 Salary by Experience Level (Box Plot)
- 🌍 Salary by Remote Ratio

## 🧪 Requirements

- Python 3.x
- Libraries: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `ipywidgets`, `streamlit`

Install dependencies with:
```bash
pip install pandas matplotlib seaborn scikit-learn ipywidgets streamlit

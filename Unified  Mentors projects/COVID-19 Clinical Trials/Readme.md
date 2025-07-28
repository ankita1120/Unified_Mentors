# 🧬 COVID Clinical Trials – EDA & Machine Learning

This project performs **Exploratory Data Analysis (EDA)** and applies a **Machine Learning (ML)** model to analyze a dataset of COVID-19 clinical trials. The goal is to understand key trends and build a predictive model to classify study types.

---

## 📂 Dataset

- File: `COVID clinical trials.csv`
- Contains metadata about various clinical trials conducted for COVID-19.
- Common columns include: `Study Type`, `Status`, `Conditions`, `Interventions`, `Phases`, `Locations`, and more.

---

## 🔍 Features

- ✅ Data Cleaning: Handle missing values, drop irrelevant columns, remove duplicates.
- ✅ EDA: Summary statistics, distribution plots, value counts.
- ✅ Label Encoding for categorical variables.
- ✅ ML Model: Random Forest Classifier to predict if a trial is "Interventional".
- ✅ Evaluation: Classification report and feature importance chart.
 Output:
- First 5 rows:
   Rank   NCT Number                                              Title  \
0     1  NCT04785898  Diagnostic Performance of the ID Now™ COVID-19...   
1     2  NCT04595136  Study to Evaluate the Efficacy of COVID19-0001...   
2     3  NCT04395482  Lung CT Scan Analysis of SARS-CoV2 Induced Lun...   
3     4  NCT04416061  The Role of a Private Hospital in Hong Kong Am...   
4     5  NCT04395924         Maternal-foetal Transmission of SARS-Cov-2   

        Acronym                  Status         Study Results  \
0   COVID-IDNow  Active, not recruiting  No Results Available   
1      COVID-19      Not yet recruiting  No Results Available   
2   TAC-COVID19              Recruiting  No Results Available   
3      COVID-19  Active, not recruiting  No Results Available   
4  TMF-COVID-19              Recruiting  No Results Available   

                                          Conditions  \
0                                            Covid19   
1                               SARS-CoV-2 Infection   
2                                            covid19   
3                                              COVID   
4  Maternal Fetal Infection Transmission|COVID-19...   

                                       Interventions  \
0   Diagnostic Test: ID Now™ COVID-19 Screening Test   
1    Drug: Drug COVID19-0001-USR|Drug: normal saline   
2  Other: Lung CT scan analysis in COVID-19 patients   
3          Diagnostic Test: COVID 19 Diagnostic Test   
4  Diagnostic Test: Diagnosis of SARS-Cov2 by RT-...   

                                    Outcome Measures  \
0  Evaluate the diagnostic performance of the ID ...   
1  Change on viral load results from baseline aft...   
2  A qualitative analysis of parenchymal lung dam...   
3  Proportion of asymptomatic subjects|Proportion...   
4  COVID-19 by positive PCR in cord blood and / o...   

                               Sponsor/Collaborators  ...         Other IDs  \
0              Groupe Hospitalier Paris Saint Joseph  ...       COVID-IDNow   
1                         United Medical Specialties  ...  COVID19-0001-USR   
2                       University of Milano Bicocca  ...       TAC-COVID19   
3                    Hong Kong Sanatorium & Hospital  ...        RC-2020-08   
4  Centre Hospitalier Régional d'Orléans|Centre d...  ...      CHRO-2020-10   

         Start Date Primary Completion Date   Completion Date  \
0  November 9, 2020       December 22, 2020    April 30, 2021   
1  November 2, 2020       December 15, 2020  January 29, 2021   
2       May 7, 2020           June 15, 2021     June 15, 2021   
3      May 25, 2020           July 31, 2020   August 31, 2020   
4       May 5, 2020                May 2021          May 2021   

       First Posted Results First Posted Last Update Posted  \
0     March 8, 2021                  NaN      March 8, 2021   
1  October 20, 2020                  NaN   October 20, 2020   
2      May 20, 2020                  NaN   November 9, 2020   
3      June 4, 2020                  NaN       June 4, 2020   
4      May 20, 2020                  NaN       June 4, 2020   

                                           Locations Study Documents  \
0  Groupe Hospitalier Paris Saint-Joseph, Paris, ...             NaN   
1       Cimedical, Barranquilla, Atlantico, Colombia             NaN   
2  Ospedale Papa Giovanni XXIII, Bergamo, Italy|P...             NaN   
3  Hong Kong Sanatorium & Hospital, Hong Kong, Ho...             NaN   
4                       CHR Orléans, Orléans, France             NaN   

                                           URL  
0  https://ClinicalTrials.gov/show/NCT04785898  
1  https://ClinicalTrials.gov/show/NCT04595136  
2  https://ClinicalTrials.gov/show/NCT04395482  
3  https://ClinicalTrials.gov/show/NCT04416061  
4  https://ClinicalTrials.gov/show/NCT04395924  

[5 rows x 27 columns]

Dataset Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5783 entries, 0 to 5782
Data columns (total 27 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   Rank                     5783 non-null   int64  
 1   NCT Number               5783 non-null   object 
 2   Title                    5783 non-null   object 
 3   Acronym                  2480 non-null   object 
 4   Status                   5783 non-null   object 
 5   Study Results            5783 non-null   object 
 6   Conditions               5783 non-null   object 
 7   Interventions            4897 non-null   object 
 8   Outcome Measures         5748 non-null   object 
 9   Sponsor/Collaborators    5783 non-null   object 
 10  Gender                   5773 non-null   object 
 11  Age                      5783 non-null   object 
 12  Phases                   3322 non-null   object 
 13  Enrollment               5749 non-null   float64
 14  Funded Bys               5783 non-null   object 
 15  Study Type               5783 non-null   object 
 16  Study Designs            5748 non-null   object 
 17  Other IDs                5782 non-null   object 
 18  Start Date               5749 non-null   object 
 19  Primary Completion Date  5747 non-null   object 
 20  Completion Date          5747 non-null   object 
 21  First Posted             5783 non-null   object 
 22  Results First Posted     36 non-null     object 
 23  Last Update Posted       5783 non-null   object 
 24  Locations                5198 non-null   object 
 25  Study Documents          182 non-null    object 
 26  URL                      5783 non-null   object 
dtypes: float64(1), int64(1), object(25)
memory usage: 1.2+ MB
None

Missing values:
Rank                          0
NCT Number                    0
Title                         0
Acronym                    3303
Status                        0
Study Results                 0
Conditions                    0
Interventions               886
Outcome Measures             35
Sponsor/Collaborators         0
Gender                       10
Age                           0
Phases                     2461
Enrollment                   34
Funded Bys                    0
Study Type                    0
Study Designs                35
Other IDs                     1
Start Date                   34
Primary Completion Date      36
Completion Date              36
First Posted                  0
Results First Posted       5747
Last Update Posted            0
Locations                   585
Study Documents            5601
URL                           0
dtype: int64

Unique values per column:
Rank                       5783
NCT Number                 5783
Title                      5775
Acronym                    2338
Status                       12
Study Results                 2
Conditions                 3067
Interventions              4337
Outcome Measures           5687
Sponsor/Collaborators      3631
Gender                        3
Age                         417
Phases                        8
Enrollment                  962
Funded Bys                   26
Study Type                    9
Study Designs               267
Other IDs                  5734
Start Date                  654
Primary Completion Date     877
Completion Date             978
First Posted                438
Results First Posted         33
Last Update Posted          269
Locations                  4255
Study Documents             182
URL                        5783
dtype: int64

Value counts for some potentially important columns:

Study Results:
Study Results
No Results Available    5747
Has Results               36
Name: count, dtype: int64

Gender:
Gender
All       5567
Female     162
Male        44
Name: count, dtype: int64

Phases:
Phases
Not Applicable     1354
Phase 2             685
Phase 3             450
Phase 1             234
Phase 2|Phase 3     200
Phase 1|Phase 2     192
Phase 4             161
Early Phase 1        46
Name: count, dtype: int64

Study Type:
Study Type
Interventional                                                         3322
Observational                                                          2427
Expanded Access:Intermediate-size Population                             15
Expanded Access:Treatment IND/Protocol                                    8
Expanded Access:Intermediate-size Population|Treatment IND/Protocol       5
Expanded Access:Individual Patients                                       3
Expanded Access:Individual Patients|Intermediate-size Population          1
Expanded Access                                                           1
Expanded Access:Individual Patients|Treatment IND/Protocol                1
Name: count, dtype: int64

![clipboard.png](../../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard.png)

---
--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       489
           1       1.00      1.00      1.00       668

    accuracy                           1.00      1157
   macro avg       1.00      1.00      1.00      1157
weighted avg       1.00      1.00      1.00      1157


Accuracy Score: 1.0
![clipboard1.png](../../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard1.png)

## 📦 Requirements

Install the following Python packages:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run

1. **Download the dataset**: Make sure the CSV file is saved in your working directory.

2. **Run the script**: Use Jupyter Notebook, VS Code, or a Python IDE of your choice.

3. **Adjust file path**: If needed, modify this line in the script to point to your dataset:
   ```python
   file_path = 'COVID clinical trials.csv'
   ```

---

## 📊 Output

- Printed summary and structure of the dataset.
- Count plots showing distributions of study types and statuses.
- Accuracy score, classification report.
- Top feature importances plotted using a bar chart.

---

## 🚀 Future Improvements

- Add more ML models for comparison (e.g., Logistic Regression, XGBoost).
- Perform time-series analysis on trial start dates.
- Use NLP to extract insights from study descriptions.

---

## 👩‍💻 Author

Developed by **Ankita** as part of a data analysis and ML learning project.

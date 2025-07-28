# Instagram Fake Account Detection 🚫📸

This project aims to classify Instagram accounts as **Fake (Spammer)** or **Genuine**, using machine learning and data analysis techniques. It leverages structured Instagram profile data, including username patterns, bio characteristics, follower/following counts, and posting behavior.

---

## 🧠 Project Objectives

- Detect fake/spammer Instagram accounts using supervised learning.
- Identify the most important features distinguishing genuine from fake accounts.
- Perform exploratory data analysis (EDA) and feature engineering.
- Visualize insights using matplotlib/seaborn and evaluate the model using classification metrics.

---

## 🔧 Tools & Technologies

- **Python 3**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn**
- **Tableau, Excel (optional for EDA/visualization)**
- **Jupyter Notebook / Kaggle**

---

## 📁 Dataset

The dataset includes manually labeled Instagram accounts as **fake (1)** or **genuine (0)**.

📥 [Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1tgbfqoj2_rob-ZFAdDHXwl-LlE0Xf4T-?usp=sharing)

### Features:
- `profile_pic`: 1 if profile picture exists
- `nums/length username`: ratio of digits to total length in the username
- `fullname words`: number of words in full name
- `nums/length fullname`: ratio of digits in full name
- `name==username`: 1 if full name and username match
- `description length`: number of characters in bio
- `external URL`: 1 if a link is in the bio
- `private`: 1 if the account is private
- `#posts`, `#followers`, `#follows`: activity metrics
- `fake`: target label (1 = fake, 0 = genuine)

---

## 🔍 Step-by-Step Process

### 1. Data Preprocessing
- Checked for missing values
- Created additional features (e.g., follower bins)
- Scaled numerical data using `StandardScaler`

### 2. Exploratory Data Analysis (EDA)
- Visualized distribution of target labels
- Explored relationships between features (e.g., followers vs. fake status)
- Used heatmaps, boxplots, and count plots

### 3. Model Building
- Used **Random Forest** and **Decision Tree Classifiers**
- Trained using 70/30 and 80/20 train/test splits
- Achieved up to **94% accuracy** on test data

### 4. Model Evaluation
- Metrics: accuracy, precision, recall, F1-score
- Confusion matrix visualization with `ConfusionMatrixDisplay`

---

## 📊 Key Insights

- Fake accounts often:
  - Have no profile pictures
  - Have odd username/fullname patterns
  - Follow many but are followed by few
  - Have short bios or missing URLs

- Most predictive features:
  - `profile_pic`, `#followers`, `#follows`, `description length`

---

## 📈 Example Output

```
Accuracy: 94%
Precision (Fake): 95%
Recall (Fake): 93%
```

Confusion Matrix:
```
          Predicted
          0     1
Actual  57     3
        4     56
```

---

## 📌 Future Improvements

- Use more advanced models (e.g., XGBoost, LightGBM)
- Apply cross-validation and hyperparameter tuning (GridSearchCV)
- Handle class imbalance with SMOTE or class weighting
- Deploy as a REST API or integrate with Tableau dashboards

---

## 🔗 References

- 📚 [Kaggle Version of Dataset](https://www.kaggle.com/datasets)
- 📂 [Original Project Dataset Folder](https://drive.google.com/drive/folders/1tgbfqoj2_rob-ZFAdDHXwl-LlE0Xf4T-?usp=sharing)
- 💻 [Inspiration GitHub Repo](https://github.com/deepd1534/instagram-fake-account-detection)

---

## 👤 Author

**Final Year ML/DA Student**  


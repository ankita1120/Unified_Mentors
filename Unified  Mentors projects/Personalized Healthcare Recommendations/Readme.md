# 🩸 Personalized Healthcare Recommendation System

This project uses machine learning to predict blood donation eligibility and provide personalized healthcare recommendations based on donation history data. It supports a command-line workflow, a Flask web app, and a Streamlit interface for live prediction and interaction.

---

## 📊 Dataset

- **Filename**: `healthcare _ data.csv`
- **Source**: Based on the [UCI Blood Donation dataset](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)
- **Features**:
  - `Recency`: Months since the last donation
  - `Frequency`: Number of donations
  - `Monetary`: Total blood donated (in c.c.)
  - `Time`: Months since the first donation
- **Target**:
  - `Class`: 1 = Eligible, 0 = Not eligible

---

## 🛠️ Technologies Used

- Python 3.x
- pandas
- scikit-learn
- matplotlib / seaborn (for visualizations)
- Flask (web deployment)
- Streamlit (interactive web UI)
- joblib (model persistence)
- 
![clipboard.png](../../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard.png)
![clipboard1.png](../../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard1.png)
![clipboard2.png](../../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard2.png)
![clipboard4.png](../../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard4.png)
![clipboard8.png](../../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard8.png)
---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/healthcare-recommendation-system.git
cd healthcare-recommendation-system

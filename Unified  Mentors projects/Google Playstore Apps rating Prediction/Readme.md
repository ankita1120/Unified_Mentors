📱 Google Play Store Data Cleaning and EDA
This project focuses on cleaning, exploring, and visualizing the Google Play Store dataset for app analysis and modeling.

🚩 Project Goals
✅ Clean the dataset to handle missing values, inconsistent formats, and non-numeric entries.
✅ Explore app distribution across categories, genres, and price ranges.
✅ Identify most installed apps and highest-rated categories.
✅ Prepare the dataset for downstream machine learning tasks.

📂 Dataset Overview
The dataset includes:

App: App name

Category: App category

Rating: User rating (0-5)

Reviews: Number of reviews

Installs: Number of installs (formatted with K, M, +)

Price: App price (formatted with $)

Genres: App genres

Other metadata fields

🛠️ Cleaning Steps
Fill missing ratings with category-wise mean.

Drop rows with missing critical values (App, Category, Reviews, Installs).

Convert Reviews to integers, handling non-numeric safely.

Clean Price column by removing $ and converting to float.

Clean Installs by handling K, M, +, and commas, converting to integers.

Reset indices after cleaning for consistent referencing.

📊 Exploratory Data Analysis (EDA)
Visualizations and analyses include:
![clipboard.png](../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard.png)
Distribution of app categories using count plots.
![clipboard1.png](../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard1.png)
Average rating by category (table and plots).
![clipboard2.png](../../../private/var/folders/ns/s3vsx14d50n1nptjxdcgvvch0000gn/T/clipboard2.png)
Most installed apps (top N inspection).

Top genres by frequency.

Price distribution and free vs paid app counts.

🧩 Tools Used
Python 3.12

Pandas for data cleaning and manipulation

NumPy for numerical handling

Seaborn & Matplotlib for visualizations

Jupyter Notebook / VS Code for interactive exploration

📈 Potential Next Steps
✅ Log-transform skewed features (Installs, Reviews) for modeling.
✅ Label encoding or one-hot encoding for categorical features.
✅ Correlation heatmaps for feature relationships.
✅ Model building for predicting app rating or price.
✅ Deploy EDA insights into a Streamlit dashboard for interactive exploration.

🚀 Usage
1️⃣ Clone this repository.
2️⃣ Place your googleplaystore.csv in the working directory.
3️⃣ Run:

python
Copy
Edit
import pandas as pd
df = pd.read_csv('googleplaystore.csv')
# Then run the cleaning and EDA notebook.
4️⃣ Visualize insights, clean data, and prepare for modeling.

🤝 Contributing
Pull requests and issue reports are welcome to improve cleaning robustness, add visualizations, or build modeling pipelines.

📧 Contact
If you have questions about this project or want to collaborate, feel free to reach out.

If you want, I can also prepare:
✅ A structured GitHub repo structure suggestion (data/, notebooks/, src/, etc.)
✅ A requirements.txt for consistent environment setup
✅ A Jupyter notebook template aligned with this README for your workflow.

Let me know if you would like these next to complete your project packaging efficiently.









Ask ChatGPT

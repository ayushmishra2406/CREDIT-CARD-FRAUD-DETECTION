# 💳 Credit Card Fraud Detection using Machine Learning

A machine learning project developed as part of the **CodSoft Machine Learning Internship**. This project uses a **Random Forest Classifier** to identify fraudulent credit card transactions based on transaction details.

---

## 📌 Project Overview

Credit card fraud is a major financial concern, and detecting fraudulent transactions accurately is essential. In this project, a Random Forest model is trained on a real-world credit card transaction dataset to classify transactions as either **legitimate** or **fraudulent**.

The project includes data exploration, visualization, feature engineering, model training, and performance evaluation using multiple metrics.

---

## 🚀 Features

- Data exploration and preprocessing
- Exploratory Data Analysis (EDA)
- Class distribution analysis
- Feature engineering
- Random Forest Classifier
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
- Feature Importance visualization

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Project Structure

```
CREDIT CARD FRAUD DETECTION/
│
├── Dataset/
│   └── (Download creditcard.csv from Kaggle)
│
├── Screenshots/
│
├── credit_card_fraud_detection.ipynb
├── credit_card_fraud_detection.py
├── Credit_Card_Fraud_Detection_Report.docx
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset is **not included** in this repository because it exceeds GitHub's file size limit.

You can download it from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file inside the **Dataset** folder.

```
Dataset/
└── creditcard.csv
```

---

## 📈 Visualizations

The project includes the following visualizations:

- Class Distribution
- Transaction Amount Distribution
- Transaction Time Distribution
- Fraud vs Normal Transaction Amount
- Correlation Heatmap
- Amount vs Time
- Fraud Percentage
- Confusion Matrix
- Feature Importance
- ROC Curve
- Precision-Recall Curve

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/ayushmishra2406/CREDIT CARD FRAUD DETECTION.git
```

### Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Download the dataset

Download the dataset from Kaggle and place **creditcard.csv** inside the **Dataset** folder.

### Run the project

```bash
python credit_card_fraud_detection.py
```

---

## 📌 Results

The Random Forest model successfully classified fraudulent transactions with excellent performance on the test dataset. The project demonstrates the effectiveness of machine learning techniques for fraud detection while handling an imbalanced dataset.

---

## 🔮 Future Improvements

- Hyperparameter tuning
- XGBoost and LightGBM implementation
- SMOTE for handling class imbalance
- Real-time fraud detection system
- Web application deployment using Streamlit

---

## 👨‍💻 Author

**Ayush Mishra**

Developed as part of the **CodSoft Machine Learning Internship**.

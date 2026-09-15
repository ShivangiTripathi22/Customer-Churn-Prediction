# 🏦 Customer Churn Prediction

An end-to-end machine learning project that predicts whether a bank customer is likely to churn based on demographic and account-related information.

## 📌 Project Overview

Customer churn is an important business problem for banks because retaining existing customers can be more cost-effective than acquiring new ones.

This project uses machine learning to estimate the probability that a customer will leave the bank. An interactive Streamlit application allows users to enter customer details and receive a real-time churn prediction.

## 🎯 Objectives

* Analyze customer characteristics associated with churn.
* Build a machine learning model to predict customer churn.
* Evaluate the model using appropriate classification metrics.
* Develop an interactive web application for real-time predictions.
* Provide business-oriented insights to support customer retention.

## 📊 Dataset

The dataset contains information about **10,000 bank customers**.

The target variable is:

* `Exited = 0` → Customer stayed
* `Exited = 1` → Customer churned

### Features Used

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Member Status
* Estimated Salary

## 🤖 Machine Learning

The project includes data preprocessing and machine learning modeling using **Scikit-learn**.

The trained model is saved as:

```text
customer_churn_model.pkl
```

The saved model is loaded by the Streamlit application to generate predictions for new customers.

## 📈 Model Performance

The model achieved the following results on the evaluation dataset:

Metric	            Score
Accuracy	        86.55%
Precision	        76.95%
Recall	            48.40%
F1 Score	        59.43%
ROC-AUC	            87.27%

The ROC-AUC score of 87.27% indicates strong overall ability of the model to distinguish between customers who churn and those who stay.

## 🖥️ Streamlit Application

The application provides an interactive dashboard where users can enter:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Member Status
* Estimated Salary

The application then displays:

* Churn / Stay prediction
* Estimated churn probability
* Customer age summary
* Recommended retention action

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Tableau

## 📁 Project Structure

```text
Customer-Churn-prediction-Project/
│
├── app.py
├── customer_churn_model.pkl
├── requirements.txt
├── README.md
└── European_Bank.csv
```

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/ShivangiTripathi22/Customer-Churn-prediction-Project.git
```

Navigate to the project directory:

```bash
cd Customer-Churn-prediction-Project
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Demo

The Streamlit deployment link will be added here after deployment.

## 📌 Key Takeaway

This project demonstrates an end-to-end machine learning workflow, from customer data preprocessing and churn prediction to deployment through an interactive Streamlit application.

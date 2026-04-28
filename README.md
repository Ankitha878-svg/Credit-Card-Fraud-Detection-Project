# Credit-Card-Fraud-Detection-Project

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The goal is to identify anomalous transaction patterns and classify transactions as **legitimate or fraudulent**, helping in improving **financial risk detection and control systems**.

---

## 🎯 Objective

To build a predictive model that can:

* Identify fraudulent transactions in real-time transaction data
* Improve risk detection in financial systems
* Support data-driven decision-making for fraud prevention

---

## 🧠 Business Relevance

* Helps reduce **financial risk and fraud losses**
* Enhances **transaction monitoring and control mechanisms**
* Supports **risk-based decision-making in financial systems**
* Improves **operational efficiency in fraud detection processes**

---

## 📊 Dataset Information

* Source: Credit Card Transaction Dataset
* Records: 284,807 transactions
* Features: 30 numerical features (PCA transformed) + Class label
* Target Variable:

  * `0` → Legitimate Transaction
  * `1` → Fraudulent Transaction

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Logistic Regression
* StandardScaler

---

## 🔄 Workflow

### 1. Data Collection & Exploration

* Loaded dataset using Pandas
* Explored structure, missing values, and class distribution
* Identified highly imbalanced dataset

---

### 2. Data Preprocessing

* Separated legitimate and fraudulent transactions
* Applied **undersampling technique** to balance dataset
* Created a balanced dataset for better model performance

---

### 3. Feature Engineering

* Split data into features (X) and target (Y)
* Standardized features using **StandardScaler**

---

### 4. Model Building

* Applied **Logistic Regression model**
* Trained model on training dataset
* Tuned using increased iterations for convergence

---

### 5. Model Evaluation

* Evaluated using Accuracy Score

**Results:**

* Training Accuracy: ~95%
* Test Accuracy: ~91%

---

## 📈 Key Insights

* Transaction data is highly imbalanced, requiring resampling techniques
* Fraudulent transactions show distinguishable patterns in data distribution
* Logistic Regression performs effectively for baseline fraud detection
* Data preprocessing significantly impacts model performance

---

## 🔐 Risk & Control Perspective

* Helps identify **anomalous financial behavior patterns**
* Supports **fraud risk mitigation strategies**
* Demonstrates importance of **data validation and imbalance handling**
* Enhances understanding of **control failures in transaction systems**

---

## 🚀 Future Improvements

* Implement advanced models (Random Forest, XGBoost)
* Use anomaly detection techniques
* Apply real-time fraud detection system
* Handle imbalance using SMOTE instead of undersampling

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to detect fraudulent transactions and support financial risk management by identifying anomalies in transactional data.

---

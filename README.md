# 🚗 UK Road Safety — Accident Severity Prediction

**Epsilon AI Certified Data Science Professional — Final Project**

Done By: Mohamed Momen

---

## 📌 Project Overview

Real-world datasets are often noisy, incomplete, and inconsistent. This project focuses on analyzing a large-scale **UK Road Safety dataset (STATS19, 2005–2017)** to predict accident severity and uncover the conditions most associated with serious and fatal outcomes on the road.

The dataset contains more than **2,000,000 records**, including accident location, road type, speed limit, weather, light conditions, junction details, and casualty information. The goal of this project is to apply data cleaning, preprocessing, feature engineering, exploratory data analysis (EDA), feature selection, and machine learning / deep learning modeling to predict whether an accident is **Slight** or **Serious/Fatal**, then deploy the final model as a live web application.

This project is part of the **Certified Data Science Professional (CDSP)** program at **[Epsilon AI Academy](https://github.com/Epsilon-AI)**.

---

## 🔗 Links

| Resource | Link |
|---|---|
| **GitHub Repo** | [UK-Road-Safety-Accident-Severity-Prediction](https://github.com/eng-mohamedmomen/UK-Road-Safety-Accident-Severity-Prediction) |
| **LinkedIn Post** | _[add link here after posting]_ |
| **Dataset (Kaggle)** | UK Road Safety — STATS19 Accident Data (2005–2017) |
| **Live Web App** | _[add Streamlit Cloud link here after deployment]_ |
| **Epsilon AI Academy Main Repo** | [Epsilon AI Academy](https://github.com/Epsilon-AI) |

---

## ❓ Questions Answered by Analysis

1. What is the overall distribution of accident severity?
2. How have accidents trended over the years 2005–2017?
3. At what hours and days do most accidents occur?
4. What is the distribution of speed limits where accidents occur?
5. Does speed limit affect accident severity?
6. How do weather and road surface conditions affect severity?
7. Are rural accidents more severe than urban accidents?
8. How does light condition relate to accident severity?
9. What combination of road type, speed limit, and urban/rural area produces the most severe accidents?
10. Does junction type combined with light conditions create higher severity accidents?

---

## 🎯 Objectives

- Data cleaning and preprocessing (raw 34 columns → clean 22-column dataset, 0 nulls)
- Feature engineering (`Hour`, `Month`, `Season`, `is_rush_hour`, `is_night`, `is_weekend`)
- Univariate, bivariate, and multivariate exploratory analysis
- Feature selection using Filter (Chi-Square), Wrapper (RFE), and Embedded (Random Forest) methods
- Model training and comparison: Logistic Regression, Decision Tree, XGBoost, Naive Bayes, and a PyTorch Neural Network
- Handling severe class imbalance (84.7% Slight vs. 15.3% Serious/Fatal)
- Hyperparameter tuning with GridSearchCV
- Model validation and evaluation against project criteria (Precision ≥ 0.3, Recall ≥ 0.3)
- Deployment as an interactive Streamlit web application

---

## 🛠️ Tools & Stack

- **Language:** Python 3.12
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Deep Learning:** PyTorch
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Environment:** Google Colab (GPU-accelerated, T4)

---

## 📊 Key Insights

- The vast majority of accidents are **Slight (84.7%)**, while Serious and Fatal accidents make up only **15.3%**, creating a significant class imbalance challenge for modeling
- Accidents peak sharply during **morning (08:00) and evening (17:00) rush hours**
- The **highest rate** of serious and fatal accidents occurs on roads with a **60 mph speed limit** (>22%), despite the highest accident *volume* occurring at 30 mph
- **Rural areas** experience a notably higher rate of serious and fatal accidents (≈19%) compared to **urban areas** (≈13%), despite lower overall accident frequency
- **Darkness without street lighting** is associated with the highest severity rate among all light conditions (≈25%)
- Single carriageways with a 70 mph speed limit are the **single deadliest road configuration** in the dataset, with a severity rate reaching 0.50
- A full-dataset **PyTorch neural network** (trained with `pos_weight` to handle class imbalance instead of SMOTE) outperformed every classical baseline by learning from the complete 2-million-row dataset rather than a sample

---

## 📈 Final Model Performance

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.208 | 0.593 | 0.308 |
| Decision Tree | 0.215 | 0.549 | 0.309 |
| XGBoost | 0.228 | 0.615 | 0.332 |
| Naive Bayes | 0.201 | 0.484 | 0.284 |
| **PyTorch Neural Network (Full Dataset)** | **0.285** | **0.300** | **0.292** |

> A full technical breakdown of the modeling challenges, the class imbalance problem, and why this ceiling exists in the data is available in [`Project_Report_Challenges_and_Findings.md`](./Project_Report_Challenges_and_Findings.md).

---

## 📁 Repository Structure

```
UK-Road-Safety-Accident-Severity-Prediction/
│
├── Final_Project.ipynb                       # Full notebook: cleaning → EDA → modeling → deployment
├── app.py                                     # Streamlit web application
├── requirements.txt                           # Python dependencies for deployment
│
├── model.pkl                                  # Trained PyTorch model (joblib wrapper)
├── model_weights.pth                          # PyTorch model weights
├── scaler.pkl                                 # StandardScaler fitted on training data
├── feature_columns.pkl                        # Ordered list of model input features
├── ordinal_mappings.pkl                       # Ordinal encoding dictionaries
├── threshold.pkl                              # Optimal decision threshold
├── input_dim.pkl                              # Model input dimension
│
├── Dataset_Pipeline_Report.md                 # Full column-by-column data cleaning report
├── Project_Report_Challenges_and_Findings.md  # Technical report on challenges & findings
└── README.md                                  # This file
```

---

## 🚀 Running the App Locally

```bash
git clone https://github.com/eng-mohamedmomen/UK-Road-Safety-Accident-Severity-Prediction.git
cd UK-Road-Safety-Accident-Severity-Prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🙏 Acknowledgments

This project was completed as part of the **Certified Data Science Professional (CDSP)** program at **[Epsilon AI Academy](https://github.com/Epsilon-AI)**.

Special thanks to the Epsilon AI Academy instructors and the official course main repository for the guidance and structure that shaped this project.

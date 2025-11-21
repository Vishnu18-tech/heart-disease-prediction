# Heart Disease Prediction using Logistic Regression

This project predicts the **10-year Coronary Heart Disease (CHD)** risk using the Framingham Heart Study dataset.  
A Logistic Regression model is trained using cleaned and preprocessed features, and evaluated with multiple ML metrics.

---

## 📊 Model Performance

- **Accuracy:** 0.65  
- **ROC-AUC:** 0.71  
- **CHD Precision:** 0.24  
- **CHD Recall:** 0.65  
- **F1-score (CHD class):** 0.35  

These results show moderate CHD detection ability with good separability (AUC), but lower precision due to data imbalance.

---

## 📈 ROC Curve

![ROC Curve](images/roc_curve.png)

---

## 📋 Confusion Matrix & Other Metrics

![Metrics](images/metrics_screenshot.png)

---

## 🧠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🔬 Steps Performed

1. Data cleaning and handling missing values  
2. Feature scaling using `StandardScaler`  
3. Train-test split  
4. Logistic Regression with `class_weight='balanced'`  
5. Evaluation using:
   - Accuracy  
   - ROC-AUC  
   - Confusion Matrix  
   - Precision, Recall, F1-score  

---

## 📌 Conclusion

The logistic regression model shows decent performance, achieving a ROC-AUC score of 0.71, which indicates good separability between positive and negative CHD cases.
However, there is still room for improvement. Model performance can be enhanced through:

SMOTE oversampling to handle class imbalance

Feature engineering to extract more meaningful predictors

Advanced machine learning models such as Random Forest, XGBoost, or Gradient Boosting

Hyperparameter tuning to optimize model performance

---

## 🚀 How to Run
Install dependencies:

```bash
pip install -r requirements.txt

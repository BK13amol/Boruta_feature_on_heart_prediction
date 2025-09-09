# ❤️ Heart Disease Prediction using Machine Learning

This project predicts the likelihood of heart disease based on clinical features using various machine learning models.  
We apply **Boruta feature selection** to identify the most relevant features and then train multiple classifiers, comparing their performance.

---

## 📂 Dataset

The dataset used is **heart.csv**, which contains patient data with the following details:

- `age`: Age of the patient  
- `sex`: Gender (1 = male, 0 = female)  
- `cp`: Chest pain type  
- `trestbps`: Resting blood pressure  
- `chol`: Serum cholesterol (mg/dl)  
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
- `restecg`: Resting electrocardiographic results  
- `thalach`: Maximum heart rate achieved  
- `exang`: Exercise-induced angina (1 = yes, 0 = no)  
- `oldpeak`: Depression induced by exercise relative to rest  
- `slope`: Slope of the peak exercise ST segment  
- `ca`: Number of major vessels colored by fluoroscopy  
- `thal`: Thalassemia test result  
- `target`: Presence of heart disease (1 = disease, 0 = no disease)  

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```
Install required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

- 👉 numpy
- 👉 pandas
- 👉 matplotlib
- 👉 seaborn
- 👉 scikit-learn
- 👉 xgboost
- 👉 boruta

---

## 🚀 Project Workflow

### 1. Data Loading & Preprocessing
- Load dataset from heart.csv
- Split into train/test sets
- Apply Boruta feature selection to choose important features


### 2. Model Training

👉 Train multiple models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- XGBoost


### 3. Evaluation

- Accuracy scores
- Classification reports
- Confusion matrices (heatmaps)
- ROC curves and AUC


### 4. Comparison

- Visualize accuracy comparison across models

---

## 📊 Results

- Feature selection is done using BorutaPy with Random Forest.
- Model performance is compared on accuracy and AUC.
- Example output includes:
  - Confusion matrices
  - ROC curves for each model
  - Accuracy comparison bar chart

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for more robust evaluation
- Deployment with Flask/Django/Streamlit
- Integration with SHAP for explainable AI

- - -

## 🙌 Acknowledgments

- **Dataset**: [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
- **Libraries & Tools**:  
  - [Scikit-learn](https://scikit-learn.org/stable/)  
  - [XGBoost](https://xgboost.readthedocs.io/)  
  - [BorutaPy](https://github.com/scikit-learn-contrib/boruta_py)  
  - [Matplotlib](https://matplotlib.org/)  
  - [Seaborn](https://seaborn.pydata.org/)  
- Thanks to the **open-source community** for making datasets and tools available for research and learning.  

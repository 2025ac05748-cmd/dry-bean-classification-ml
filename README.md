# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for identifying different varieties of dry beans using numerical features extracted from bean images. The implemented models are Logistic Regression, Decision Tree, K-Nearest Neighbor (KNN), Gaussian Naive Bayes, and Random Forest.

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

The Dry Bean Dataset contains measurements of 13,611 dry bean grains belonging to seven different varieties. The dataset contains 16 input features and one target class attribute.

The seven classes are:

* Seker
* Barbunya
* Bombay
* Cali
* Dermason
* Horoz
* Sira

The input features describe geometric and shape characteristics of the beans, including Area, Perimeter, Major Axis Length, Minor Axis Length, Aspect Ratio, Eccentricity, Convex Area, Equivalent Diameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, and ShapeFactor4.

The dataset contains no missing values.

## 3. GitHub Repository Link

YOUR_GITHUB_REPOSITORY_LINK

## 4. Models Used

The following five classification models were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier

### Evaluation Metrics

The models were evaluated using:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model Name       | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.9214 | 0.9934 |    0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree       |   0.8917 | 0.9330 |    0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN                 |   0.9166 | 0.9812 |    0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes         |   0.7639 | 0.9644 |    0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Random Forest       |   0.9218 | 0.9917 |    0.9219 | 0.9218 | 0.9217 | 0.9054 |

## 5. Model Performance Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 92.14% and the highest AUC of 0.9934. Its precision, recall, F1 score, and MCC were also high, showing that it performed well for the multiclass classification problem.

### Decision Tree

The Decision Tree achieved an accuracy of 89.17%. Its performance was lower than Logistic Regression, KNN, and Random Forest across the main classification metrics.

### KNN

KNN achieved an accuracy of 91.66%. It performed well overall, with an F1 score of 0.9168 and MCC of 0.8992. Its performance was close to Logistic Regression and Random Forest.

### Naive Bayes

Naive Bayes achieved an accuracy of 76.39%, which was considerably lower than the other models. Although its AUC was 0.9644, its accuracy, F1 score, and MCC were lower.

### Random Forest

Random Forest achieved the highest accuracy of 92.18%, highest precision of 92.19%, highest recall of 92.18%, and highest MCC of 0.9054. Its F1 score of 0.9217 was also very close to Logistic Regression.

### Overall Winner

Random Forest is selected as the overall winner for this dataset because it achieved the highest accuracy, precision, recall, and MCC among the implemented models. Logistic Regression achieved the highest AUC, so Random Forest was not the best model on every individual metric.

## 6. Streamlit Application

The Streamlit application provides:

* CSV test-data upload
* Machine learning model selection
* Display of Accuracy, AUC, Precision, Recall, F1 Score, and MCC
* Confusion matrix
* Classification report
* Results for the selected model

The application can be used to evaluate the uploaded test dataset using any of the implemented classification models.

## 7. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── Dry_Bean_Dataset.xlsx
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── test_labels.pkl
```

## 8. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* OpenPyXL

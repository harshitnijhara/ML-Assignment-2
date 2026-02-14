# ML-Assignment-2
Breast cancer classification ML model comparison
Problem Statement

The objective of this study is to develop and evaluate multiple machine learning classification models for predicting whether a breast tumor is malignant or benign using the Breast Cancer dataset. The performance of different models is compared using evaluation metrics such as Accuracy, AUC, Precision, Recall, and Matthews Correlation Coefficient (MCC). The aim is to identify the most suitable model for breast cancer diagnosis and improve decision-making in the medical field.

Dataset Description

The Breast Cancer Wisconsin dataset is a widely used dataset in machine learning and medical diagnosis. It contains 569 instances of breast tumor samples with 30 numerical features extracted from digitized images of breast cell nuclei. These features include radius, texture, perimeter, area, smoothness, concavity, symmetry, and fractal dimension, which help in distinguishing between malignant and benign tumors. The target variable represents the type of tumor, where malignant tumors are encoded as 1 and benign tumors are encoded as 0. The dataset is well structured and suitable for evaluating classification algorithms due to its balanced nature and reliable feature representation.

Models Used and Comparison

In this study, six machine learning models were implemented and evaluated. These include Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes, Random Forest (Ensemble), and XGBoost (Ensemble). The performance of each model was evaluated using Accuracy, AUC, Precision, Recall, and MCC. The results obtained are shown below.

Model Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	MCC
Logistic Regression	0.9649	0.9960	0.9750	0.9286	0.9245
Decision Tree	0.9386	0.9365	0.9070	0.9286	0.8689
Naive Bayes	0.9386	0.9934	1.0000	0.8333	0.8715
Random Forest (Ensemble)	0.9737	0.9942	1.0000	0.9286	0.9442
XGBoost (Ensemble)	0.9649	0.9964	1.0000	0.9048	0.9258
KNN	0.9561	0.9823	0.9744	0.9048	0.9058
Observations on Model Performance

The observations regarding the performance of each model are summarized below.

Observation Table
ML Model Name	Observation about model performance
Logistic Regression	Logistic Regression demonstrated strong performance with high accuracy and AUC. It worked well because the dataset is nearly linearly separable, making it an effective baseline model.
Decision Tree	Decision Tree showed good performance but slightly lower accuracy due to overfitting and sensitivity to variations in the training data.
Naive Bayes	Naive Bayes achieved high precision but lower recall, which indicates that it classified malignant cases conservatively. The assumption of feature independence limited its performance compared to ensemble methods.
Random Forest (Ensemble)	Random Forest achieved the best overall performance among all models, with the highest accuracy and MCC. The ensemble learning approach reduced overfitting and improved generalization.
XGBoost (Ensemble)	XGBoost also showed excellent performance with very high AUC and precision. However, its recall was slightly lower than Random Forest, which indicates that some malignant cases were misclassified.
KNN	KNN performed well but was slightly less accurate than ensemble models. Its performance depends on distance measures and feature scaling, and it can be computationally expensive for larger datasets.

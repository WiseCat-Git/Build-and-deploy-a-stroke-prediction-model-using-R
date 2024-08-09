# Build-and-deploy-a-stroke-prediction-model-using-R

Overview
In this project, you will step into the shoes of an entry-level health data analyst at a leading health organization, helping to build and deploy a stroke prediction model to enhance clinical decision-making. 

Project Scenario

A leading healthcare organization has noticed a trend in an increasing number of patients being diagnosed with strokes. To mitigate this growing problem, the organization has decided to launch a project aimed at predicting the likelihood of a patient getting a stroke based on a variety of health factors. The hospital has access to a vast amount of patient data, including medical history and demographic information, which can be used to build the predictive model.

Once the predictive model is validated and tested, the healthcare organization plans to integrate it into its clinical decision-making process. The model will be used to identify patients who are at high risk of getting a stroke and provide early intervention and prevention measures. Additionally, the model will be used to track the progress of high-risk patients and monitor the impact of preventive measures on reducing the incidence of stroke.

The success of this project will not only help the healthcare organization reduce the number of strokes in its patient population, but it will also position the organization as a leader in the use of advanced analytics and machine learning to improve patient outcomes. The predictive model will be a valuable tool for healthcare providers and patients alike, providing insight into their risk of getting a stroke and the steps they can take to prevent it.   

Project Objectives

Explore the dataset to identify the most important patient and/or clinical characteristics.

Build a well-validated stroke prediction model for clinical use.

Deploy the model to enhance the organization's clinical decision-making.

Your Challenge

Your challenge will be to to build a well-validated stroke prediction model for clinical use using patient characteristics. To do this, you will load, clean, process, analyze, and visualize data. Then, you will build and deploy a prediction model using the cleaned and processed dataset.

In this project, we'll use we'll use data containing 11 clinical features for predicting stroke events.

After you perform your analysis, you will share your findings.

# Project Plan

Import Data and Data Preprocessing

Load data
Data cleaning
Data transformation
Feature engineering
Missing data imputation
Build Prediction Models

Logistic regression
Support vector machine (SVM)
Decision trees
Random forest
XGBoost
Evaluate and Select Prediction Models

Accuracy
Sensitivity
Recall
F-score
AUC (Area Under Curve)
Deploy the Best Prediction Model

Save the model
Deploy using R

Install and Load Required Libraries

# Install required packages if not already installed
if (!requireNamespace("tidyverse", quietly = TRUE)) {
  install.packages("tidyverse")
}

if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret")
}

if (!requireNamespace("pROC", quietly = TRUE)) {
  install.packages("pROC")
}

# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)

1. Data Preprocessing

# Assuming healthcare_dataset_stroke_data is already loaded
# Check the structure of the dataset
str(healthcare_dataset_stroke_data)

# Check for missing values
colSums(is.na(healthcare_dataset_stroke_data))

# Impute missing values (example: using median for 'bmi')
healthcare_dataset_stroke_data$bmi <- ifelse(is.na(healthcare_dataset_stroke_data$bmi), median(healthcare_dataset_stroke_data$bmi, na.rm = TRUE), healthcare_dataset_stroke_data$bmi)

# Convert categorical variables to factors
healthcare_dataset_stroke_data$gender <- as.factor(healthcare_dataset_stroke_data$gender)
healthcare_dataset_stroke_data$ever_married <- as.factor(healthcare_dataset_stroke_data$ever_married)
healthcare_dataset_stroke_data$work_type <- as.factor(healthcare_dataset_stroke_data$work_type)
healthcare_dataset_stroke_data$Residence_type <- as.factor(healthcare_dataset_stroke_data$Residence_type)
healthcare_dataset_stroke_data$smoking_status <- as.factor(healthcare_dataset_stroke_data$smoking_status)
healthcare_dataset_stroke_data$stroke <- as.factor(healthcare_dataset_stroke_data$stroke)

2. Build Prediction Models

# Split data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(healthcare_dataset_stroke_data$stroke, p = .8, list = FALSE, times = 1)
train_data <- healthcare_dataset_stroke_data[trainIndex, ]
test_data <- healthcare_dataset_stroke_data[-trainIndex, ]

# Train logistic regression model
logistic_model <- glm(stroke ~ ., data = train_data, family = binomial)
summary(logistic_model)

# Train SVM model
svm_model <- train(stroke ~ ., data = train_data, method = "svmRadial", trControl = trainControl(method = "cv", number = 10))

# Train decision tree model
tree_model <- train(stroke ~ ., data = train_data, method = "rpart", trControl = trainControl(method = "cv", number = 10))

# Train random forest model
rf_model <- train(stroke ~ ., data = train_data, method = "rf", trControl = trainControl(method = "cv", number = 10))

# Train XGBoost model
xgb_model <- train(stroke ~ ., data = train_data, method = "xgbTree", trControl = trainControl(method = "cv", number = 10))

3.Evaluate and Select Prediction Models

# Evaluate models
logistic_pred <- predict(logistic_model, newdata = test_data, type = "response")
svm_pred <- predict(svm_model, newdata = test_data)
tree_pred <- predict(tree_model, newdata = test_data)
rf_pred <- predict(rf_model, newdata = test_data)
xgb_pred <- predict(xgb_model, newdata = test_data)

# Convert predictions to factors for comparison
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)
logistic_pred_class <- factor(logistic_pred_class, levels = c(0, 1))

# Confusion matrices
confusionMatrix(logistic_pred_class, test_data$stroke)
confusionMatrix(svm_pred, test_data$stroke)
confusionMatrix(tree_pred, test_data$stroke)
confusionMatrix(rf_pred, test_data$stroke)
confusionMatrix(xgb_pred, test_data$stroke)

# Calculate evaluation metrics
eval_metrics <- function(model_pred, true_values) {
  confusion <- confusionMatrix(model_pred, true_values)
  accuracy <- confusion$overall['Accuracy']
  sensitivity <- confusion$byClass['Sensitivity']
  recall <- confusion$byClass['Recall']
  f_score <- confusion$byClass['F1']
  auc <- roc(as.numeric(true_values), as.numeric(model_pred))$auc
  
  return(c(accuracy, sensitivity, recall, f_score, auc))
}

# Compare models
logistic_metrics <- eval_metrics(logistic_pred_class, test_data$stroke)
svm_metrics <- eval_metrics(svm_pred, test_data$stroke)
tree_metrics <- eval_metrics(tree_pred, test_data$stroke)
rf_metrics <- eval_metrics(rf_pred, test_data$stroke)
xgb_metrics <- eval_metrics(xgb_pred, test_data$stroke)

# Create a comparison dataframe
comparison <- data.frame(
  Model = c("Logistic Regression", "SVM", "Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = c(logistic_metrics[1], svm_metrics[1], tree_metrics[1], rf_metrics[1], xgb_metrics[1]),
  Sensitivity = c(logistic_metrics[2], svm_metrics[2], tree_metrics[2], rf_metrics[2], xgb_metrics[2]),
  Recall = c(logistic_metrics[3], svm_metrics[3], tree_metrics[3], rf_metrics[3], xgb_metrics[3]),
  F_Score = c(logistic_metrics[4], svm_metrics[4], tree_metrics[4], rf_metrics[4], xgb_metrics[4]),
  AUC = c(logistic_metrics[5], svm_metrics[5], tree_metrics[5], rf_metrics[5], xgb_metrics[5])
)

print(comparison)

4. Deploy the Best Prediction Model

# Save the model
saveRDS(xgb_model, file = "xgboost_model.rds")

# Load the model for deployment
best_model <- readRDS("xgboost_model.rds")

# Deploy model: Make predictions
new_data <- test_data # Replace with actual new data
predictions <- predict(best_model, new_data)

# Add predictions to the dataframe
new_data$predictions <- predictions
head(new_data)


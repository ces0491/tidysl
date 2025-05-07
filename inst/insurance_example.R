# tidylearn: Insurance Cost Analysis Example
# ===========================================
# This example demonstrates how to use the tidylearn package to analyze
# the medical insurance cost dataset from the assignments.

# Load required libraries
library(tidylearn)
library(dplyr)
library(ggplot2)

# Set seed for reproducibility
set.seed(42)

# Load the insurance data
insurance_data <- read.csv("./data/insurance.csv")

# Convert categorical variables to factors
insurance_data <- insurance_data %>%
  mutate(
    sex = as.factor(sex),
    smoker = as.factor(smoker),
    region = as.factor(region)
  )

# Split the data into training and testing sets (80/20 split)
n <- nrow(insurance_data)
train_idx <- sample(1:n, size = 0.8 * n)
train_data <- insurance_data[train_idx, ]
test_data <- insurance_data[-train_idx, ]

# Part 1: Regression Analysis (Assignment 1)
# ------------------------------------------

# Fit a basic linear model
linear_model <- tl_model(train_data, charges ~ age + sex + bmi + children + smoker + region,
                         method = "linear")

# Examine model summary
summary(linear_model)

# Check model assumptions
assumptions <- tl_check_assumptions(linear_model)
print(assumptions$overall$status)

# Test for significant interactions
interactions <- tl_test_interactions(train_data, charges ~ age + sex + bmi + children + smoker + region,
                                     all_pairs = TRUE)
print(interactions)

# Based on interaction testing, create a model with the BMI-smoker interaction
interaction_model <- tl_model(train_data,
                              charges ~ age + sex + bmi + children + smoker + region + bmi:smoker,
                              method = "linear")

# Create diagnostic dashboard
tl_diagnostic_dashboard(interaction_model)

# Plot the interaction
tl_plot_interaction(interaction_model, "bmi", "smoker")

# Try regularization methods
ridge_model <- tl_model(train_data, charges ~ age + sex + bmi + children + smoker + region,
                        method = "ridge")
lasso_model <- tl_model(train_data, charges ~ age + sex + bmi + children + smoker + region,
                        method = "lasso")

# Compare models with cross-validation
cv_results <- tl_compare_cv(
  data = train_data,
  models = list(
    linear = linear_model,
    interaction = interaction_model,
    ridge = ridge_model,
    lasso = lasso_model
  ),
  folds = 5,
  metrics = c("rmse", "mae", "rsq")
)

# Plot comparison
tl_plot_cv_comparison(cv_results)

# Evaluate the best model on test data
test_metrics <- tl_evaluate(interaction_model, test_data)
print(test_metrics)

# Create a pipeline approach
regression_pipeline <- tl_pipeline(
  data = insurance_data,
  formula = charges ~ .,
  preprocessing = list(
    impute_missing = TRUE,
    standardize = TRUE,
    dummy_encode = TRUE
  ),
  models = list(
    linear = list(method = "linear"),
    interaction = list(method = "linear", formula = charges ~ . + bmi:smoker),
    lasso = list(method = "lasso"),
    ridge = list(method = "ridge"),
    forest = list(method = "forest", ntree = 500)
  ),
  evaluation = list(
    metrics = c("rmse", "mae", "rsq"),
    validation = "cv",
    cv_folds = 5,
    best_metric = "rmse"
  )
)

# Run the pipeline
regression_results <- tl_run_pipeline(regression_pipeline)

# Compare models
tl_compare_pipeline_models(regression_results)

# Get best model
best_regression_model <- tl_get_best_model(regression_results)
summary(best_regression_model)

# Part 2: Classification Analysis (Assignment 2)
# ----------------------------------------------

# Create binary target for classification
insurance_data$charges_binary <- ifelse(insurance_data$charges > median(insurance_data$charges),
                                        "high", "low")
insurance_data$charges_binary <- factor(insurance_data$charges_binary,
                                        levels = c("low", "high"))

# Split data again for classification task
train_idx <- sample(1:n, size = 0.8 * n)
train_data_class <- insurance_data[train_idx, ]
test_data_class <- insurance_data[-train_idx, ]

# Fit logistic regression with LASSO
lasso_logistic <- tl_model(train_data_class, charges_binary ~ age + sex + bmi + children + smoker + region,
                           method = "logistic")

# Fit tree-based models
tree_model <- tl_model(train_data_class, charges_binary ~ age + sex + bmi + children + smoker + region,
                       method = "tree")
forest_model <- tl_model(train_data_class, charges_binary ~ age + sex + bmi + children + smoker + region,
                         method = "forest", ntree = 500)
xgb_model <- tl_model(train_data_class, charges_binary ~ age + sex + bmi + children + smoker + region,
                      method = "xgboost", nrounds = 100, max_depth = 3, eta = 0.1)

# Visualize feature importance
tl_plot_importance(forest_model)
tl_plot_xgboost_importance(xgb_model)

# Plot SHAP values for XGBoost
tl_plot_xgboost_shap_summary(xgb_model)
tl_plot_xgboost_shap_dependence(xgb_model, feature = "age", interaction_feature = "smoker")

# Compare models
class_comparison <- tl_compare_cv(
  data = train_data_class,
  models = list(
    logistic = lasso_logistic,
    tree = tree_model,
    forest = forest_model,
    xgboost = xgb_model
  ),
  folds = 5,
  metrics = c("accuracy", "precision", "recall", "f1", "auc")
)

# Plot comparison
tl_plot_cv_comparison(class_comparison)

# Create classification pipeline
classification_pipeline <- tl_pipeline(
  data = insurance_data,
  formula = charges_binary ~ .,
  preprocessing = list(
    impute_missing = TRUE,
    standardize = TRUE,
    dummy_encode = TRUE
  ),
  models = list(
    logistic = list(method = "logistic"),
    tree = list(method = "tree"),
    forest = list(method = "forest", ntree = 500),
    xgboost = list(method = "xgboost", nrounds = 100, max_depth = 3, eta = 0.1)
  ),
  evaluation = list(
    metrics = c("accuracy", "precision", "recall", "f1", "auc"),
    validation = "cv",
    cv_folds = 5,
    best_metric = "f1"
  )
)

# Run the classification pipeline
classification_results <- tl_run_pipeline(classification_pipeline)

# Compare models
tl_compare_pipeline_models(classification_results)

# Get best model
best_classification_model <- tl_get_best_model(classification_results)
summary(best_classification_model)

# Plot ROC curve
pred_probs <- predict(best_classification_model, test_data_class, type = "prob")
actual <- test_data_class$charges_binary

# If best model is tree-based, plot the tree
if (best_classification_model$spec$method == "tree") {
  tl_plot_tree(best_classification_model)
}

# Create partial dependence plots
if (best_classification_model$spec$method %in% c("tree", "forest", "boost", "xgboost")) {
  # Plot partial dependence for age
  tl_plot_partial_dependence(best_classification_model, "age")

  # Plot partial dependence for bmi
  tl_plot_partial_dependence(best_classification_model, "bmi")

  # Plot partial dependence for smoker
  tl_plot_partial_dependence(best_classification_model, "smoker")
}

# Analyze Variable Interactions
if (best_classification_model$spec$method == "xgboost") {
  # Show interaction between age and smoking status
  tl_plot_xgboost_shap_dependence(best_classification_model, "age", "smoker")

  # Show interaction between BMI and smoking status
  tl_plot_xgboost_shap_dependence(best_classification_model, "bmi", "smoker")
}

# Detect outliers in the data
outliers <- tl_detect_outliers(insurance_data,
                               variables = c("age", "bmi", "children", "charges"),
                               method = "mahalanobis")

# Examine outlier results and plot
print(outliers$outlier_counts$total)
print(outliers$plot)

# Export best models
tl_export_model(best_regression_model, format = "rds", file = "best_regression_model.rds")
tl_export_model(best_classification_model, format = "rds", file = "best_classification_model.rds")

# Create report on findings
cat("Insurance Cost Analysis Results\n")
cat("==============================\n\n")
cat("Regression Task (Predicting Exact Charges):\n")
cat("Best Model:", regression_results$results$best_model_name, "\n")
cat("RMSE on Cross-Validation:", round(min(regression_results$results$metric_values), 2), "\n\n")

cat("Classification Task (Predicting High/Low Charges):\n")
cat("Best Model:", classification_results$results$best_model_name, "\n")
cat("F1 Score on Cross-Validation:", round(max(classification_results$results$metric_values), 4), "\n\n")

cat("Key Findings:\n")
cat("1. Most important factors for predicting insurance charges:\n")
cat("   - Smoking status (strongest predictor)\n")
cat("   - Age (linear relationship with charges)\n")
cat("   - BMI (stronger effect for smokers - significant interaction)\n\n")

cat("2. Classification accuracy for high/low charges category: ",
    round(classification_results$results$metric_values[classification_results$results$best_model_name], 4) * 100,
    "%\n\n", sep = "")

cat("3. Recommendations for reducing insurance costs:\n")
cat("   - Smoking cessation programs would have the largest impact\n")
cat("   - Weight management programs, especially for smokers\n")
cat("   - Age-based premium adjustments are justified by the data\n")

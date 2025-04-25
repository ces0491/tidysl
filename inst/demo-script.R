# tidylearn Complete Workflow Demo
# This script demonstrates a complete workflow using the tidylearn package

# Load necessary packages
library(tidylearn)
library(tidyverse)
library(modeldata)
library(recipes)
library(rsample)

# ---- Data Preparation ----

# Load data (classification example)
data(credit_data)
credit_df <- as_tibble(credit_data)

# Data cleaning and preparation
credit_clean <- credit_df %>%
  mutate(Status = factor(Status)) %>%
  drop_na()

# Create initial train/test split
set.seed(123)
initial_split <- initial_split(credit_clean, prop = 0.8)
train_data <- training(initial_split)
test_data <- testing(initial_split)

# Create a preprocessing recipe
credit_recipe <- recipe(Status ~ ., data = train_data) %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  prep(training = train_data)

# Apply preprocessing
train_processed <- bake(credit_recipe, new_data = train_data)
test_processed <- bake(credit_recipe, new_data = test_data)

# ---- Exploratory Data Analysis ----

# Quick look at the data
glimpse(train_processed)

# Class distribution
train_data %>%
  count(Status) %>%
  mutate(pct = n / sum(n))

# Feature correlations
train_num <- train_data %>% select_if(is.numeric)
cor_matrix <- cor(train_num)
corrplot::corrplot(cor_matrix, method = "circle", type = "upper",
                   tl.col = "black", tl.srt = 45, tl.cex = 0.7)

# ---- Model Training ----

# Train multiple models for comparison

# 1. Logistic Regression
log_model <- tl_model(
  data = train_processed,
  formula = Status ~ .,
  method = "logistic"
)

# 2. Random Forest
forest_model <- tl_model(
  data = train_processed,
  formula = Status ~ .,
  method = "forest",
  is_classification = TRUE,
  ntree = 100
)

# 3. Gradient Boosting
boost_model <- tl_model(
  data = train_processed,
  formula = Status ~ .,
  method = "boost",
  is_classification = TRUE,
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.1
)

# 4. Support Vector Machine
svm_model <- tl_model(
  data = train_processed,
  formula = Status ~ .,
  method = "svm",
  is_classification = TRUE,
  kernel = "radial",
  probability = TRUE
)

# ---- Cross-Validation ----

# Cross-validate the random forest model
forest_cv <- tl_cv(
  data = train_processed,
  formula = Status ~ .,
  method = "forest",
  is_classification = TRUE,
  folds = 5,
  ntree = 100
)

# View cross-validation results
forest_cv$summary

# Plot CV results
tl_plot_cv_results(forest_cv)

# ---- Model Evaluation ----

# Evaluate all models on test data
models <- list(
  Logistic = log_model,
  "Random Forest" = forest_model,
  "Gradient Boosting" = boost_model,
  SVM = svm_model
)

# Individual evaluation
model_metrics <- map_dfr(models, ~tl_evaluate(.x, test_processed), .id = "Model")
model_metrics %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  arrange(desc(f1))

# Model comparison plot
tl_plot_model_comparison(
  log_model, forest_model, boost_model, svm_model,
  new_data = test_processed,
  names = names(models)
)

# ---- Feature Importance ----

# Plot feature importance for each model that supports it
tree_models <- list(
  "Random Forest" = forest_model,
  "Gradient Boosting" = boost_model
)

# Individual importance plots
walk2(tree_models, names(tree_models), function(model, name) {
  print(tl_plot_importance(model) + ggtitle(paste("Feature Importance -", name)))
})

# Compare importance across models
tl_plot_importance_comparison(
  forest_model, boost_model,
  names = c("Random Forest", "Boosting")
)

# ---- Model Visualization ----

# ROC curves
walk2(models, names(models), function(model, name) {
  print(tl_plot_roc(model, test_processed) + ggtitle(paste("ROC Curve -", name)))
})

# Confusion matrices
walk2(models, names(models), function(model, name) {
  print(tl_plot_confusion(model, test_processed) + ggtitle(paste("Confusion Matrix -", name)))
})

# Precision-Recall curves
tl_plot_precision_recall(forest_model, test_processed)

# Calibration plot
tl_plot_calibration(forest_model, test_processed)

# Lift chart
tl_plot_lift(forest_model, test_processed)

# Gain chart
tl_plot_gain(forest_model, test_processed)

# ---- Model Tuning ----

# Find optimal threshold for the best model (forest)
threshold_results <- tl_find_optimal_threshold(
  forest_model,
  test_processed,
  optimize_for = "f1"
)

print(paste("Optimal threshold:", round(threshold_results$optimal_threshold, 3)))
print(paste("Optimal F1 score:", round(threshold_results$optimal_value, 3)))

# ---- Final Predictions ----

# Make predictions with the best model (forest)
class_preds <- predict(forest_model, test_processed, type = "class")
prob_preds <- predict(forest_model, test_processed, type = "prob")

# Create a results data frame
results <- tibble(
  actual = test_processed$Status,
  predicted = class_preds,
  probability = prob_preds[[2]]
) %>%
  mutate(correct = actual == predicted)

# View prediction results
head(results, 20)

# Prediction accuracy
mean(results$correct)

# ---- Partial Dependence Analysis ----

# Choose an important feature from the forest model
important_features <- tl_extract_importance(forest_model) %>%
  arrange(desc(importance)) %>%
  slice_head(n = 3) %>%
  pull(feature)

# Plot partial dependence for the top features
walk(important_features, function(feature) {
  print(tl_plot_partial_dependence(forest_model, feature))
})

# ---- Interactive Dashboard ----

# Launch interactive dashboard (uncomment to run interactively)
# if (requireNamespace("shiny", quietly = TRUE) &&
#     requireNamespace("shinydashboard", quietly = TRUE) &&
#     requireNamespace("DT", quietly = TRUE)) {
#   tl_dashboard(forest_model, test_processed)
# }

# ---- Save Model ----

# Save the best model for future use
saveRDS(forest_model, "best_credit_model.rds")

# How to load the model later
# loaded_model <- readRDS("best_credit_model.rds")

# ---- Conclusions ----

# Print final report on model performance
cat("\n=== Final Model Performance Report ===\n")
cat("Best model: Random Forest\n")
cat("Cross-validation results:\n")
print(forest_cv$summary)
cat("\nTest data performance:\n")
print(tl_evaluate(forest_model, test_processed))
cat("\nTop 5 important features:\n")
print(tl_extract_importance(forest_model) %>%
        arrange(desc(importance)) %>%
        slice_head(n = 5))
cat("\nOptimal threshold for F1 score:", threshold_results$optimal_threshold, "\n")
cat("=== End of Report ===\n")

context("Core functionality")

# Load test data
library(MASS)
data(Boston)
set.seed(123)
train_indices <- sample(1:nrow(Boston), 0.8 * nrow(Boston))
train_data <- Boston[train_indices, ]
test_data <- Boston[-train_indices, ]

# Classification data
data(iris)
# Make binary for simpler tests
binary_iris <- iris[iris$Species != "virginica", ]
binary_iris$Species <- factor(binary_iris$Species)
set.seed(123)
train_indices <- sample(1:nrow(binary_iris), 0.8 * nrow(binary_iris))
train_iris <- binary_iris[train_indices, ]
test_iris <- binary_iris[-train_indices, ]

test_that("tl_model creates model objects", {
  # Regression model
  lm_model <- tl_model(train_data, medv ~ ., method = "linear")

  # Check object structure
  expect_s3_class(lm_model, "tidylearn_model")
  expect_s3_class(lm_model, "tidylearn_linear")
  expect_true("spec" %in% names(lm_model))
  expect_true("fit" %in% names(lm_model))
  expect_true("data" %in% names(lm_model))

  # Check model specifications
  expect_equal(lm_model$spec$method, "linear")
  expect_false(lm_model$spec$is_classification)
  expect_equal(lm_model$spec$response_var, "medv")

  # Classification model
  log_model <- tl_model(train_iris, Species ~ ., method = "logistic")

  # Check object structure
  expect_s3_class(log_model, "tidylearn_model")
  expect_s3_class(log_model, "tidylearn_logistic")

  # Check model specifications
  expect_equal(log_model$spec$method, "logistic")
  expect_true(log_model$spec$is_classification)
  expect_equal(log_model$spec$response_var, "Species")
})

test_that("predict method works", {
  # Regression model
  lm_model <- tl_model(train_data, medv ~ ., method = "linear")

  # Test predictions
  preds <- predict(lm_model, test_data)
  expect_s3_class(preds, "tbl_df")
  expect_equal(nrow(preds), nrow(test_data))
  expect_equal(names(preds), "prediction")
  expect_true(is.numeric(preds$prediction))

  # Classification model
  log_model <- tl_model(train_iris, Species ~ ., method = "logistic")

  # Test class predictions
  class_preds <- predict(log_model, test_iris, type = "class")
  expect_s3_class(class_preds, "tbl_df")
  expect_equal(nrow(class_preds), nrow(test_iris))
  expect_equal(names(class_preds), "prediction")
  expect_true(is.factor(class_preds$prediction))

  # Test probability predictions
  prob_preds <- predict(log_model, test_iris, type = "prob")
  expect_s3_class(prob_preds, "tbl_df")
  expect_equal(nrow(prob_preds), nrow(test_iris))
  expect_equal(ncol(prob_preds), length(levels(train_iris$Species)))
  expect_true(all(colnames(prob_preds) %in% levels(train_iris$Species)))
  expect_true(all(prob_preds >= 0 & prob_preds <= 1))
})

test_that("tl_evaluate computes metrics correctly", {
  # Regression model
  lm_model <- tl_model(train_data, medv ~ ., method = "linear")

  # Evaluate on test data
  metrics <- tl_evaluate(lm_model, test_data)

  # Check structure
  expect_s3_class(metrics, "tbl_df")
  expect_true("metric" %in% names(metrics))
  expect_true("value" %in% names(metrics))

  # Check specific metrics
  expect_true("rmse" %in% metrics$metric)
  expect_true("mae" %in% metrics$metric)
  expect_true("rsq" %in% metrics$metric)

  # Classification model
  log_model <- tl_model(train_iris, Species ~ ., method = "logistic")

  # Evaluate on test data
  metrics <- tl_evaluate(log_model, test_iris)

  # Check structure
  expect_s3_class(metrics, "tbl_df")

  # Check specific metrics
  expect_true("accuracy" %in% metrics$metric)
  expect_true("precision" %in% metrics$metric)
  expect_true("recall" %in% metrics$metric)
  expect_true("f1" %in% metrics$metric)
  expect_true("auc" %in% metrics$metric)
})

test_that("tl_cv performs cross-validation", {
  # Cross-validation for regression
  lm_cv <- tl_cv(train_data, medv ~ ., method = "linear", folds = 3)

  # Check structure
  expect_type(lm_cv, "list")
  expect_true("fold_metrics" %in% names(lm_cv))
  expect_true("summary" %in% names(lm_cv))

  # Check fold_metrics
  expect_s3_class(lm_cv$fold_metrics, "tbl_df")
  expect_true("fold" %in% names(lm_cv$fold_metrics))
  expect_true("metric" %in% names(lm_cv$fold_metrics))
  expect_true("value" %in% names(lm_cv$fold_metrics))

  # Check summary
  expect_s3_class(lm_cv$summary, "tbl_df")
  expect_true("metric" %in% names(lm_cv$summary))
  expect_true("mean_value" %in% names(lm_cv$summary))
  expect_true("sd_value" %in% names(lm_cv$summary))

  # Cross-validation for classification
  log_cv <- tl_cv(train_iris, Species ~ ., method = "logistic", folds = 3)

  # Check structure
  expect_type(log_cv, "list")

  # Check classification metrics
  expect_true("accuracy" %in% log_cv$summary$metric)
  expect_true("f1" %in% log_cv$summary$metric)
})

test_that("print and summary methods work", {
  # Regression model
  lm_model <- tl_model(train_data, medv ~ ., method = "linear")

  # Check print method
  expect_output(print(lm_model), "Tidylearn linear model")
  expect_output(print(lm_model), "Formula:")
  expect_output(print(lm_model), "Type: Regression")
  expect_output(print(lm_model), "Training metrics:")

  # Check summary method
  expect_output(summary(lm_model), "Tidylearn linear model")
  expect_output(summary(lm_model), "Evaluation metrics:")
  expect_output(summary(lm_model), "Model details:")

  # Classification model
  log_model <- tl_model(train_iris, Species ~ ., method = "logistic")

  # Check print method
  expect_output(print(log_model), "Tidylearn logistic model")
  expect_output(print(log_model), "Type: Classification")

  # Check summary method
  expect_output(summary(log_model), "Tidylearn logistic model")
})

test_that("tl_check_packages works", {
  # Test with installed packages
  expect_true(tl_check_packages("stats", error = FALSE))

  # Test with non-existent package (should not error with error = FALSE)
  expect_false(tl_check_packages("non_existent_package", error = FALSE))

  # Test with non-existent package (should error with error = TRUE)
  expect_error(tl_check_packages("non_existent_package", error = TRUE))
})

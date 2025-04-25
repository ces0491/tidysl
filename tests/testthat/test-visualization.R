context("Visualization functions")

# Load test data
library(MASS)
data(Boston)
set.seed(123)
train_indices <- sample(1:nrow(Boston), 0.8 * nrow(Boston))
train_data <- Boston[train_indices, ]
test_data <- Boston[-train_indices, ]

# Classification data
data(iris)
set.seed(123)
train_indices <- sample(1:nrow(iris), 0.8 * nrow(iris))
train_iris <- iris[train_indices, ]
test_iris <- iris[-train_indices, ]

# Binary classification data
binary_iris <- iris[iris$Species != "virginica", ]
binary_iris$Species <- factor(binary_iris$Species)
set.seed(123)
train_indices <- sample(1:nrow(binary_iris), 0.8 * nrow(binary_iris))
train_binary <- binary_iris[train_indices, ]
test_binary <- binary_iris[-train_indices, ]

# Train models for testing
lm_model <- tl_model(train_data, medv ~ ., method = "linear")
log_model <- tl_model(train_binary, Species ~ ., method = "logistic")

test_that("regression visualization functions work", {
  # Actual vs predicted plot
  p1 <- tl_plot_actual_predicted(lm_model, test_data)
  expect_s3_class(p1, "ggplot")

  # Residuals plot
  p2 <- tl_plot_residuals(lm_model, test_data)
  expect_s3_class(p2, "ggplot")

  # Residuals histogram
  p3 <- tl_plot_residuals(lm_model, type = "histogram")
  expect_s3_class(p3, "ggplot")

  # Diagnostic plots
  diagnostics <- tl_plot_diagnostics(lm_model)
  expect_type(diagnostics, "list")
  expect_s3_class(diagnostics$residuals_vs_fitted, "ggplot")
  expect_s3_class(diagnostics$qq, "ggplot")
  expect_s3_class(diagnostics$scale_location, "ggplot")
  expect_s3_class(diagnostics$residuals_vs_leverage, "ggplot")

  # Intervals plot
  p4 <- tl_plot_intervals(lm_model, test_data)
  expect_s3_class(p4, "ggplot")
})

test_that("classification visualization functions work", {
  # ROC curve
  p1 <- tl_plot_roc(log_model, test_binary)
  expect_s3_class(p1, "ggplot")

  # Confusion matrix
  p2 <- tl_plot_confusion(log_model, test_binary)
  expect_s3_class(p2, "ggplot")

  # Precision-recall curve
  p3 <- tl_plot_precision_recall(log_model, test_binary)
  expect_s3_class(p3, "ggplot")

  # Calibration curve
  p4 <- tl_plot_calibration(log_model, test_binary)
  expect_s3_class(p4, "ggplot")

  # Lift chart
  p5 <- tl_plot_lift(log_model, test_binary)
  expect_s3_class(p5, "ggplot")

  # Gain chart
  p6 <- tl_plot_gain(log_model, test_binary)
  expect_s3_class(p6, "ggplot")
})

test_that("plot method dispatches correctly", {
  # For regression models
  p1 <- plot(lm_model, type = "diagnostics")
  expect_type(p1, "list")

  p2 <- plot(lm_model, type = "residuals")
  expect_s3_class(p2, "ggplot")

  p3 <- plot(lm_model, type = "actual_vs_predicted")
  expect_s3_class(p3, "ggplot")

  # For classification models
  p4 <- plot(log_model, type = "roc")
  expect_s3_class(p4, "ggplot")

  p5 <- plot(log_model, type = "confusion")
  expect_s3_class(p5, "ggplot")

  p6 <- plot(log_model, type = "precision_recall")
  expect_s3_class(p6, "ggplot")

  # Error for invalid type
  expect_error(plot(lm_model, type = "invalid_type"))
  expect_error(plot(log_model, type = "invalid_type"))
})

test_that("model comparison functions work", {
  # Train additional models
  ridge_model <- tl_model(train_data, medv ~ ., method = "ridge")

  # Model comparison
  p1 <- tl_plot_model_comparison(
    lm_model, ridge_model,
    new_data = test_data,
    names = c("Linear", "Ridge")
  )
  expect_s3_class(p1, "ggplot")

  # Train tree models if available
  if (requireNamespace("rpart", quietly = TRUE) &&
      requireNamespace("randomForest", quietly = TRUE)) {

    tree_model <- tl_model(train_data, medv ~ ., method = "tree", is_classification = FALSE)
    forest_model <- tl_model(train_data, medv ~ ., method = "forest", is_classification = FALSE, ntree = 50)

    # Feature importance comparison
    p2 <- tl_plot_importance_comparison(
      tree_model, forest_model,
      names = c("Tree", "Forest")
    )
    expect_s3_class(p2, "ggplot")
  }

  # Classification model comparison
  if (requireNamespace("e1071", quietly = TRUE)) {
    svm_model <- tl_model(train_binary, Species ~ ., method = "svm", is_classification = TRUE)

    p3 <- tl_plot_model_comparison(
      log_model, svm_model,
      new_data = test_binary,
      names = c("Logistic", "SVM")
    )
    expect_s3_class(p3, "ggplot")
  }
})

test_that("CV results plotting works", {
  # Perform cross-validation
  lm_cv <- tl_cv(train_data, medv ~ ., method = "linear", folds = 3)

  # Plot CV results
  p <- tl_plot_cv_results(lm_cv)
  expect_s3_class(p, "ggplot")

  # Plot with specific metrics
  p2 <- tl_plot_cv_results(lm_cv, metrics = c("rmse", "rsq"))
  expect_s3_class(p2, "ggplot")
})

test_that("dashboard creation works if required packages are available", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("shinydashboard")
  skip_if_not_installed("DT")

  # Only test that creation doesn't error
  expect_error(tl_dashboard(lm_model, test_data, launch.browser = FALSE), NA)
})

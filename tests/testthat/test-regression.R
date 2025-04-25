context("Regression models")

# Load test data
library(MASS)
data(Boston)
set.seed(123)
train_indices <- sample(1:nrow(Boston), 0.8 * nrow(Boston))
train_data <- Boston[train_indices, ]
test_data <- Boston[-train_indices, ]

test_that("linear regression model works", {
  # Train model
  lm_model <- tl_model(train_data, medv ~ ., method = "linear")

  # Check model object
  expect_s3_class(lm_model$fit, "lm")

  # Check predictions
  preds <- predict(lm_model, test_data)
  expect_s3_class(preds, "tbl_df")
  expect_equal(nrow(preds), nrow(test_data))

  # Check evaluation
  metrics <- tl_evaluate(lm_model, test_data)
  expect_true("rmse" %in% metrics$metric)
  expect_true("rsq" %in% metrics$metric)

  # Check plotting functions
  p1 <- tl_plot_actual_predicted(lm_model, test_data)
  expect_s3_class(p1, "ggplot")

  p2 <- tl_plot_residuals(lm_model, test_data)
  expect_s3_class(p2, "ggplot")

  p3 <- tl_plot_diagnostics(lm_model)
  expect_type(p3, "list")
  expect_s3_class(p3$residuals_vs_fitted, "ggplot")
})

test_that("polynomial regression model works", {
  # Train model
  poly_model <- tl_model(train_data, medv ~ ., method = "polynomial", degree = 2)

  # Check model object
  expect_s3_class(poly_model$fit, "lm")
  expect_equal(attr(poly_model$fit, "poly_degree"), 2)

  # Check predictions
  preds <- predict(poly_model, test_data)
  expect_s3_class(preds, "tbl_df")

  # Check evaluation
  metrics <- tl_evaluate(poly_model, test_data)
  expect_true("rmse" %in% metrics$metric)
})

test_that("ridge regression model works", {
  skip_if_not_installed("glmnet")

  # Train model
  ridge_model <- tl_model(train_data, medv ~ ., method = "ridge")

  # Check model object
  expect_s3_class(ridge_model$fit, "glmnet")
  expect_equal(attr(ridge_model$fit, "alpha"), 0)

  # Check predictions
  preds <- predict(ridge_model, test_data)
  expect_s3_class(preds, "tbl_df")

  # Check regularization path plot
  p <- tl_plot_regularization_path(ridge_model)
  expect_s3_class(p, "ggplot")
})

test_that("lasso regression model works", {
  skip_if_not_installed("glmnet")

  # Train model
  lasso_model <- tl_model(train_data, medv ~ ., method = "lasso")

  # Check model object
  expect_s3_class(lasso_model$fit, "glmnet")
  expect_equal(attr(lasso_model$fit, "alpha"), 1)

  # Check predictions
  preds <- predict(lasso_model, test_data)
  expect_s3_class(preds, "tbl_df")

  # Check feature importance
  p <- tl_plot_importance_regularized(lasso_model)
  expect_s3_class(p, "ggplot")
})

test_that("elastic net regression model works", {
  skip_if_not_installed("glmnet")

  # Train model
  elastic_net_model <- tl_model(train_data, medv ~ ., method = "elastic_net", alpha = 0.5)

  # Check model object
  expect_s3_class(elastic_net_model$fit, "glmnet")
  expect_equal(attr(elastic_net_model$fit, "alpha"), 0.5)

  # Check predictions
  preds <- predict(elastic_net_model, test_data)
  expect_s3_class(preds, "tbl_df")
})

test_that("prediction intervals work", {
  # Train model
  lm_model <- tl_model(train_data, medv ~ ., method = "linear")

  # Check prediction intervals
  intervals <- tl_prediction_intervals(lm_model, test_data)

  expect_s3_class(intervals, "tbl_df")
  expect_equal(nrow(intervals), nrow(test_data))
  expect_true("prediction" %in% names(intervals))
  expect_true("conf_lower" %in% names(intervals))
  expect_true("conf_upper" %in% names(intervals))
  expect_true("pred_lower" %in% names(intervals))
  expect_true("pred_upper" %in% names(intervals))

  # Check interval plot
  p <- tl_plot_intervals(lm_model, test_data)
  expect_s3_class(p, "ggplot")
})

test_that("regression metrics are calculated correctly", {
  # Create some test data
  set.seed(123)
  actuals <- 1:100
  predicted <- actuals + rnorm(100, 0, 10)

  # Calculate metrics
  metrics <- tl_calc_regression_metrics(actuals, predicted)

  # Check metric names
  expect_true("rmse" %in% metrics$metric)
  expect_true("mae" %in% metrics$metric)
  expect_true("rsq" %in% metrics$metric)
  expect_true("mape" %in% metrics$metric)

  # Check metric values
  rmse_idx <- which(metrics$metric == "rmse")
  expect_true(metrics$value[rmse_idx] > 0)
  expect_true(metrics$value[rmse_idx] < 15)  # Reasonable for our noise level

  rsq_idx <- which(metrics$metric == "rsq")
  expect_true(metrics$value[rsq_idx] > 0.8)  # Should be high with our data

  # Test with zeros in actuals (should warn and handle)
  actuals_with_zeros <- c(0, actuals[-1])
  expect_warning(
    metrics_with_zeros <- tl_calc_regression_metrics(actuals_with_zeros, predicted)
  )

  # All zeroes case
  all_zeros <- rep(0, 100)
  expect_warning(
    metrics_all_zeros <- tl_calc_regression_metrics(all_zeros, predicted)
  )

  mape_idx <- which(metrics_all_zeros$metric == "mape")
  expect_true(is.na(metrics_all_zeros$value[mape_idx]))
})

context("Core functionality tests")

# Sample data for testing
data(mtcars)

test_that("tl_model can fit a linear regression model", {
  model <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")

  # Check if model is created with the right class
  expect_s3_class(model, "tidysl_model")
  expect_s3_class(model, "tidysl_linear")

  # Check if model specification is correct
  expect_equal(model$spec$method, "linear")
  expect_false(model$spec$is_classification)
  expect_equal(model$spec$response_var, "mpg")

  # Check if underlying model is a linear model
  expect_s3_class(model$fit, "lm")

  # Check if predictions work
  preds <- predict(model, mtcars[1:5, ])
  expect_length(preds$prediction, 5)
  expect_true(is.numeric(preds$prediction))
})

test_that("tl_model can fit a logistic regression model", {
  # Create binary outcome
  mtcars_binary <- mtcars
  mtcars_binary$am <- as.factor(mtcars_binary$am)

  model <- tl_model(mtcars_binary, am ~ hp + wt, method = "logistic")

  # Check if model is created with the right class
  expect_s3_class(model, "tidysl_model")
  expect_s3_class(model, "tidysl_logistic")

  # Check if model specification is correct
  expect_equal(model$spec$method, "logistic")
  expect_true(model$spec$is_classification)
  expect_equal(model$spec$response_var, "am")

  # Check if underlying model is a glm
  expect_s3_class(model$fit, "glm")

  # Check if predictions work
  probs <- predict(model, mtcars_binary[1:5, ], type = "prob")
  expect_equal(ncol(probs), 2)
  expect_true(all(colSums(as.matrix(probs)) > 0))

  classes <- predict(model, mtcars_binary[1:5, ], type = "class")
  expect_s3_class(classes$prediction, "factor")
  expect_length(classes$prediction, 5)
})

test_that("tl_evaluate calculates appropriate metrics", {
  # Create a model
  model <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")

  # Evaluate on training data
  metrics <- tl_evaluate(model, metrics = c("rmse", "mae", "rsq"))

  # Check if metrics are calculated
  expect_s3_class(metrics, "data.frame")
  expect_equal(nrow(metrics), 3)
  expect_equal(metrics$metric, c("rmse", "mae", "rsq"))
  expect_true(all(is.numeric(metrics$value)))

  # For classification
  mtcars_binary <- mtcars
  mtcars_binary$am <- as.factor(mtcars_binary$am)

  model_class <- tl_model(mtcars_binary, am ~ hp + wt, method = "logistic")

  # Evaluate classification metrics
  class_metrics <- tl_evaluate(model_class, metrics = c("accuracy", "precision", "recall"))

  # Check if metrics are calculated
  expect_s3_class(class_metrics, "data.frame")
  expect_equal(nrow(class_metrics), 3)
  expect_equal(class_metrics$metric, c("accuracy", "precision", "recall"))
  expect_true(all(is.numeric(class_metrics$value)))
})

test_that("tl_cv can perform cross-validation", {
  # Skip if rsample not installed
  skip_if_not_installed("rsample")

  # Perform cross-validation
  cv_results <- tl_cv(mtcars, mpg ~ hp + wt, method = "linear", folds = 3)

  # Check structure of results
  expect_type(cv_results, "list")
  expect_true("fold_metrics" %in% names(cv_results))
  expect_true("summary" %in% names(cv_results))

  # Check fold metrics
  expect_s3_class(cv_results$fold_metrics, "data.frame")
  expect_equal(length(unique(cv_results$fold_metrics$fold)), 3)

  # Check summary metrics
  expect_s3_class(cv_results$summary, "data.frame")
  expect_true("mean_value" %in% names(cv_results$summary))
})

test_that("plot.tidysl_model produces appropriate plots", {
  # Skip if ggplot2 not installed
  skip_if_not_installed("ggplot2")

  # Create a model
  model <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")

  # Create a plot
  p <- plot(model, type = "diagnostics")

  # Check if plot is created
  expect_s3_class(p, "ggplot")
})

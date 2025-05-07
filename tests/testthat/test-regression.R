context("Regression functionality tests")

# Sample data for testing
data(mtcars)

test_that("tl_fit_linear correctly fits linear regression", {
  # Test internal function
  fit <- tl_fit_linear(mtcars, mpg ~ hp + wt)

  # Check if fit is a linear model
  expect_s3_class(fit, "lm")

  # Check if coefficients are calculated
  expect_named(coef(fit), c("(Intercept)", "hp", "wt"))

  # Check if predictions work
  preds <- predict(fit, mtcars[1:5, ])
  expect_length(preds, 5)
})

test_that("tl_predict_linear correctly predicts with different types", {
  # Create a model
  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "linear",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = lm(mpg ~ hp + wt, data = mtcars),
    data = mtcars
  )
  class(model) <- c("tidylearn_linear", "tidylearn_model")

  # Test response predictions
  preds <- tl_predict_linear(model, mtcars[1:5, ], type = "response")
  expect_length(preds, 5)
  expect_true(is.numeric(preds))

  # Test confidence interval predictions
  preds_conf <- tl_predict_linear(model, mtcars[1:5, ], type = "confidence")
  expect_s3_class(preds_conf, "data.frame")
  expect_equal(colnames(preds_conf), c("fit", "lwr", "upr"))

  # Test prediction interval predictions
  preds_pred <- tl_predict_linear(model, mtcars[1:5, ], type = "prediction")
  expect_s3_class(preds_pred, "data.frame")
  expect_equal(colnames(preds_pred), c("fit", "lwr", "upr"))
})

test_that("tl_fit_polynomial correctly fits polynomial regression", {
  # Test internal function
  fit <- tl_fit_polynomial(mtcars, mpg ~ hp + wt, degree = 2)

  # Check if fit is a linear model
  expect_s3_class(fit, "lm")

  # Check if formula contains polynomial terms
  expect_true(grepl("poly", deparse(formula(fit))))

  # Check if predictions work
  preds <- predict(fit, mtcars[1:5, ])
  expect_length(preds, 5)
})

test_that("tl_predict_polynomial correctly predicts", {
  # Create a polynomial model
  fit <- tl_fit_polynomial(mtcars, mpg ~ hp + wt, degree = 2)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "polynomial",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_polynomial", "tidylearn_model")

  # Test predictions
  preds <- tl_predict_polynomial(model, mtcars[1:5, ])
  expect_length(preds, 5)
  expect_true(is.numeric(preds))
})

test_that("tl_plot_diagnostics creates diagnostic plots", {
  # Skip if ggplot2 not installed
  skip_if_not_installed("ggplot2")

  # Create a model
  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "linear",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = lm(mpg ~ hp + wt, data = mtcars),
    data = mtcars
  )
  class(model) <- c("tidylearn_linear", "tidylearn_model")

  # Test single plot
  p1 <- tl_plot_diagnostics(model, which = 1)
  expect_s3_class(p1, "ggplot")

  # Test multiple plots
  p_all <- tl_plot_diagnostics(model)
  expect_type(p_all, "list")
  expect_length(p_all, 4)
  expect_s3_class(p_all[[1]], "ggplot")
})

test_that("tl_plot_actual_predicted creates comparison plots", {
  # Skip if ggplot2 not installed
  skip_if_not_installed("ggplot2")

  # Create a model
  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "linear",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = lm(mpg ~ hp + wt, data = mtcars),
    data = mtcars
  )
  class(model) <- c("tidylearn_linear", "tidylearn_model")

  # Test plot creation
  p <- tl_plot_actual_predicted(model)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_residuals creates residual plots", {
  # Skip if ggplot2 not installed
  skip_if_not_installed("ggplot2")

  # Create a model
  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "linear",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = lm(mpg ~ hp + wt, data = mtcars),
    data = mtcars
  )
  class(model) <- c("tidylearn_linear", "tidylearn_model")

  # Test different plot types
  p1 <- tl_plot_residuals(model, type = "fitted")
  expect_s3_class(p1, "ggplot")

  p2 <- tl_plot_residuals(model, type = "histogram")
  expect_s3_class(p2, "ggplot")

  p3 <- tl_plot_residuals(model, type = "predicted")
  expect_s3_class(p3, "ggplot")
})

test_that("tl_plot_intervals creates interval plots", {
  # Skip if ggplot2 not installed
  skip_if_not_installed("ggplot2")

  # Create a model
  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "linear",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = lm(mpg ~ hp + wt, data = mtcars),
    data = mtcars
  )
  class(model) <- c("tidylearn_linear", "tidylearn_model")

  # Test plot creation with different confidence levels
  p1 <- tl_plot_intervals(model, level = 0.95)
  expect_s3_class(p1, "ggplot")

  p2 <- tl_plot_intervals(model, level = 0.90)
  expect_s3_class(p2, "ggplot")
})

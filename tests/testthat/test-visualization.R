# tests/testthat/test-visualization.R

context("Visualization functions")

# Setup: Create simple models for testing
setup_test_models <- function() {
  # Create a simple dataset
  set.seed(123)
  test_data <- data.frame(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100),
    y_reg = rnorm(100),
    y_class = factor(sample(c("A", "B"), 100, replace = TRUE))
  )

  # Create a regression model
  reg_model <- tl_model(data = test_data, formula = y_reg ~ x1 + x2 + x3, method = "linear")

  # Create a classification model
  class_model <- tl_model(data = test_data, formula = y_class ~ x1 + x2 + x3, method = "logistic")

  # Create a tree model (for feature importance)
  tree_model <- tl_model(data = test_data, formula = y_reg ~ x1 + x2 + x3, method = "tree")

  return(list(
    data = test_data,
    reg_model = reg_model,
    class_model = class_model,
    tree_model = tree_model
  ))
}

test_that("tl_plot_importance_comparison returns a ggplot object", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Call the function with multiple models
  p <- tl_plot_importance_comparison(test_env$tree_model, test_env$tree_model,
                                     names = c("Model1", "Model2"))

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_model_comparison returns a ggplot object", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Call the function with multiple models
  p <- tl_plot_model_comparison(test_env$reg_model, test_env$reg_model,
                                names = c("Model1", "Model2"))

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_cv_results returns a ggplot object", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Create mock CV results
  cv_results <- list(
    fold_metrics = data.frame(
      metric = rep(c("rmse", "mae"), each = 5),
      value = runif(10),
      fold = rep(1:5, 2)
    ),
    summary = data.frame(
      metric = c("rmse", "mae"),
      mean_value = c(0.5, 0.3),
      sd_value = c(0.1, 0.05)
    )
  )

  # Call the function
  p <- tl_plot_cv_results(cv_results)

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_lift returns a ggplot object for classification", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Call the function
  p <- tl_plot_lift(test_env$class_model)

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_gain returns a ggplot object for classification", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Call the function
  p <- tl_plot_gain(test_env$class_model)

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_roc returns a ggplot object for classification", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Call the function
  p <- tl_plot_roc(test_env$class_model)

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_confusion returns a ggplot object for classification", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Call the function
  p <- tl_plot_confusion(test_env$class_model)

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_precision_recall returns a ggplot object for classification", {
  skip_if_not_installed("ggplot2")

  test_env <- setup_test_models()

  # Call the function
  p <- tl_plot_precision_recall(test_env$class_model)

  # Check that it returns a ggplot object
  expect_s3_class(p, "ggplot")
})

test_that("tl_dashboard returns a shiny app object", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("shinydashboard")

  test_env <- setup_test_models()

  # Call the function and check it returns a shiny app
  app <- tl_dashboard(test_env$reg_model)

  expect_s3_class(app, "shiny.appobj")
})

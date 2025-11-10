context("Model selection functionality tests")

# Sample data for testing
data(mtcars)

test_that("tl_step_selection performs stepwise selection", {
  # Test with backward selection
  model <- tl_step_selection(mtcars, mpg ~ ., direction = "backward")

  # Check if result is a tidysl model
  expect_s3_class(model, "tidysl_model")
  expect_s3_class(model, "tidysl_linear")

  # Check if formula has been updated from the original
  expect_false(identical(formula(model$fit), mpg ~ .))

  # Check selection details
  expect_equal(model$spec$selection$direction, "backward")
  expect_equal(model$spec$selection$criterion, "AIC")

  # Test with forward selection
  model_forward <- tl_step_selection(mtcars, mpg ~ 1, direction = "forward",
                                     scope = mpg ~ cyl + disp + hp + wt)

  # Check if result is a tidysl model
  expect_s3_class(model_forward, "tidysl_model")

  # Check selection details
  expect_equal(model_forward$spec$selection$direction, "forward")
})

test_that("tl_compare_cv compares models using cross-validation", {
  # Skip if rsample not installed
  skip_if_not_installed("rsample")

  # Create multiple models
  model1 <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")
  model2 <- tl_model(mtcars, mpg ~ hp + wt + qsec, method = "linear")

  # Compare models
  cv_results <- tl_compare_cv(mtcars, list(model1 = model1, model2 = model2), folds = 3)

  # Check structure of results
  expect_type(cv_results, "list")
  expect_true("fold_metrics" %in% names(cv_results))
  expect_true("summary" %in% names(cv_results))

  # Check if both models are included
  model_names <- unique(cv_results$fold_metrics$model)
  expect_equal(sort(model_names), sort(c("model1", "model2")))

  # Check summary metrics
  expect_s3_class(cv_results$summary, "data.frame")
  expect_true(all(c("model", "metric", "mean_value") %in% names(cv_results$summary)))
})

test_that("tl_plot_cv_comparison creates comparison plots", {
  # Skip if required packages not installed
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("rsample")

  # Create multiple models
  model1 <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")
  model2 <- tl_model(mtcars, mpg ~ hp + wt + qsec, method = "linear")

  # Compare models
  cv_results <- tl_compare_cv(mtcars, list(model1 = model1, model2 = model2), folds = 3)

  # Create comparison plot
  p <- tl_plot_cv_comparison(cv_results)

  # Check if plot is created
  expect_s3_class(p, "ggplot")
})

test_that("tl_test_model_difference performs statistical tests", {
  # Skip if required packages not installed
  skip_if_not_installed("rsample")

  # Create multiple models
  model1 <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")
  model2 <- tl_model(mtcars, mpg ~ hp + wt + qsec, method = "linear")

  # Compare models
  cv_results <- tl_compare_cv(mtcars, list(model1 = model1, model2 = model2), folds = 3)

  # Test for differences
  test_results <- tl_test_model_difference(cv_results, baseline_model = "model1")

  # Check structure of results
  expect_s3_class(test_results, "data.frame")
  expect_true(all(c("metric", "model", "baseline", "mean_diff", "p_value") %in% names(test_results)))

  # Check if model2 is included in the comparison
  expect_equal(unique(test_results$model), "model2")

  # Check if baseline is correctly specified
  expect_equal(unique(test_results$baseline), "model1")

  # Check if p-values are numeric
  expect_true(is.numeric(test_results$p_value))
})

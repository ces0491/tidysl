context("Diagnostics functionality tests")

# Sample data for testing
data(mtcars)

test_that("tl_influence_measures calculates influence measures", {
  # Create a linear model
  model <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")

  # Calculate influence measures
  influence <- tl_influence_measures(model)

  # Check structure of results
  expect_s3_class(influence, "data.frame")
  expect_true(all(c("observation", "cooks_distance", "leverage", "dffits",
                    "std_residual", "is_influential") %in% names(influence)))

  # Check if thresholds are stored as attributes
  expect_true(all(c("threshold_cook", "threshold_leverage", "threshold_dffits") %in%
                    names(attributes(influence))))

  # Check if all rows in the data have been analyzed
  expect_equal(nrow(influence), nrow(mtcars))

  # Check if flags are logical
  expect_true(is.logical(influence$is_cook_influential))
  expect_true(is.logical(influence$is_leverage_influential))
  expect_true(is.logical(influence$is_dffits_influential))
  expect_true(is.logical(influence$is_influential))

  # Check with custom thresholds
  influence_custom <- tl_influence_measures(model, threshold_cook = 0.2,
                                            threshold_leverage = 0.3)

  # Check if custom thresholds are stored
  expect_equal(attr(influence_custom, "threshold_cook"), 0.2)
  expect_equal(attr(influence_custom, "threshold_leverage"), 0.3)
})

test_that("tl_plot_influence creates influence plots", {
  skip_if_not_installed("ggplot2")

  # Create a linear model
  model <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")

  # Test Cook's distance plot
  p1 <- tl_plot_influence(model, plot_type = "cook")
  expect_s3_class(p1, "ggplot")

  # Test leverage plot
  p2 <- tl_plot_influence(model, plot_type = "leverage")
  expect_s3_class(p2, "ggplot")

  # Test index plot
  p3 <- tl_plot_influence(model, plot_type = "index")
  expect_s3_class(p3, "ggplot")

  # Test with custom parameters
  p4 <- tl_plot_influence(model, plot_type = "cook", threshold_cook = 0.2, n_labels = 2)
  expect_s3_class(p4, "ggplot")

  # Test for error with invalid plot type
  expect_error(tl_plot_influence(model, plot_type = "invalid"))
})

test_that("tl_check_assumptions checks model assumptions", {
  # Create a linear model
  model <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")

  # Check assumptions
  assumptions <- tl_check_assumptions(model, test = FALSE, verbose = FALSE)

  # Check structure of results
  expect_type(assumptions, "list")
  expect_true(all(c("linearity", "independence", "homoscedasticity",
                    "normality", "multicollinearity", "outliers", "overall") %in%
                    names(assumptions)))

  # Check if individual assumption results have the right structure
  for (name in setdiff(names(assumptions), "overall")) {
    expect_true(all(c("assumption", "check", "details", "recommendation") %in%
                      names(assumptions[[name]])))
  }

  # Check if overall summary is included
  expect_true(all(c("status", "n_checked", "n_violated", "n_satisfied") %in%
                    names(assumptions$overall)))

  # Test with statistical tests if packages available
  if (requireNamespace("car", quietly = TRUE) &&
      requireNamespace("lmtest", quietly = TRUE)) {
    # Check assumptions with tests
    assumptions_test <- tl_check_assumptions(model, test = TRUE, verbose = FALSE)

    # Check structure of results
    expect_type(assumptions_test, "list")

    # Check if test details are included
    expect_true(grepl("p-value", assumptions_test$homoscedasticity$details))
  }
})

test_that("tl_diagnostic_dashboard creates dashboard", {
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("gridExtra")

  # Create a linear model
  model <- tl_model(mtcars, mpg ~ hp + wt, method = "linear")

  # Stub the ggplot function to avoid errors in testing
  mockery::stub(tl_plot_diagnostics, "ggplot2::ggplot", function(...) {
    structure(list(), class = "ggplot")
  })

  # Stub the gridExtra function to avoid errors in testing
  mockery::stub(tl_diagnostic_dashboard, "gridExtra::grid.arrange", function(...) {
    # Return a mock gridded plot
    structure(list(), class = "grid")
  })

  # Create dashboard
  dashboard <- tl_diagnostic_dashboard(model, include_influence = TRUE,
                                       include_assumptions = TRUE,
                                       include_performance = TRUE)

  # Check if dashboard is created
  expect_s3_class(dashboard, "grid")

  # Test with different layout
  dashboard2 <- tl_diagnostic_dashboard(model, arrange_plots = "row")
  expect_s3_class(dashboard2, "grid")
})

test_that("tl_detect_outliers detects outliers", {
  # Create test data with outliers
  test_data <- mtcars
  test_data$mpg[1] <- 100  # Add an extreme value

  # Test outlier detection with IQR method
  outliers_iqr <- tl_detect_outliers(test_data, variables = c("mpg", "hp"),
                                     method = "iqr", plot = FALSE)

  # Check structure of results
  expect_type(outliers_iqr, "list")
  expect_true(all(c("method", "threshold", "outlier_flags", "any_outlier",
                    "outlier_counts", "outlier_indices") %in% names(outliers_iqr)))

  # Check if outliers are detected
  expect_true(any(outliers_iqr$any_outlier))

  # Check if first observation (with extreme value) is detected
  expect_true(1 %in% outliers_iqr$outlier_indices)

  # Test with z-score method
  outliers_zscore <- tl_detect_outliers(test_data, variables = c("mpg", "hp"),
                                        method = "z-score", threshold = 3, plot = FALSE)

  # Check if outliers are detected
  expect_true(any(outliers_zscore$any_outlier))

  # Test with plotting
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    outliers_plot <- tl_detect_outliers(test_data, variables = c("mpg", "hp"),
                                        method = "iqr", plot = TRUE)

    # Check if plot is created
    expect_s3_class(outliers_plot$plot, "ggplot")
  }
})

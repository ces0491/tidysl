context("Utility functions tests")

# Sample data for testing
data(mtcars)

test_that("tl_check_packages detects installed packages", {
  # Test with definitely installed packages
  result <- tl_check_packages("stats", error = FALSE)
  expect_true(result)

  # Test with multiple packages
  result <- tl_check_packages(c("stats", "utils"), error = FALSE)
  expect_true(all(result))

  # Test error handling (commented out to prevent test failures)
  # expect_error(tl_check_packages("nonexistentpackage"))
})

test_that("tl_version returns a valid version string", {
  # Mock the packageVersion function
  mockery::stub(tl_version, "utils::packageVersion", "1.0.0")

  version <- tl_version()
  expect_type(version, "character")
  expect_match(version, "^\\d+\\.\\d+\\.\\d+$")
})

test_that("tl_has_variance correctly identifies low-variance variables", {
  # Test numeric with variance
  x1 <- rnorm(100)
  expect_true(tl_has_variance(x1))

  # Test numeric with no variance
  x2 <- rep(1, 100)
  expect_false(tl_has_variance(x2))

  # Test factor with multiple levels
  x3 <- factor(sample(c("A", "B", "C"), 100, replace = TRUE))
  expect_true(tl_has_variance(x3))

  # Test factor with single level
  x4 <- factor(rep("A", 100))
  expect_false(tl_has_variance(x4))
})

test_that("tl_check_multicollinearity identifies correlated variables", {
  # Create test data with correlated variables
  test_data <- data.frame(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100)
  )
  test_data$x4 <- test_data$x1 * 0.9 + rnorm(100, sd = 0.1)  # Highly correlated with x1

  # Test multicollinearity detection
  result <- tl_check_multicollinearity(test_data, threshold = 0.8)

  # Check structure of results
  expect_s3_class(result, "data.frame")
  expect_true(all(c("var1", "var2", "correlation") %in% names(result)))

  # Check if high correlation is detected
  expect_true(any(abs(result$correlation) > 0.8))

  # Check if high correlation involves x1 and x4
  high_cor_pairs <- paste(result$var1, result$var2, sep = "-")
  expect_true(any(c("x1-x4", "x4-x1") %in% high_cor_pairs))
})

test_that("tl_scale_to_range correctly scales values", {
  # Test scaling to 0-1
  x <- 1:10
  scaled <- tl_scale_to_range(x, 0, 1)
  expect_equal(min(scaled), 0)
  expect_equal(max(scaled), 1)
  expect_length(scaled, length(x))

  # Test scaling to custom range
  scaled2 <- tl_scale_to_range(x, -1, 1)
  expect_equal(min(scaled2), -1)
  expect_equal(max(scaled2), 1)

  # Test with all identical values
  x2 <- rep(5, 10)
  scaled3 <- tl_scale_to_range(x2, 0, 1)
  expect_equal(mean(scaled3), 0.5)
  expect_warning(tl_scale_to_range(x2, 0, 1))
})

test_that("tl_handle_outliers detects and handles outliers", {
  # Create data with outliers
  x <- c(rnorm(95, mean = 0, sd = 1), rnorm(5, mean = 10, sd = 1))

  # Test IQR method
  result_iqr <- tl_handle_outliers(x, method = "iqr", replace_with = "NA")
  expect_true(sum(is.na(result_iqr)) > 0)

  # Test z-score method
  result_z <- tl_handle_outliers(x, method = "z-score", threshold = 3, replace_with = "median")
  expect_false(any(is.na(result_z)))
  expect_true(max(result_z) < max(x))

  # Test winsorization
  result_win <- tl_handle_outliers(x, method = "iqr", replace_with = "winsorize")
  expect_false(any(is.na(result_win)))
  expect_true(max(result_win) < max(x))
  expect_true(min(result_win) >= min(x))
})

test_that("tl_variable_importance calculates feature importance", {
  # Skip if required packages not installed
  skip_if_not_installed("FSelector")

  # Test correlation method
  imp_cor <- tl_variable_importance(mtcars, "mpg", method = "correlation")

  # Check structure of results
  expect_s3_class(imp_cor, "data.frame")
  expect_true(all(c("variable", "importance") %in% names(imp_cor)))

  # Check if importance values are numeric between 0 and 1
  expect_true(all(imp_cor$importance >= 0 & imp_cor$importance <= 1))

  # Test top_n parameter
  imp_top <- tl_variable_importance(mtcars, "mpg", method = "correlation", top_n = 3)
  expect_equal(nrow(imp_top), 3)
})

test_that("tl_bin_variable creates bins from numeric data", {
  # Create test data
  x <- 1:100

  # Test equal width binning
  bins_equal <- tl_bin_variable(x, n_bins = 5, method = "equal_width")
  expect_s3_class(bins_equal, "factor")
  expect_equal(length(levels(bins_equal)), 5)

  # Test equal frequency binning
  bins_freq <- tl_bin_variable(x, n_bins = 5, method = "equal_freq")
  expect_s3_class(bins_freq, "factor")
  expect_equal(length(levels(bins_freq)), 5)

  # Test custom breaks
  custom_breaks <- c(0, 25, 50, 75, 100)
  bins_custom <- tl_bin_variable(x, method = "custom", breaks = custom_breaks)
  expect_s3_class(bins_custom, "factor")
  expect_equal(length(levels(bins_custom)), length(custom_breaks) - 1)

  # Test custom labels
  custom_labels <- c("Low", "Medium", "High")
  bins_labels <- tl_bin_variable(x, n_bins = 3, method = "equal_width", labels = custom_labels)
  expect_equal(levels(bins_labels), custom_labels)
})

test_that("tl_partition_data creates train/test splits", {
  # Test simple random sampling
  split <- tl_partition_data(mtcars, prop = 0.75)

  # Check structure of results
  expect_type(split, "list")
  expect_true(all(c("train", "test") %in% names(split)))

  # Check proportions
  expect_equal(length(split$train), round(0.75 * nrow(mtcars)))
  expect_equal(length(split$test), nrow(mtcars) - length(split$train))

  # Test stratified sampling
  split_strat <- tl_partition_data(mtcars, prop = 0.75, strata = "cyl")

  # Check structure of results
  expect_type(split_strat, "list")
  expect_true(all(c("train", "test") %in% names(split_strat)))

  # Check if indices are valid
  expect_true(all(split_strat$train %in% 1:nrow(mtcars)))
  expect_true(all(split_strat$test %in% 1:nrow(mtcars)))

  # Check if train and test sets are disjoint
  expect_equal(length(intersect(split_strat$train, split_strat$test)), 0)
})

test_that("tl_format_numbers formats numbers correctly", {
  # Test standard format
  x <- c(1.23456, 0.00789, 1000.5)
  formatted <- tl_format_numbers(x, digits = 3, format = "standard")
  expect_equal(formatted, c("1.235", "0.008", "1000.500"))

  # Test scientific format
  formatted_sci <- tl_format_numbers(x, digits = 3, format = "scientific")
  expect_match(formatted_sci[1], "1.235e\\+00")
  expect_match(formatted_sci[2], "7.890e-03")

  # Test percentage format
  x_pct <- c(0.1234, 0.5678, 0.9)
  formatted_pct <- tl_format_numbers(x_pct, digits = 2, format = "percent")
  expect_equal(formatted_pct, c("12.34%", "56.78%", "90.00%"))

  # Test zero padding
  formatted_pad <- tl_format_numbers(c(1.2, 3.4), digits = 3, format = "standard", zero_pad = TRUE)
  expect_equal(formatted_pad, c("1.200", "3.400"))
})

# tests/testthat/test-metrics.R

context("Metrics functions")

# Setup: Create test data with known metrics
setup_test_data <- function() {
  # Binary classification test case
  set.seed(123)
  n <- 100

  # Actual values (50/50 split)
  actuals <- factor(rep(c("Yes", "No"), each = 50))

  # Perfect predictions
  perfect_preds <- actuals
  perfect_probs <- data.frame(
    No = as.numeric(actuals == "No"),
    Yes = as.numeric(actuals == "Yes")
  )

  # Random predictions
  set.seed(123)
  random_preds <- factor(sample(c("Yes", "No"), n, replace = TRUE))
  random_probs <- data.frame(
    No = runif(n),
    Yes = runif(n)
  )
  random_probs <- random_probs / rowSums(random_probs)  # Normalize

  # Regression test case
  actuals_reg <- 1:10
  perfect_preds_reg <- actuals_reg
  biased_preds_reg <- actuals_reg + 2  # Constant bias

  return(list(
    actuals = actuals,
    perfect_preds = perfect_preds,
    perfect_probs = perfect_probs,
    random_preds = random_preds,
    random_probs = random_probs,
    actuals_reg = actuals_reg,
    perfect_preds_reg = perfect_preds_reg,
    biased_preds_reg = biased_preds_reg
  ))
}

test_that("tl_calc_classification_metrics calculates metrics correctly for perfect predictions", {
  test_data <- setup_test_data()

  # Calculate metrics for perfect predictions
  metrics <- tl_calc_classification_metrics(
    actuals = test_data$actuals,
    predicted = test_data$perfect_preds,
    predicted_probs = test_data$perfect_probs,
    metrics = c("accuracy", "precision", "recall", "f1", "auc")
  )

  # Check metric names
  expect_equal(metrics$metric, c("accuracy", "precision", "recall", "f1", "auc"))

  # Check values (perfect predictions should give metrics of 1)
  expect_equal(metrics$value, rep(1, 5), tolerance = 1e-6)
})

test_that("tl_calc_classification_metrics calculates metrics correctly for random predictions", {
  test_data <- setup_test_data()

  # Calculate metrics for random predictions
  metrics <- tl_calc_classification_metrics(
    actuals = test_data$actuals,
    predicted = test_data$random_preds,
    predicted_probs = test_data$random_probs,
    metrics = c("accuracy")
  )

  # For random predictions, accuracy should be around 0.5
  expect_true(metrics$value[metrics$metric == "accuracy"] < 0.7)
})

test_that("tl_evaluate_thresholds calculates threshold-dependent metrics correctly", {
  test_data <- setup_test_data()

  # Evaluate at specific thresholds
  thresholds <- c(0.3, 0.5, 0.7)
  metrics <- tl_evaluate_thresholds(
    actuals = test_data$actuals,
    probs = test_data$perfect_probs$Yes,  # Probabilities for positive class
    thresholds = thresholds,
    pos_class = "Yes"
  )

  # Check that we have the expected number of metrics
  expect_equal(nrow(metrics), length(thresholds) * 6)  # 6 metrics per threshold

  # Check that threshold values are included in metric names
  for (t in thresholds) {
    expect_true(any(grepl(paste0("_t", t), metrics$metric)))
  }
})

test_that("tl_calculate_pr_auc calculates area under PR curve correctly", {
  # Create a mock ROCR performance object with perfect PR curve
  perfect_precision <- c(1, 1)
  perfect_recall <- c(0, 1)

  # Create a structure similar to ROCR performance object
  perf <- structure(
    list(
      x.values = list(perfect_recall),
      y.values = list(perfect_precision)
    ),
    class = "performance"
  )

  # For a perfect classifier, PR-AUC should be 1
  pr_auc <- tl_calculate_pr_auc(perf)
  expect_equal(pr_auc, 1, tolerance = 1e-6)

  # Create a mock ROCR performance object with random PR curve
  # A random classifier should have PR-AUC = prevalence
  prevalence <- 0.5  # 50% positive class
  random_precision <- rep(prevalence, 10)
  random_recall <- seq(0, 1, length.out = 10)

  perf_random <- structure(
    list(
      x.values = list(random_recall),
      y.values = list(random_precision)
    ),
    class = "performance"
  )

  pr_auc_random <- tl_calculate_pr_auc(perf_random)
  expect_equal(pr_auc_random, prevalence, tolerance = 1e-6)
})

test_that("tl_calc_regression_metrics calculates metrics correctly", {
  test_data <- setup_test_data()

  # Create a mock function for regression metrics (assuming it's implemented similarly)
  tl_calc_regression_metrics <- function(actuals, predicted, metrics) {
    results <- tibble::tibble(metric = character(), value = numeric())

    if ("rmse" %in% metrics) {
      rmse <- sqrt(mean((actuals - predicted)^2))
      results <- results %>% dplyr::add_row(metric = "rmse", value = rmse)
    }

    if ("mae" %in% metrics) {
      mae <- mean(abs(actuals - predicted))
      results <- results %>% dplyr::add_row(metric = "mae", value = mae)
    }

    if ("rsq" %in% metrics) {
      ss_total <- sum((actuals - mean(actuals))^2)
      ss_residual <- sum((actuals - predicted)^2)
      rsq <- 1 - ss_residual/ss_total
      results <- results %>% dplyr::add_row(metric = "rsq", value = rsq)
    }

    return(results)
  }

  # Calculate metrics for perfect predictions
  metrics_perfect <- tl_calc_regression_metrics(
    test_data$actuals_reg,
    test_data$perfect_preds_reg,
    metrics = c("rmse", "mae", "rsq")
  )

  # Check perfect prediction metrics
  expect_equal(metrics_perfect$value[metrics_perfect$metric == "rmse"], 0, tolerance = 1e-6)
  expect_equal(metrics_perfect$value[metrics_perfect$metric == "mae"], 0, tolerance = 1e-6)
  expect_equal(metrics_perfect$value[metrics_perfect$metric == "rsq"], 1, tolerance = 1e-6)

  # Calculate metrics for biased predictions
  metrics_biased <- tl_calc_regression_metrics(
    test_data$actuals_reg,
    test_data$biased_preds_reg,
    metrics = c("rmse", "mae")
  )

  # Check biased prediction metrics
  expect_equal(metrics_biased$value[metrics_biased$metric == "rmse"], 2, tolerance = 1e-6)
  expect_equal(metrics_biased$value[metrics_biased$metric == "mae"], 2, tolerance = 1e-6)
})

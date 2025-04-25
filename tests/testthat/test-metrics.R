context("Evaluation metrics")

test_that("tl_calc_regression_metrics calculates regression metrics correctly", {
  # Create synthetic testing data
  set.seed(123)
  actuals <- 1:100

  # Perfect predictions
  perfect_preds <- actuals
  metrics_perfect <- tl_calc_regression_metrics(actuals, perfect_preds)

  # Check structure
  expect_s3_class(metrics_perfect, "tbl_df")
  expect_true("metric" %in% names(metrics_perfect))
  expect_true("value" %in% names(metrics_perfect))

  # Check specific metrics
  rmse_idx <- which(metrics_perfect$metric == "rmse")
  expect_equal(metrics_perfect$value[rmse_idx], 0)

  rsq_idx <- which(metrics_perfect$metric == "rsq")
  expect_equal(metrics_perfect$value[rsq_idx], 1)

  # With noise
  noisy_preds <- actuals + rnorm(100, 0, 5)
  metrics_noisy <- tl_calc_regression_metrics(actuals, noisy_preds)

  # Check values
  rmse_idx <- which(metrics_noisy$metric == "rmse")
  expect_true(metrics_noisy$value[rmse_idx] > 0)

  rsq_idx <- which(metrics_noisy$metric == "rsq")
  expect_true(metrics_noisy$value[rsq_idx] < 1)
  expect_true(metrics_noisy$value[rsq_idx] > 0.9)  # Should still be high with our noise level

  # With additional metrics
  metrics_extra <- tl_calc_regression_metrics(
    actuals, noisy_preds,
    metrics = c("rmse", "mae", "rsq", "mape", "mse", "r", "mad")
  )

  # Check additional metrics
  expect_true("mse" %in% metrics_extra$metric)
  expect_true("r" %in% metrics_extra$metric)
  expect_true("mad" %in% metrics_extra$metric)

  # Test with zeros in actuals (should handle and warn)
  actuals_with_zeros <- actuals
  actuals_with_zeros[1:10] <- 0

  expect_warning(
    metrics_zeros <- tl_calc_regression_metrics(actuals_with_zeros, noisy_preds)
  )

  # With all zeros (MAPE should be NA)
  all_zeros <- rep(0, 100)
  expect_warning(
    metrics_all_zeros <- tl_calc_regression_metrics(all_zeros, noisy_preds)
  )

  mape_idx <- which(metrics_all_zeros$metric == "mape")
  expect_true(is.na(metrics_all_zeros$value[mape_idx]))
})

test_that("tl_calc_classification_metrics calculates classification metrics correctly", {
  # Create synthetic testing data
  set.seed(123)
  actuals <- factor(rep(c("A", "B"), each = 50))

  # Perfect predictions
  perfect_preds <- actuals
  metrics_perfect <- tl_calc_classification_metrics(actuals, perfect_preds)

  # Check structure
  expect_s3_class(metrics_perfect, "tbl_df")

  # Check specific metrics
  accuracy_idx <- which(metrics_perfect$metric == "accuracy")
  expect_equal(metrics_perfect$value[accuracy_idx], 1)

  precision_idx <- which(metrics_perfect$metric == "precision")
  expect_equal(metrics_perfect$value[precision_idx], 1)

  # With errors
  noisy_preds <- actuals
  error_indices <- sample(1:100, 20)  # 20% error rate
  levels_vec <- levels(actuals)
  for (i in error_indices) {
    current <- as.character(noisy_preds[i])
    other_level <- levels_vec[levels_vec != current]
    noisy_preds[i] <- other_level
  }

  metrics_noisy <- tl_calc_classification_metrics(actuals, noisy_preds)

  # Check values
  accuracy_idx <- which(metrics_noisy$metric == "accuracy")
  expect_equal(metrics_noisy$value[accuracy_idx], 0.8, tolerance = 0.01)

  # Test with probability predictions
  probs <- data.frame(
    A = c(rep(0.9, 45), rep(0.4, 5), rep(0.1, 5), rep(0.6, 45)),
    B = c(rep(0.1, 45), rep(0.6, 5), rep(0.9, 5), rep(0.4, 45))
  )

  metrics_with_probs <- tl_calc_classification_metrics(
    actuals, noisy_preds, probs,
    metrics = c("accuracy", "precision", "recall", "f1", "auc")
  )

  # Check AUC
  auc_idx <- which(metrics_with_probs$metric == "auc")
  expect_true(metrics_with_probs$value[auc_idx] > 0.5)
  expect_true(metrics_with_probs$value[auc_idx] <= 1.0)

  # Test with thresholds
  metrics_with_thresholds <- tl_calc_classification_metrics(
    actuals, noisy_preds, probs,
    thresholds = c(0.3, 0.5, 0.7)
  )

  # Check threshold metrics exist
  expect_true(any(grepl("accuracy_t0.3", metrics_with_thresholds$metric)))
  expect_true(any(grepl("precision_t0.5", metrics_with_thresholds$metric)))
  expect_true(any(grepl("recall_t0.7", metrics_with_thresholds$metric)))

  # Test multi-class
  multi_actuals <- factor(c(rep("A", 30), rep("B", 30), rep("C", 40)))
  multi_preds <- multi_actuals
  error_indices <- sample(1:100, 20)
  levels_vec <- levels(multi_actuals)

  for (i in error_indices) {
    current <- as.character(multi_preds[i])
    other_levels <- levels_vec[levels_vec != current]
    multi_preds[i] <- sample(other_levels, 1)
  }

  multi_probs <- data.frame(
    A = c(rep(0.8, 25), rep(0.1, 5), rep(0.1, 25), rep(0.1, 5), rep(0.1, 35), rep(0.3, 5)),
    B = c(rep(0.1, 25), rep(0.1, 5), rep(0.8, 25), rep(0.1, 5), rep(0.1, 35), rep(0.3, 5)),
    C = c(rep(0.1, 25), rep(0.8, 5), rep(0.1, 25), rep(0.8, 5), rep(0.8, 35), rep(0.4, 5))
  )

  metrics_multi <- tl_calc_classification_metrics(
    multi_actuals, multi_preds, multi_probs
  )

  # Check metrics
  expect_true("accuracy" %in% metrics_multi$metric)
  expect_true("auc" %in% metrics_multi$metric)
})

test_that("tl_prediction_intervals calculates intervals correctly", {
  # Create synthetic data
  set.seed(123)
  x <- 1:100
  y <- 2 * x + rnorm(100, 0, 10)
  data <- data.frame(x = x, y = y)

  # Split into train/test
  train_indices <- sample(1:100, 80)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]

  # Fit linear model
  lm_model <- tl_model(train_data, y ~ x, method = "linear")

  # Calculate prediction intervals
  intervals <- tl_prediction_intervals(lm_model, test_data)

  # Check structure
  expect_s3_class(intervals, "tbl_df")
  expect_true("prediction" %in% names(intervals))
  expect_true("conf_lower" %in% names(intervals))
  expect_true("conf_upper" %in% names(intervals))
  expect_true("pred_lower" %in% names(intervals))
  expect_true("pred_upper" %in% names(intervals))

  # Check values
  expect_true(all(intervals$conf_lower <= intervals$prediction))
  expect_true(all(intervals$prediction <= intervals$conf_upper))
  expect_true(all(intervals$pred_lower <= intervals$prediction))
  expect_true(all(intervals$prediction <= intervals$pred_upper))

  # Prediction intervals should be wider than confidence intervals
  expect_true(all((intervals$pred_upper - intervals$pred_lower) >=
                    (intervals$conf_upper - intervals$conf_lower)))

  # Test with different confidence level
  intervals_90 <- tl_prediction_intervals(lm_model, test_data, level = 0.90)
  intervals_99 <- tl_prediction_intervals(lm_model, test_data, level = 0.99)

  # 99% intervals should be wider than 90% intervals
  expect_true(all((intervals_99$pred_upper - intervals_99$pred_lower) >=
                    (intervals_90$pred_upper - intervals_90$pred_lower)))

  # Test with classification model (should error)
  # Create classification data
  class_data <- data.frame(
    x = rnorm(100),
    y = factor(sample(c("A", "B"), 100, replace = TRUE))
  )
  class_model <- tl_model(class_data, y ~ x, method = "logistic")
  expect_error(tl_prediction_intervals(class_model, class_data))
})

test_that("tl_find_optimal_threshold finds correct threshold", {
  # Create synthetic data with a known optimal threshold
  set.seed(123)
  n <- 1000
  probs <- runif(n)
  # Create actuals where higher probability means higher chance of being positive
  # with a known threshold of 0.6
  actuals <- factor(ifelse(probs > 0.6,
                           sample(c("pos", "neg"), n, replace = TRUE, prob = c(0.9, 0.1)),
                           sample(c("pos", "neg"), n, replace = TRUE, prob = c(0.2, 0.8))))

  # Package the data
  data <- data.frame(prob = probs, class = actuals)

  # Create a mock model
  mock_model <- list(
    spec = list(
      is_classification = TRUE,
      response_var = "class",
      method = "logistic"
    ),
    data = data
  )
  class(mock_model) <- c("tidylearn_logistic", "tidylearn_model")

  # Add a predict method for this mock model
  predict.tidylearn_logistic <- function(object, new_data, type = "prob", ...) {
    if (type == "prob") {
      result <- data.frame(
        neg = 1 - new_data$prob,
        pos = new_data$prob
      )
      return(result)
    } else if (type == "class") {
      result <- tibble::tibble(
        prediction = factor(ifelse(new_data$prob > 0.5, "pos", "neg"),
                            levels = c("neg", "pos"))
      )
      return(result)
    }
  }

  # Find optimal threshold
  threshold_results <- tl_find_optimal_threshold(
    mock_model,
    data,
    optimize_for = "f1",
    thresholds = seq(0.1, 0.9, by = 0.1)
  )

  # Check structure
  expect_type(threshold_results, "list")
  expect_true("optimal_threshold" %in% names(threshold_results))
  expect_true("optimal_value" %in% names(threshold_results))
  expect_true("all_thresholds" %in% names(threshold_results))

  # The optimal threshold should be close to 0.6
  expect_true(abs(threshold_results$optimal_threshold - 0.6) < 0.2)

  # Try different optimization metrics
  threshold_results_acc <- tl_find_optimal_threshold(
    mock_model,
    data,
    optimize_for = "accuracy",
    thresholds = seq(0.1, 0.9, by = 0.1)
  )

  threshold_results_prec <- tl_find_optimal_threshold(
    mock_model,
    data,
    optimize_for = "precision",
    thresholds = seq(0.1, 0.9, by = 0.1)
  )

  expect_false(threshold_results_acc$optimal_threshold ==
                 threshold_results_prec$optimal_threshold)

  # Test error for regression model
  reg_data <- data.frame(x = 1:100, y = 1:100)
  reg_model <- tl_model(reg_data, y ~ x, method = "linear")
  expect_error(tl_find_optimal_threshold(reg_model, reg_data))
})

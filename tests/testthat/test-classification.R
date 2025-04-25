context("Classification models")

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

test_that("logistic regression model works", {
  # Train model (binary classification)
  log_model <- tl_model(train_binary, Species ~ ., method = "logistic")

  # Check model object
  expect_s3_class(log_model$fit, "glm")

  # Check predictions
  class_preds <- predict(log_model, test_binary, type = "class")
  expect_s3_class(class_preds, "tbl_df")
  expect_equal(nrow(class_preds), nrow(test_binary))
  expect_true(is.factor(class_preds$prediction))

  prob_preds <- predict(log_model, test_binary, type = "prob")
  expect_s3_class(prob_preds, "tbl_df")
  expect_equal(nrow(prob_preds), nrow(test_binary))
  expect_equal(ncol(prob_preds), length(levels(binary_iris$Species)))

  # Check evaluation
  metrics <- tl_evaluate(log_model, test_binary)
  expect_true("accuracy" %in% metrics$metric)
  expect_true("precision" %in% metrics$metric)
  expect_true("recall" %in% metrics$metric)
  expect_true("f1" %in% metrics$metric)
  expect_true("auc" %in% metrics$metric)

  # Check plotting functions
  p1 <- tl_plot_roc(log_model, test_binary)
  expect_s3_class(p1, "ggplot")

  p2 <- tl_plot_confusion(log_model, test_binary)
  expect_s3_class(p2, "ggplot")

  p3 <- tl_plot_precision_recall(log_model, test_binary)
  expect_s3_class(p3, "ggplot")

  p4 <- tl_plot_calibration(log_model, test_binary)
  expect_s3_class(p4, "ggplot")
})

test_that("classification tree model works", {
  skip_if_not_installed("rpart")

  # Train model
  tree_model <- tl_model(
    train_iris,
    Species ~ .,
    method = "tree",
    is_classification = TRUE
  )

  # Check model object
  expect_s3_class(tree_model$fit, "rpart")

  # Check predictions
  class_preds <- predict(tree_model, test_iris, type = "class")
  expect_s3_class(class_preds, "tbl_df")
  expect_true(is.factor(class_preds$prediction))

  prob_preds <- predict(tree_model, test_iris, type = "prob")
  expect_s3_class(prob_preds, "tbl_df")
  expect_equal(ncol(prob_preds), length(levels(iris$Species)))

  # Check feature importance
  if (requireNamespace("rpart.plot", quietly = TRUE)) {
    expect_error(tl_plot_tree(tree_model), NA)
  }

  p <- tl_plot_importance(tree_model)
  expect_s3_class(p, "ggplot")
})

test_that("random forest classification model works", {
  skip_if_not_installed("randomForest")

  # Train model
  forest_model <- tl_model(
    train_iris,
    Species ~ .,
    method = "forest",
    is_classification = TRUE,
    ntree = 50  # Use fewer trees for test speed
  )

  # Check model object
  expect_s3_class(forest_model$fit, "randomForest")

  # Check predictions
  class_preds <- predict(forest_model, test_iris, type = "class")
  expect_s3_class(class_preds, "tbl_df")

  prob_preds <- predict(forest_model, test_iris, type = "prob")
  expect_s3_class(prob_preds, "tbl_df")

  # Check feature importance
  p <- tl_plot_importance(forest_model)
  expect_s3_class(p, "ggplot")

  # Check partial dependence
  p <- tl_plot_partial_dependence(forest_model, "Petal.Length")
  expect_s3_class(p, "ggplot")
})

test_that("boosting classification model works", {
  skip_if_not_installed("gbm")

  # Train model
  boost_model <- tl_model(
    train_binary,
    Species ~ .,
    method = "boost",
    is_classification = TRUE,
    n.trees = 50  # Use fewer trees for test speed
  )

  # Check model object
  expect_s3_class(boost_model$fit, "gbm")

  # Check predictions
  class_preds <- predict(boost_model, test_binary, type = "class")
  expect_s3_class(class_preds, "tbl_df")

  prob_preds <- predict(boost_model, test_binary, type = "prob")
  expect_s3_class(prob_preds, "tbl_df")

  # Check plotting functions
  p <- tl_plot_importance(boost_model)
  expect_s3_class(p, "ggplot")
})

test_that("SVM classification model works", {
  skip_if_not_installed("e1071")

  # Train model
  svm_model <- tl_model(
    train_iris,
    Species ~ .,
    method = "svm",
    is_classification = TRUE,
    kernel = "radial",
    probability = TRUE
  )

  # Check model object
  expect_s3_class(svm_model$fit, "svm")

  # Check predictions
  class_preds <- predict(svm_model, test_iris, type = "class")
  expect_s3_class(class_preds, "tbl_df")

  prob_preds <- predict(svm_model, test_iris, type = "prob")
  expect_s3_class(prob_preds, "tbl_df")

  # Check SVM boundary plot (if e1071 is available)
  expect_error(
    tl_plot_svm_boundary(svm_model, "Petal.Length", "Petal.Width"),
    NA
  )
})

test_that("optimal threshold finding works", {
  # Train logistic model
  log_model <- tl_model(train_binary, Species ~ ., method = "logistic")

  # Find optimal threshold
  threshold_results <- tl_find_optimal_threshold(
    log_model,
    test_binary,
    optimize_for = "f1"
  )

  # Check structure
  expect_type(threshold_results, "list")
  expect_true("optimal_threshold" %in% names(threshold_results))
  expect_true("optimal_value" %in% names(threshold_results))
  expect_true("all_thresholds" %in% names(threshold_results))

  # Check values
  expect_true(threshold_results$optimal_threshold >= 0)
  expect_true(threshold_results$optimal_threshold <= 1)
  expect_true(threshold_results$optimal_value >= 0)
  expect_true(threshold_results$optimal_value <= 1)

  # Check threshold metrics
  expect_s3_class(threshold_results$all_thresholds, "tbl_df")
  expect_true("threshold" %in% names(threshold_results$all_thresholds))
  expect_true("value" %in% names(threshold_results$all_thresholds))
})

test_that("classification metrics are calculated correctly", {
  # Create test predictions
  set.seed(123)
  actuals <- factor(sample(c("A", "B"), 100, replace = TRUE))
  predicted <- factor(sample(c("A", "B"), 100, replace = TRUE))
  predicted_probs <- data.frame(
    A = runif(100, 0, 1),
    B = runif(100, 0, 1)
  )
  predicted_probs <- predicted_probs / rowSums(predicted_probs)

  # Calculate metrics
  metrics <- tl_calc_classification_metrics(actuals, predicted, predicted_probs)

  # Check metric names
  expect_true("accuracy" %in% metrics$metric)
  expect_true("precision" %in% metrics$metric)
  expect_true("recall" %in% metrics$metric)
  expect_true("f1" %in% metrics$metric)
  expect_true("auc" %in% metrics$metric)

  # Test threshold evaluation
  thresholds <- c(0.3, 0.5, 0.7)
  metrics_with_thresholds <- tl_calc_classification_metrics(
    actuals, predicted, predicted_probs, thresholds = thresholds
  )

  # Check threshold metrics exist
  for (t in thresholds) {
    expect_true(paste0("accuracy_t", t) %in% metrics_with_thresholds$metric)
    expect_true(paste0("precision_t", t) %in% metrics_with_thresholds$metric)
    expect_true(paste0("recall_t", t) %in% metrics_with_thresholds$metric)
    expect_true(paste0("f1_t", t) %in% metrics_with_thresholds$metric)
  }

  # Test PR-AUC calculation
  pr_auc_idx <- which(metrics$metric == "pr_auc")
  if (length(pr_auc_idx) > 0) {
    expect_true(metrics$value[pr_auc_idx] >= 0)
    expect_true(metrics$value[pr_auc_idx] <= 1)
  }

  # Test evaluate_thresholds function directly
  threshold_results <- tl_evaluate_thresholds(
    actuals = actuals,
    probs = predicted_probs$B,
    thresholds = c(0.4, 0.6),
    pos_class = "B"
  )

  expect_s3_class(threshold_results, "tbl_df")
  expect_equal(nrow(threshold_results), 12)  # 2 thresholds × 6 metrics
  expect_true("threshold" %in% names(threshold_results))
  expect_true("metric" %in% names(threshold_results))
  expect_true("value" %in% names(threshold_results))
})

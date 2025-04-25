context("Tree-based regression models")

# Load test data
library(MASS)
data(Boston)
set.seed(123)
train_indices <- sample(1:nrow(Boston), 0.8 * nrow(Boston))
train_data <- Boston[train_indices, ]
test_data <- Boston[-train_indices, ]

test_that("regression tree model works", {
  skip_if_not_installed("rpart")

  # Train model
  tree_model <- tl_model(
    train_data,
    medv ~ .,
    method = "tree",
    is_classification = FALSE,
    cp = 0.01
  )

  # Check model object
  expect_s3_class(tree_model$fit, "rpart")

  # Check predictions
  preds <- predict(tree_model, test_data)
  expect_s3_class(preds, "tbl_df")
  expect_equal(nrow(preds), nrow(test_data))
  expect_equal(names(preds), "prediction")

  # Check evaluation
  metrics <- tl_evaluate(tree_model, test_data)
  expect_true("rmse" %in% metrics$metric)
  expect_true("mae" %in% metrics$metric)

  # Check feature importance
  p <- tl_plot_importance(tree_model)
  expect_s3_class(p, "ggplot")
})

test_that("random forest regression model works", {
  skip_if_not_installed("randomForest")

  # Train model
  forest_model <- tl_model(
    train_data,
    medv ~ .,
    method = "forest",
    is_classification = FALSE,
    ntree = 50  # Use fewer trees for test speed
  )

  # Check model object
  expect_s3_class(forest_model$fit, "randomForest")

  # Check predictions
  preds <- predict(forest_model, test_data)
  expect_s3_class(preds, "tbl_df")

  # Check evaluation
  metrics <- tl_evaluate(forest_model, test_data)
  expect_true("rmse" %in% metrics$metric)

  # Check feature importance
  p <- tl_plot_importance(forest_model)
  expect_s3_class(p, "ggplot")

  # Check partial dependence
  p <- tl_plot_partial_dependence(forest_model, "lstat")
  expect_s3_class(p, "ggplot")
})

test_that("gradient boosting regression model works", {
  skip_if_not_installed("gbm")

  # Train model
  boost_model <- tl_model(
    train_data,
    medv ~ .,
    method = "boost",
    is_classification = FALSE,
    n.trees = 50,  # Use fewer trees for test speed
    interaction.depth = 3,
    shrinkage = 0.1
  )

  # Check model object
  expect_s3_class(boost_model$fit, "gbm")

  # Check predictions
  preds <- predict(boost_model, test_data)
  expect_s3_class(preds, "tbl_df")

  # Check evaluation
  metrics <- tl_evaluate(boost_model, test_data)
  expect_true("rmse" %in% metrics$metric)

  # Check feature importance
  p <- tl_plot_importance(boost_model)
  expect_s3_class(p, "ggplot")
})

test_that("importance extraction works for tree models", {
  skip_if_not_installed("rpart")
  skip_if_not_installed("randomForest")
  skip_if_not_installed("gbm")

  # Train models
  tree_model <- tl_model(train_data, medv ~ ., method = "tree", is_classification = FALSE)
  forest_model <- tl_model(train_data, medv ~ ., method = "forest", is_classification = FALSE, ntree = 50)
  boost_model <- tl_model(train_data, medv ~ ., method = "boost", is_classification = FALSE, n.trees = 50)

  # Extract importance
  tree_imp <- tl_extract_importance(tree_model)
  forest_imp <- tl_extract_importance(forest_model)
  boost_imp <- tl_extract_importance(boost_model)

  # Check structure
  expect_s3_class(tree_imp, "tbl_df")
  expect_s3_class(forest_imp, "tbl_df")
  expect_s3_class(boost_imp, "tbl_df")

  expect_true("feature" %in% names(tree_imp))
  expect_true("importance" %in% names(tree_imp))

  # Check normalization to 0-100 scale
  expect_true(max(tree_imp$importance) <= 100)
  expect_true(max(forest_imp$importance) <= 100)
  expect_true(max(boost_imp$importance) <= 100)

  # Check importance comparison
  p <- tl_plot_importance_comparison(forest_model, boost_model, names = c("Random Forest", "Boosting"))
  expect_s3_class(p, "ggplot")
})

context("Utility functions")

test_that("tl_check_packages works", {
  # Test with installed packages
  expect_true(tl_check_packages("stats", error = FALSE))
  expect_true(tl_check_packages(c("stats", "graphics"), error = FALSE))

  # Test with non-existent package (should not error with error = FALSE)
  expect_false(tl_check_packages("non_existent_package_12345", error = FALSE))
  expect_false(tl_check_packages(c("stats", "non_existent_package_12345"), error = FALSE))

  # Test with non-existent package (should error with error = TRUE)
  expect_error(tl_check_packages("non_existent_package_12345", error = TRUE))
  expect_error(tl_check_packages(c("stats", "non_existent_package_12345"), error = TRUE))
})

test_that("tl_calculate_pr_auc works", {
  # Create mock ROCR performance object
  mock_perf <- list(
    x.values = list(c(0, 0.25, 0.5, 0.75, 1)),
    y.values = list(c(1, 0.8, 0.6, 0.4, 0.2))
  )
  class(mock_perf) <- "performance"

  # Calculate AUC
  auc <- tl_calculate_pr_auc(mock_perf)

  # Check result
  expect_type(auc, "double")
  expect_true(auc >= 0 && auc <= 1)

  # Test with NA values
  mock_perf_na <- list(
    x.values = list(c(0, 0.25, NA, 0.75, 1)),
    y.values = list(c(1, 0.8, 0.6, NA, 0.2))
  )
  class(mock_perf_na) <- "performance"

  # Calculate AUC (should handle NAs)
  auc_na <- tl_calculate_pr_auc(mock_perf_na)

  # Check result
  expect_type(auc_na, "double")
  expect_true(auc_na >= 0 && auc_na <= 1)
})

test_that("tl_extract_importance works for tree models", {
  # Skip if tree packages are not available
  skip_if_not_installed("rpart")
  skip_if_not_installed("randomForest")
  skip_if_not_installed("gbm")

  # Load test data
  data(mtcars)

  # Train tree models
  tree_model <- list(
    spec = list(method = "tree"),
    fit = rpart::rpart(mpg ~ ., data = mtcars)
  )
  class(tree_model) <- c("tidylearn_tree", "tidylearn_model")

  # Extract importance
  tree_imp <- tl_extract_importance(tree_model)

  # Check structure
  expect_s3_class(tree_imp, "tbl_df")
  expect_true("feature" %in% names(tree_imp))
  expect_true("importance" %in% names(tree_imp))

  # Check normalization
  expect_true(max(tree_imp$importance) <= 100)
  expect_true(min(tree_imp$importance) >= 0)

  # Test with unsupported model type
  unsupported_model <- list(spec = list(method = "unsupported"))
  class(unsupported_model) <- c("tidylearn_unsupported", "tidylearn_model")
  expect_error(tl_extract_importance(unsupported_model))
})

test_that("tl_extract_importance_regularized works", {
  # Skip if glmnet is not available
  skip_if_not_installed("glmnet")

  # Load test data
  data(mtcars)

  # Create a simple model matrix
  X <- stats::model.matrix(mpg ~ ., data = mtcars)[, -1]
  y <- mtcars$mpg

  # Fit a glmnet model
  if (requireNamespace("glmnet", quietly = TRUE)) {
    glmnet_fit <- glmnet::glmnet(X, y, alpha = 1)

    # Create mock tidylearn model object
    lasso_model <- list(
      spec = list(method = "lasso"),
      fit = glmnet_fit,
      data = mtcars
    )
    attr(lasso_model$fit, "lambda_min") <- glmnet_fit$lambda[5]
    attr(lasso_model$fit, "lambda_1se") <- glmnet_fit$lambda[10]
    class(lasso_model) <- c("tidylearn_lasso", "tidylearn_model")

    # Extract importance
    # Default lambda = "1se"
    lasso_imp <- tl_extract_importance_regularized(lasso_model)

    # Check structure
    expect_s3_class(lasso_imp, "tbl_df")
    expect_true("feature" %in% names(lasso_imp))
    expect_true("importance" %in% names(lasso_imp))

    # Check with lambda = "min"
    lasso_imp_min <- tl_extract_importance_regularized(lasso_model, lambda = "min")
    expect_s3_class(lasso_imp_min, "tbl_df")

    # Check with numeric lambda
    lasso_imp_num <- tl_extract_importance_regularized(lasso_model, lambda = glmnet_fit$lambda[3])
    expect_s3_class(lasso_imp_num, "tbl_df")

    # Check with invalid lambda specification
    expect_error(tl_extract_importance_regularized(lasso_model, lambda = "invalid"))
  }
})

test_that("tidylearn exports tidyverse functions correctly", {
  # Test that the package exports these functions
  expect_true(exists("tibble", mode = "function"))
  expect_true(exists("filter", mode = "function"))
  expect_true(exists("select", mode = "function"))
  expect_true(exists("mutate", mode = "function"))
  expect_true(exists("group_by", mode = "function"))
  expect_true(exists("summarize", mode = "function"))
})

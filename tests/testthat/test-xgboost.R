context("XGBoost functionality tests")

# Sample data for testing
data(mtcars)
data(iris)

test_that("tl_fit_xgboost correctly fits XGBoost models", {
  skip_if_not_installed("xgboost")

  # Test regression XGBoost
  fit_reg <- tl_fit_xgboost(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                            nrounds = 10)  # Use small nrounds for tests

  # Check if fit is an xgboost model
  expect_s3_class(fit_reg, "xgb.Booster")

  # Check if predictions work
  pred_data <- data.frame(hp = mtcars$hp[1:5], wt = mtcars$wt[1:5])
  preds <- predict(fit_reg, xgboost::xgb.DMatrix(as.matrix(pred_data)))
  expect_length(preds, 5)

  # Test binary classification XGBoost
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_binary <- tl_fit_xgboost(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                               is_classification = TRUE, nrounds = 10)

  # Check if fit is an xgboost model
  expect_s3_class(fit_binary, "xgb.Booster")

  # Check if feature names are stored
  feature_names <- attr(fit_binary, "feature_names")
  expect_true(!is.null(feature_names))
  expect_true(all(c("Sepal.Length", "Sepal.Width") %in% feature_names))

  # Test multiclass classification XGBoost
  fit_multi <- tl_fit_xgboost(iris, Species ~ Sepal.Length + Sepal.Width,
                              is_classification = TRUE, nrounds = 10)

  # Check if fit is an xgboost model
  expect_s3_class(fit_multi, "xgb.Booster")
})

test_that("tl_predict_xgboost correctly predicts with different types", {
  skip_if_not_installed("xgboost")

  # Create a regression XGBoost model
  fit_reg <- tl_fit_xgboost(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                            nrounds = 10)

  # Store feature names attribute
  attr(fit_reg, "feature_names") <- c("hp", "wt")

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "xgboost",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_xgboost", "tidylearn_model")

  # Mock model.matrix and predict functions to avoid errors in testing
  mockery::stub(tl_predict_xgboost, "stats::model.matrix", function(...) {
    as.matrix(mtcars[1:5, c("hp", "wt")])
  })

  mockery::stub(tl_predict_xgboost, "xgboost::predict", function(...) {
    runif(5, 10, 30)  # Mocked predictions
  })

  # Test response predictions for regression
  preds_reg <- tl_predict_xgboost(model_reg, mtcars[1:5, ])
  expect_length(preds_reg, 5)
  expect_true(is.numeric(preds_reg))

  # Create a binary classification XGBoost model
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_binary <- tl_fit_xgboost(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                               is_classification = TRUE, nrounds = 10)

  # Store feature names and response levels attributes
  attr(fit_binary, "feature_names") <- c("Sepal.Length", "Sepal.Width")
  attr(fit_binary, "response_levels") <- levels(iris_sub$Species)

  model_binary <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "xgboost",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_binary,
    data = iris_sub
  )
  class(model_binary) <- c("tidylearn_xgboost", "tidylearn_model")

  # Mock model.matrix and predict functions for binary classification
  mockery::stub(tl_predict_xgboost, "xgboost::predict", function(...) {
    # Return probabilities if probability parameter is TRUE
    args <- list(...)
    runif(5, 0, 1)  # Mocked probabilities for binary classification
  })

  # Test probability predictions for binary classification
  probs <- tl_predict_xgboost(model_binary, iris_sub[1:5, ], type = "prob")
  expect_true(is.data.frame(probs) || is.matrix(probs))
  expect_equal(ncol(probs), 2)  # Binary classification has 2 classes

  # Test class predictions for binary classification
  classes <- tl_predict_xgboost(model_binary, iris_sub[1:5, ], type = "class")
  expect_true(is.factor(classes) || is.character(classes))
  expect_length(classes, 5)
})

test_that("tl_plot_xgboost_importance creates importance plots", {
  skip_if_not_installed("xgboost")
  skip_if_not_installed("ggplot2")

  # Create an XGBoost model
  fit <- tl_fit_xgboost(mtcars, mpg ~ hp + wt + cyl + disp,
                        is_classification = FALSE, nrounds = 10)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "xgboost",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_xgboost", "tidylearn_model")

  # Mock the xgb.importance function to avoid errors in testing
  mockery::stub(tl_plot_xgboost_importance, "xgboost::xgb.importance", function(...) {
    data.frame(
      Feature = c("hp", "wt", "cyl", "disp"),
      Gain = c(0.4, 0.3, 0.2, 0.1),
      Cover = c(0.3, 0.4, 0.2, 0.1),
      Frequency = c(0.25, 0.25, 0.25, 0.25)
    )
  })

  # Mock the xgb.plot.importance function to avoid errors in testing
  mockery::stub(tl_plot_xgboost_importance, "xgboost::xgb.plot.importance", function(...) {
    structure(list(), class = "ggplot")
  })

  # Test plot creation
  p <- tl_plot_xgboost_importance(model)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_xgboost_tree creates tree plots", {
  skip_if_not_installed("xgboost")

  # Create an XGBoost model
  fit <- tl_fit_xgboost(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                        nrounds = 10)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "xgboost",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_xgboost", "tidylearn_model")

  # Mock the xgb.plot.tree function to avoid errors in testing
  mockery::stub(tl_plot_xgboost_tree, "xgboost::xgb.plot.tree", function(...) TRUE)

  # Test plot creation
  result <- tl_plot_xgboost_tree(model, tree_index = 0)
  expect_true(result)
})

test_that("tl_tune_xgboost tunes hyperparameters", {
  skip_if_not_installed("xgboost")

  # Define parameter grid
  param_grid <- list(
    max_depth = c(2, 3),
    eta = c(0.1, 0.3),
    subsample = c(0.8, 1.0)
  )

  # Mock cv.glmnet and xgb.cv to avoid errors in testing
  mockery::stub(tl_tune_xgboost, "xgboost::xgb.cv", function(...) {
    # Return a mock CV result
    list(
      evaluation_log = data.frame(
        iter = 1:10,
        train_rmse_mean = runif(10, 2, 3),
        train_rmse_std = runif(10, 0.1, 0.2),
        test_rmse_mean = runif(10, 3, 4),
        test_rmse_std = runif(10, 0.2, 0.3)
      ),
      best_iteration = 5,
      best_score = 3.5
    )
  })

  mockery::stub(tl_tune_xgboost, "xgboost::xgb.train", function(...) {
    # Return a mock XGBoost model
    model <- list(
      niter = 10,
      params = list(max_depth = 3, eta = 0.1)
    )
    class(model) <- "xgb.Booster"
    return(model)
  })

  # Test hyperparameter tuning
  tuned_model <- tl_tune_xgboost(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                                 param_grid = param_grid, cv_folds = 3,
                                 early_stopping_rounds = 5, verbose = FALSE)

  # Check if result is a tidylearn model
  expect_s3_class(tuned_model, "tidylearn_model")
  expect_s3_class(tuned_model, "tidylearn_xgboost")

  # Check if tuning results are stored
  expect_true(!is.null(attr(tuned_model, "tuning_results")))

  # Check if tuning results contain required elements
  tuning_results <- attr(tuned_model, "tuning_results")
  expect_true(all(c("param_grid", "results", "best_params", "best_score") %in%
                    names(tuning_results)))
})

test_that("tl_xgboost_shap calculates SHAP values", {
  skip_if_not_installed("xgboost")

  # Create an XGBoost model
  fit <- tl_fit_xgboost(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                        nrounds = 10)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "xgboost",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_xgboost", "tidylearn_model")

  # Mock predict with contrib parameter to avoid errors in testing
  mockery::stub(tl_xgboost_shap, "xgboost::predict", function(...) {
    # Return mock SHAP values
    matrix(rnorm(nrow(mtcars) * 2),
           nrow = nrow(mtcars),
           ncol = 2,
           dimnames = list(NULL, c("hp", "wt")))
  })

  # Test SHAP value calculation
  shap_values <- tl_xgboost_shap(model)

  # Check structure of results
  expect_s3_class(shap_values, "data.frame")
  expect_true(all(c("hp", "wt", "row_id") %in% names(shap_values)))

  # Check if all rows in the data have SHAP values
  expect_equal(nrow(shap_values), nrow(mtcars))
})

test_that("tl_plot_xgboost_shap_summary creates SHAP summary plots", {
  skip_if_not_installed("xgboost")
  skip_if_not_installed("ggplot2")

  # Create an XGBoost model
  fit <- tl_fit_xgboost(mtcars, mpg ~ hp + wt + cyl + disp,
                        is_classification = FALSE, nrounds = 10)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "xgboost",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_xgboost", "tidylearn_model")

  # Mock tl_xgboost_shap to avoid errors in testing
  mockery::stub(tl_plot_xgboost_shap_summary, "tl_xgboost_shap", function(...) {
    # Return mock SHAP values
    data.frame(
      hp = rnorm(nrow(mtcars)),
      wt = rnorm(nrow(mtcars)),
      cyl = rnorm(nrow(mtcars)),
      disp = rnorm(nrow(mtcars)),
      row_id = 1:nrow(mtcars)
    )
  })

  # Test plot creation
  if (requireNamespace("ggforce", quietly = TRUE)) {
    mockery::stub(tl_plot_xgboost_shap_summary, "ggforce::geom_sina", function(...) {
      ggplot2::geom_point(...)
    })
  }

  p <- tl_plot_xgboost_shap_summary(model, top_n = 3)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_xgboost_shap_dependence creates dependence plots", {
  skip_if_not_installed("xgboost")
  skip_if_not_installed("ggplot2")

  # Create an XGBoost model
  fit <- tl_fit_xgboost(mtcars, mpg ~ hp + wt + cyl + disp,
                        is_classification = FALSE, nrounds = 10)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "xgboost",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_xgboost", "tidylearn_model")

  # Mock tl_xgboost_shap to avoid errors in testing
  mockery::stub(tl_plot_xgboost_shap_dependence, "tl_xgboost_shap", function(...) {
    # Return mock SHAP values
    data.frame(
      hp = rnorm(nrow(mtcars)),
      wt = rnorm(nrow(mtcars)),
      cyl = rnorm(nrow(mtcars)),
      disp = rnorm(nrow(mtcars)),
      row_id = 1:nrow(mtcars)
    )
  })

  # Test plot for single feature
  p1 <- tl_plot_xgboost_shap_dependence(model, feature = "hp")
  expect_s3_class(p1, "ggplot")

  # Test plot with interaction feature
  p2 <- tl_plot_xgboost_shap_dependence(model, feature = "hp",
                                        interaction_feature = "wt")
  expect_s3_class(p2, "ggplot")
})

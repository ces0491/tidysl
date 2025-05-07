context("Regularization methods functionality tests")

# Sample data for testing
data(mtcars)
data(iris)

test_that("tl_fit_ridge correctly fits ridge regression", {
  skip_if_not_installed("glmnet")

  # Test regression ridge
  fit_reg <- tl_fit_ridge(mtcars, mpg ~ hp + wt + cyl + disp,
                          is_classification = FALSE, alpha = 0)

  # Check if fit is a glmnet model
  expect_s3_class(fit_reg, "glmnet")

  # Check if alpha is correctly set
  expect_equal(fit_reg$alpha, 0)

  # Check if predictions work
  preds <- predict(fit_reg, newx = as.matrix(mtcars[1:5, c("hp", "wt", "cyl", "disp")]))
  expect_true(is.matrix(preds))

  # Test classification ridge
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_ridge(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                            is_classification = TRUE, alpha = 0)

  # Check if fit is a glmnet model
  expect_s3_class(fit_class, "glmnet")

  # Check if alpha is correctly set
  expect_equal(fit_class$alpha, 0)

  # Check if family is correctly set
  expect_equal(fit_class$family, "binomial")
})

test_that("tl_fit_lasso correctly fits lasso regression", {
  skip_if_not_installed("glmnet")

  # Test regression lasso
  fit_reg <- tl_fit_lasso(mtcars, mpg ~ hp + wt + cyl + disp,
                          is_classification = FALSE, alpha = 1)

  # Check if fit is a glmnet model
  expect_s3_class(fit_reg, "glmnet")

  # Check if alpha is correctly set
  expect_equal(fit_reg$alpha, 1)

  # Check if predictions work
  preds <- predict(fit_reg, newx = as.matrix(mtcars[1:5, c("hp", "wt", "cyl", "disp")]))
  expect_true(is.matrix(preds))

  # Test classification lasso
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_lasso(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                            is_classification = TRUE, alpha = 1)

  # Check if fit is a glmnet model
  expect_s3_class(fit_class, "glmnet")

  # Check if alpha is correctly set
  expect_equal(fit_class$alpha, 1)

  # Check if family is correctly set
  expect_equal(fit_class$family, "binomial")
})

test_that("tl_fit_elastic_net correctly fits elastic net regression", {
  skip_if_not_installed("glmnet")

  # Test regression elastic net
  fit_reg <- tl_fit_elastic_net(mtcars, mpg ~ hp + wt + cyl + disp,
                                is_classification = FALSE, alpha = 0.5)

  # Check if fit is a glmnet model
  expect_s3_class(fit_reg, "glmnet")

  # Check if alpha is correctly set
  expect_equal(fit_reg$alpha, 0.5)

  # Check if predictions work
  preds <- predict(fit_reg, newx = as.matrix(mtcars[1:5, c("hp", "wt", "cyl", "disp")]))
  expect_true(is.matrix(preds))

  # Test classification elastic net
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_elastic_net(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                                  is_classification = TRUE, alpha = 0.5)

  # Check if fit is a glmnet model
  expect_s3_class(fit_class, "glmnet")

  # Check if alpha is correctly set
  expect_equal(fit_class$alpha, 0.5)

  # Check if family is correctly set
  expect_equal(fit_class$family, "binomial")
})

test_that("tl_predict_regularized correctly predicts with different types", {
  skip_if_not_installed("glmnet")

  # Create a regression ridge model
  x <- as.matrix(mtcars[, c("hp", "wt", "cyl", "disp")])
  y <- mtcars$mpg

  fit_reg <- glmnet::glmnet(x = x, y = y, alpha = 0)
  attr(fit_reg, "lambda_min") <- fit_reg$lambda[1]
  attr(fit_reg, "lambda_1se") <- fit_reg$lambda[1]

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "ridge",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_ridge", "tidylearn_model")

  # Mock model.matrix to avoid errors in testing
  mockery::stub(tl_predict_regularized, "stats::model.matrix", function(...) {
    as.matrix(mtcars[1:5, c("hp", "wt", "cyl", "disp")])
  })

  # Test response predictions for regression
  preds_reg <- tl_predict_regularized(model_reg, mtcars[1:5, ], lambda = "1se")
  expect_true(is.numeric(preds_reg))
  expect_length(preds_reg, 5)

  # Create a classification ridge model
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  x_class <- as.matrix(iris_sub[, c("Sepal.Length", "Sepal.Width")])
  y_class <- iris_sub$Species

  fit_class <- glmnet::glmnet(x = x_class, y = y_class, family = "binomial", alpha = 0)
  fit_class$family <- "binomial"  # Add family manually for test
  attr(fit_class, "lambda_min") <- fit_class$lambda[1]
  attr(fit_class, "lambda_1se") <- fit_class$lambda[1]

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "ridge",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris_sub
  )
  class(model_class) <- c("tidylearn_ridge", "tidylearn_model")

  # Mock model.matrix and predict.glmnet to avoid errors in testing
  mockery::stub(tl_predict_regularized, "glmnet::predict.glmnet", function(...) {
    # Return probabilities for "prob" type, classes for "class" type
    args <- list(...)
    if ("type" %in% names(args) && args$type == "response") {
      return(runif(5, 0, 1))  # Mocked probabilities
    } else {
      return(matrix(runif(5), ncol = 1))  # Mocked linear predictor
    }
  })

  # Test probability predictions for classification
  preds_prob <- tl_predict_regularized(model_class, iris_sub[1:5, ], type = "prob")
  expect_s3_class(preds_prob, "data.frame")
  expect_equal(ncol(preds_prob), 2)

  # Test class predictions for classification
  preds_class <- tl_predict_regularized(model_class, iris_sub[1:5, ], type = "class")
  expect_s3_class(preds_class, "factor")
  expect_length(preds_class, 5)
})

test_that("tl_plot_regularization_path creates regularization path plots", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("ggplot2")

  # Create a ridge model
  x <- as.matrix(mtcars[, c("hp", "wt", "cyl", "disp")])
  y <- mtcars$mpg

  fit_reg <- glmnet::glmnet(x = x, y = y, alpha = 0)

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "ridge",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_ridge", "tidylearn_model")

  # Mock coef function to avoid errors in testing
  mockery::stub(tl_plot_regularization_path, "coef", function(...) {
    # Create mock coefficient matrix with row and column names
    coef_matrix <- matrix(
      rnorm(4 * 10),
      nrow = 4,
      ncol = 10,
      dimnames = list(c("hp", "wt", "cyl", "disp"), NULL)
    )
    return(coef_matrix)
  })

  # Test plot creation
  p <- tl_plot_regularization_path(model_reg)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_cv_results creates cross-validation plots", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("ggplot2")

  # Create a ridge model
  x <- as.matrix(mtcars[, c("hp", "wt", "cyl", "disp")])
  y <- mtcars$mpg

  # Create mock CV results
  cv_results <- list(
    lambda = seq(0.1, 1, length.out = 10),
    cvm = rnorm(10, 5, 1),
    cvsd = rep(0.5, 10),
    lambda.min = 0.3,
    lambda.1se = 0.7
  )

  fit_reg <- glmnet::glmnet(x = x, y = y, alpha = 0)
  attr(fit_reg, "cv_results") <- cv_results

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "ridge",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_ridge", "tidylearn_model")

  # Test plot creation
  p <- tl_plot_cv_results(model_reg)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_importance_regularized creates importance plots", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("ggplot2")

  # Create a lasso model
  x <- as.matrix(mtcars[, c("hp", "wt", "cyl", "disp")])
  y <- mtcars$mpg

  fit_reg <- glmnet::glmnet(x = x, y = y, alpha = 1)
  attr(fit_reg, "lambda_min") <- fit_reg$lambda[1]
  attr(fit_reg, "lambda_1se") <- fit_reg$lambda[1]

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "lasso",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_lasso", "tidylearn_model")

  # Mock coef function to avoid errors in testing
  mockery::stub(tl_plot_importance_regularized, "coef", function(...) {
    # Create mock coefficient matrix with row names
    coef_matrix <- matrix(
      c(1.5, 0.3, 0, -0.5, 0),
      ncol = 1,
      dimnames = list(c("(Intercept)", "hp", "wt", "cyl", "disp"), NULL)
    )
    return(coef_matrix)
  })

  # Test plot creation
  p <- tl_plot_importance_regularized(model_reg)
  expect_s3_class(p, "ggplot")
})

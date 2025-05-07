context("Tree-based models functionality tests")

# Sample data for testing
data(mtcars)
data(iris)

test_that("tl_fit_tree correctly fits decision trees", {
  skip_if_not_installed("rpart")

  # Test regression tree
  fit_reg <- tl_fit_tree(mtcars, mpg ~ hp + wt, is_classification = FALSE)

  # Check if fit is an rpart model
  expect_s3_class(fit_reg, "rpart")

  # Check if predictions work
  preds <- predict(fit_reg, mtcars[1:5, ])
  expect_length(preds, 5)

  # Test classification tree
  # Subset iris to make it faster for tests
  iris_sub <- iris[1:50, ]
  fit_class <- tl_fit_tree(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                           is_classification = TRUE)

  # Check if fit is an rpart model
  expect_s3_class(fit_class, "rpart")

  # Check if predictions work
  preds_class <- predict(fit_class, iris_sub[1:5, ])
  expect_true(is.matrix(preds_class))
})

test_that("tl_predict_tree correctly predicts with different types", {
  skip_if_not_installed("rpart")

  # Create a regression tree model
  fit_reg <- tl_fit_tree(mtcars, mpg ~ hp + wt, is_classification = FALSE)

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "tree",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_tree", "tidylearn_model")

  # Test response predictions for regression
  preds_reg <- tl_predict_tree(model_reg, mtcars[1:5, ])
  expect_length(preds_reg, 5)
  expect_true(is.numeric(preds_reg))

  # Create a classification tree model
  # Subset iris to make it faster for tests
  iris_sub <- iris[1:50, ]
  fit_class <- tl_fit_tree(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                           is_classification = TRUE)

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "tree",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris_sub
  )
  class(model_class) <- c("tidylearn_tree", "tidylearn_model")

  # Test probability predictions for classification
  probs <- tl_predict_tree(model_class, iris_sub[1:5, ], type = "prob")
  expect_s3_class(probs, "data.frame")
  expect_equal(ncol(probs), length(levels(iris_sub$Species)))

  # Test class predictions for classification
  classes <- tl_predict_tree(model_class, iris_sub[1:5, ], type = "class")
  expect_s3_class(classes, "factor")
  expect_length(classes, 5)
})

test_that("tl_fit_forest correctly fits random forests", {
  skip_if_not_installed("randomForest")

  # Test regression forest
  fit_reg <- tl_fit_forest(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                           ntree = 10)  # Use small ntree for tests

  # Check if fit is a randomForest model
  expect_s3_class(fit_reg, "randomForest")

  # Check if predictions work
  preds <- predict(fit_reg, mtcars[1:5, ])
  expect_length(preds, 5)

  # Test classification forest
  # Subset iris to make it faster for tests
  iris_sub <- iris[1:50, ]
  fit_class <- tl_fit_forest(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                             is_classification = TRUE, ntree = 10)  # Use small ntree for tests

  # Check if fit is a randomForest model
  expect_s3_class(fit_class, "randomForest")

  # Check if predictions work
  preds_class <- predict(fit_class, iris_sub[1:5, ])
  expect_s3_class(preds_class, "factor")
})

test_that("tl_predict_forest correctly predicts with different types", {
  skip_if_not_installed("randomForest")

  # Create a regression forest model
  fit_reg <- tl_fit_forest(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                           ntree = 10)  # Use small ntree for tests

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "forest",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_forest", "tidylearn_model")

  # Test response predictions for regression
  preds_reg <- tl_predict_forest(model_reg, mtcars[1:5, ])
  expect_length(preds_reg, 5)
  expect_true(is.numeric(preds_reg))

  # Create a classification forest model
  # Subset iris to make it faster for tests
  iris_sub <- iris[1:50, ]
  fit_class <- tl_fit_forest(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                             is_classification = TRUE, ntree = 10)  # Use small ntree for tests

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "forest",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris_sub
  )
  class(model_class) <- c("tidylearn_forest", "tidylearn_model")

  # Test probability predictions for classification
  probs <- tl_predict_forest(model_class, iris_sub[1:5, ], type = "prob")
  expect_s3_class(probs, "data.frame")
  expect_equal(ncol(probs), length(levels(iris_sub$Species)))

  # Test class predictions for classification
  classes <- tl_predict_forest(model_class, iris_sub[1:5, ], type = "class")
  expect_s3_class(classes, "factor")
  expect_length(classes, 5)
})

test_that("tl_fit_boost correctly fits gradient boosting models", {
  skip_if_not_installed("gbm")

  # Test regression boosting
  fit_reg <- tl_fit_boost(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                          n.trees = 10)  # Use small n.trees for tests

  # Check if fit is a gbm model
  expect_s3_class(fit_reg, "gbm")

  # Check if predictions work
  preds <- predict(fit_reg, mtcars[1:5, ], n.trees = 10)
  expect_length(preds, 5)

  # Test classification boosting
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_boost(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                            is_classification = TRUE, n.trees = 10)  # Use small n.trees for tests

  # Check if fit is a gbm model
  expect_s3_class(fit_class, "gbm")

  # Check if predictions work
  preds_class <- predict(fit_class, iris_sub[1:5, ], n.trees = 10)
  expect_length(preds_class, 5)
})

test_that("tl_predict_boost correctly predicts with different types", {
  skip_if_not_installed("gbm")

  # Create a regression boosting model
  fit_reg <- tl_fit_boost(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                          n.trees = 10)  # Use small n.trees for tests

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "boost",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidylearn_boost", "tidylearn_model")

  # Test response predictions for regression
  preds_reg <- tl_predict_boost(model_reg, mtcars[1:5, ])
  expect_length(preds_reg, 5)
  expect_true(is.numeric(preds_reg))

  # Create a classification boosting model
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_boost(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                            is_classification = TRUE, n.trees = 10)  # Use small n.trees for tests

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "boost",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris_sub
  )
  class(model_class) <- c("tidylearn_boost", "tidylearn_model")

  # Test probability predictions for classification
  probs <- tl_predict_boost(model_class, iris_sub[1:5, ], type = "prob")
  expect_s3_class(probs, "data.frame")
  expect_equal(ncol(probs), length(levels(iris_sub$Species)))

  # Test class predictions for classification
  classes <- tl_predict_boost(model_class, iris_sub[1:5, ], type = "class")
  expect_s3_class(classes, "factor")
  expect_length(classes, 5)
})

test_that("tl_plot_tree creates tree plots", {
  skip_if_not_installed("rpart")
  skip_if_not_installed("rpart.plot")

  # Create a tree model
  fit <- tl_fit_tree(mtcars, mpg ~ hp + wt, is_classification = FALSE)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "tree",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_tree", "tidylearn_model")

  # Stub the rpart.plot function to verify it's called
  mockery::stub(tl_plot_tree, "rpart.plot::rpart.plot", function(...) TRUE)

  # Test plot creation
  result <- tl_plot_tree(model)
  expect_true(result)
})

test_that("tl_plot_importance creates importance plots", {
  skip_if_not_installed("randomForest")
  skip_if_not_installed("ggplot2")

  # Create a forest model
  fit <- tl_fit_forest(mtcars, mpg ~ hp + wt + cyl + disp,
                       is_classification = FALSE, ntree = 10)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt + cyl + disp,
      method = "forest",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_forest", "tidylearn_model")

  # Mock the importance function to avoid errors in testing
  mockery::stub(tl_plot_importance, "randomForest::importance", function(...) {
    matrix(runif(4), nrow = 4, dimnames = list(c("hp", "wt", "cyl", "disp"), "%IncMSE"))
  })

  # Test plot creation
  p <- tl_plot_importance(model)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_partial_dependence creates partial dependence plots", {
  skip_if_not_installed("randomForest")
  skip_if_not_installed("ggplot2")

  # Create a forest model
  fit <- tl_fit_forest(mtcars, mpg ~ hp + wt,
                       is_classification = FALSE, ntree = 10)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "forest",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidylearn_forest", "tidylearn_model")

  # Mock the predict function to avoid errors in testing
  mockery::stub(tl_plot_partial_dependence, "predict", function(...) {
    runif(nrow(mtcars), 10, 30)
  })

  # Test plot creation
  p <- tl_plot_partial_dependence(model, "hp")
  expect_s3_class(p, "ggplot")
})

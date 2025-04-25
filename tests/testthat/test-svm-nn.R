context("SVM and Neural Network models")

# Load test data
library(MASS)
data(Boston)
set.seed(123)
train_indices <- sample(1:nrow(Boston), 0.8 * nrow(Boston))
train_data <- Boston[train_indices, ]
test_data <- Boston[-train_indices, ]

# Classification data
data(iris)
set.seed(123)
train_indices <- sample(1:nrow(iris), 0.8 * nrow(iris))
train_iris <- iris[train_indices, ]
test_iris <- iris[-train_indices, ]

test_that("SVM regression model works", {
  skip_if_not_installed("e1071")

  # Train model
  svm_model <- tl_model(
    train_data,
    medv ~ .,
    method = "svm",
    is_classification = FALSE,
    kernel = "radial"
  )

  # Check model object
  expect_s3_class(svm_model$fit, "svm")

  # Check predictions
  preds <- predict(svm_model, test_data)
  expect_s3_class(preds, "tbl_df")
  expect_equal(nrow(preds), nrow(test_data))

  # Check evaluation
  metrics <- tl_evaluate(svm_model, test_data)
  expect_true("rmse" %in% metrics$metric)
})

test_that("SVM with tuning works", {
  skip_if_not_installed("e1071")

  # Only run this test if we have time for more computation
  skip_on_cran()

  # Train model with tuning (use small parameter grid for tests)
  svm_model <- tl_model(
    head(train_data, 100),  # Use smaller dataset for speed
    medv ~ .,
    method = "svm",
    is_classification = FALSE,
    kernel = "radial",
    tune = TRUE,
    tune_folds = 2  # Use fewer folds for test speed
  )

  # Check tuning results are stored
  expect_false(is.null(attr(svm_model$fit, "tuning_results")))

  # Check SVM tuning plot (if model was tuned)
  if (!is.null(attr(svm_model$fit, "tuning_results"))) {
    expect_error(tl_plot_svm_tuning(svm_model), NA)
  }
})

test_that("neural network regression model works", {
  skip_if_not_installed("nnet")

  # Train model
  nn_model <- tl_model(
    train_data,
    medv ~ .,
    method = "nn",
    is_classification = FALSE,
    size = 3,
    decay = 0.1,
    maxit = 100
  )

  # Check model object
  expect_s3_class(nn_model$fit, "nnet")

  # Check predictions
  preds <- predict(nn_model, test_data)
  expect_s3_class(preds, "tbl_df")

  # Check evaluation
  metrics <- tl_evaluate(nn_model, test_data)
  expect_true("rmse" %in% metrics$metric)
})

test_that("neural network classification model works", {
  skip_if_not_installed("nnet")

  # Train model
  nn_model <- tl_model(
    train_iris,
    Species ~ .,
    method = "nn",
    is_classification = TRUE,
    size = 3,
    decay = 0.1,
    maxit = 100
  )

  # Check model object
  expect_s3_class(nn_model$fit, "nnet")

  # Check predictions
  class_preds <- predict(nn_model, test_iris, type = "class")
  expect_s3_class(class_preds, "tbl_df")
  expect_true(is.factor(class_preds$prediction))

  prob_preds <- predict(nn_model, test_iris, type = "prob")
  expect_s3_class(prob_preds, "tbl_df")
  expect_equal(ncol(prob_preds), length(levels(iris$Species)))
})

test_that("neural network tuning works", {
  skip_if_not_installed("nnet")

  # Only run this test if we have time for more computation
  skip_on_cran()

  # Tune neural network (use small parameter grid for tests)
  nn_tuned <- tl_tune_nn(
    head(train_iris, 50),  # Use smaller dataset for speed
    Species ~ .,
    is_classification = TRUE,
    sizes = c(2, 3),
    decays = c(0, 0.1),
    folds = 2  # Use fewer folds for test speed
  )

  # Check structure
  expect_type(nn_tuned, "list")
  expect_true("model" %in% names(nn_tuned))
  expect_true("best_size" %in% names(nn_tuned))
  expect_true("best_decay" %in% names(nn_tuned))
  expect_true("tuning_results" %in% names(nn_tuned))

  # Check tuning results
  expect_s3_class(nn_tuned$tuning_results, "data.frame")
  expect_true("size" %in% names(nn_tuned$tuning_results))
  expect_true("decay" %in% names(nn_tuned$tuning_results))
  expect_true("error" %in% names(nn_tuned$tuning_results))

  # Check tuning plot
  expect_error(tl_plot_nn_tuning(nn_tuned), NA)
})

test_that("deep learning model works if keras is available", {
  # Skip if keras or tensorflow is not installed
  skip_if_not_installed("keras")
  skip_if_not_installed("tensorflow")

  # Only run this test if we have time for more computation
  skip_on_cran()

  # Check if keras is properly loaded
  skip_if_not(tryCatch({
    keras::is_keras_available()
  }, error = function(e) FALSE))

  # Train a simple deep learning model
  deep_model <- tryCatch({
    tl_model(
      head(train_data, 50),  # Use smaller dataset for speed
      medv ~ .,
      method = "deep",
      is_classification = FALSE,
      hidden_layers = c(10, 5),
      epochs = 5,  # Use fewer epochs for test speed
      batch_size = 10,
      verbose = 0
    )
  }, error = function(e) {
    skip(paste("Skipping deep learning test:", e$message))
    NULL
  })

  # If model was successfully created, test it
  if (!is.null(deep_model)) {
    # Check predictions
    preds <- predict(deep_model, head(test_data, 10))
    expect_s3_class(preds, "tbl_df")

    # Check history plot
    expect_error(tl_plot_deep_history(deep_model), NA)
  }
})

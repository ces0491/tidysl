context("SVM and Neural Network functionality tests")

# Sample data for testing
data(mtcars)
data(iris)

test_that("tl_fit_svm correctly fits SVM models", {
  skip_if_not_installed("e1071")

  # Test regression SVM
  fit_reg <- tl_fit_svm(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                        kernel = "radial", cost = 1)

  # Check if fit is an SVM model
  expect_s3_class(fit_reg, "svm")

  # Check if kernel is correctly set
  expect_equal(fit_reg$kernel, 1)  # 0=linear, 1=polynomial, 2=radial, 3=sigmoid

  # Check if cost is correctly set
  expect_equal(fit_reg$cost, 1)

  # Check if predictions work
  preds <- predict(fit_reg, mtcars[1:5, ])
  expect_length(preds, 5)

  # Test classification SVM
  # Subset iris to make it faster for tests
  iris_sub <- iris[1:50, ]
  fit_class <- tl_fit_svm(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                          is_classification = TRUE, kernel = "radial", cost = 1)

  # Check if fit is an SVM model
  expect_s3_class(fit_class, "svm")

  # Check if type is correctly set
  expect_equal(fit_class$type, 0)  # 0=C-classification, others are regression

  # Check if predictions work
  preds_class <- predict(fit_class, iris_sub[1:5, ])
  expect_s3_class(preds_class, "factor")
})

test_that("tl_predict_svm correctly predicts with different types", {
  skip_if_not_installed("e1071")

  # Create a regression SVM model
  fit_reg <- tl_fit_svm(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                        kernel = "radial", cost = 1)

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "svm",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidysl_svm", "tidysl_model")

  # Test response predictions for regression
  preds_reg <- tl_predict_svm(model_reg, mtcars[1:5, ])
  expect_length(preds_reg, 5)
  expect_true(is.numeric(preds_reg))

  # Create a classification SVM model
  # Subset iris to make it faster for tests
  iris_sub <- iris[1:50, ]
  fit_class <- tl_fit_svm(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                          is_classification = TRUE, kernel = "radial",
                          cost = 1, probability = TRUE)

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "svm",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris_sub
  )
  class(model_class) <- c("tidysl_svm", "tidysl_model")

  # Mock the predict.svm function to avoid errors in testing
  mockery::stub(tl_predict_svm, "e1071::predict.svm", function(object, newdata, probability, ...) {
    if (missing(probability)) probability <- FALSE

    if (probability) {
      # Return class predictions with probability attribute
      preds <- factor(c("setosa", "setosa", "setosa", "setosa", "setosa"),
                      levels = c("setosa", "versicolor"))
      probs <- matrix(c(0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5),
                      ncol = 2,
                      dimnames = list(NULL, c("setosa", "versicolor")))
      attr(preds, "probabilities") <- probs
      return(preds)
    } else {
      # Return class predictions
      return(factor(c("setosa", "setosa", "setosa", "setosa", "setosa"),
                    levels = c("setosa", "versicolor")))
    }
  })

  # Test probability predictions for classification
  probs <- tl_predict_svm(model_class, iris_sub[1:5, ], type = "prob")
  expect_s3_class(probs, "data.frame")
  expect_equal(ncol(probs), 2)  # Binary classification has 2 classes

  # Test class predictions for classification
  classes <- tl_predict_svm(model_class, iris_sub[1:5, ], type = "class")
  expect_s3_class(classes, "factor")
  expect_length(classes, 5)
})

test_that("tl_plot_svm_boundary creates decision boundary plots", {
  skip_if_not_installed("e1071")
  skip_if_not_installed("ggplot2")

  # Create a classification SVM model
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_svm(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                          is_classification = TRUE, kernel = "radial")

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "svm",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris_sub
  )
  class(model_class) <- c("tidysl_svm", "tidysl_model")

  # Mock the predict.svm function to avoid errors in testing
  mockery::stub(tl_plot_svm_boundary, "e1071::predict.svm", function(...) {
    # Return mocked probabilities or decision values
    runif(100, 0, 1)  # For 10x10 grid
  })

  # Test plot creation
  p <- tl_plot_svm_boundary(model_class, x_var = "Sepal.Length", y_var = "Sepal.Width")
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_svm_tuning creates tuning plots", {
  skip_if_not_installed("e1071")
  skip_if_not_installed("ggplot2")

  # Create a mock tuning result
  tune_result <- list(
    performances = data.frame(
      gamma = rep(c(0.1, 1), each = 3),
      cost = rep(c(0.1, 1, 10), times = 2),
      error = c(0.3, 0.2, 0.1, 0.15, 0.1, 0.2)
    ),
    best.parameters = list(gamma = 1, cost = 1),
    best.performance = 0.1
  )

  # Create an SVM model with tuning results
  fit_class <- tl_fit_svm(iris, Species ~ Sepal.Length + Sepal.Width,
                          is_classification = TRUE, kernel = "radial")

  # Add tuning results attribute
  attr(fit_class, "tuning_results") <- tune_result

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "svm",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris
  )
  class(model_class) <- c("tidysl_svm", "tidysl_model")

  # Test plot creation
  p <- tl_plot_svm_tuning(model_class)
  expect_s3_class(p, "ggplot")
})

test_that("tl_fit_nn correctly fits neural network models", {
  skip_if_not_installed("nnet")

  # Test regression neural network
  fit_reg <- tl_fit_nn(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                       size = 3, decay = 0.01, maxit = 100)

  # Check if fit is a neural network model
  expect_s3_class(fit_reg, "nnet")

  # Check if size is correctly set
  expect_equal(fit_reg$n[2], 3)  # n[2] is the size of the hidden layer

  # Check if decay is correctly set
  expect_equal(fit_reg$decay, 0.01)

  # Test classification neural network
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_nn(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                         is_classification = TRUE, size = 3, decay = 0.01)

  # Check if fit is a neural network model
  expect_s3_class(fit_class, "nnet")

  # Check if size is correctly set
  expect_equal(fit_class$n[2], 3)  # n[2] is the size of the hidden layer
})

test_that("tl_predict_nn correctly predicts with different types", {
  skip_if_not_installed("nnet")

  # Create a regression neural network model
  fit_reg <- tl_fit_nn(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                       size = 3, decay = 0.01, maxit = 100)

  model_reg <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "nn",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit_reg,
    data = mtcars
  )
  class(model_reg) <- c("tidysl_nn", "tidysl_model")

  # Mock the predict.nnet function to avoid errors in testing
  mockery::stub(tl_predict_nn, "nnet::predict.nnet", function(...) {
    runif(5, 10, 30)  # Mocked predictions
  })

  # Test response predictions for regression
  preds_reg <- tl_predict_nn(model_reg, mtcars[1:5, ])
  expect_length(preds_reg, 5)
  expect_true(is.numeric(preds_reg))

  # Create a classification neural network model
  # Subset iris to make binary classification problem
  iris_sub <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_sub$Species <- factor(iris_sub$Species)

  fit_class <- tl_fit_nn(iris_sub, Species ~ Sepal.Length + Sepal.Width,
                         is_classification = TRUE, size = 3, decay = 0.01)

  model_class <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "nn",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit_class,
    data = iris_sub
  )
  class(model_class) <- c("tidysl_nn", "tidysl_model")

  # Mock the predict.nnet function to avoid errors in testing for classification
  mockery::stub(tl_predict_nn, "nnet::predict.nnet", function(object, newdata, type, ...) {
    if (type == "raw") {
      # Return probabilities
      matrix(c(0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5),
             ncol = 2)
    } else {
      # Return class
      c(1, 1, 1, 1, 1)
    }
  })

  # Test probability predictions for classification
  probs <- tl_predict_nn(model_class, iris_sub[1:5, ], type = "prob")
  expect_s3_class(probs, "data.frame")
  expect_equal(ncol(probs), 2)  # Binary classification has 2 classes

  # Test class predictions for classification
  classes <- tl_predict_nn(model_class, iris_sub[1:5, ], type = "class")
  expect_s3_class(classes, "factor")
  expect_length(classes, 5)
})

test_that("tl_plot_nn_architecture creates architecture plots", {
  skip_if_not_installed("nnet")

  # Create a neural network model
  fit <- tl_fit_nn(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                   size = 3, decay = 0.01, maxit = 100)

  model <- list(
    spec = list(
      formula = mpg ~ hp + wt,
      method = "nn",
      is_classification = FALSE,
      response_var = "mpg"
    ),
    fit = fit,
    data = mtcars
  )
  class(model) <- c("tidysl_nn", "tidysl_model")

  # Mock the NeuralNetTools::plotnet function to avoid errors in testing
  mockery::stub(tl_plot_nn_architecture, "NeuralNetTools::plotnet", function(...) TRUE)

  # Test plot creation
  result <- tl_plot_nn_architecture(model)
  expect_true(result)
})

test_that("tl_tune_nn tunes neural network hyperparameters", {
  skip_if_not_installed("nnet")

  # Mock the tl_fit_nn function to avoid errors in testing
  mockery::stub(tl_tune_nn, "tl_fit_nn", function(...) {
    # Return a mock neural network model
    list(
      n = c(2, 3, 1),  # Input, hidden, output nodes
      decay = 0.01,
      wts = runif(10),  # Random weights
      value = 0.1       # Error value
    )
  })

  # Test hyperparameter tuning
  tuned_nn <- tl_tune_nn(mtcars, mpg ~ hp + wt, is_classification = FALSE,
                         sizes = c(1, 3), decays = c(0, 0.01))

  # Check structure of results
  expect_type(tuned_nn, "list")
  expect_true(all(c("model", "best_size", "best_decay", "tuning_results") %in%
                    names(tuned_nn)))

  # Check if tuning results has the right structure
  expect_s3_class(tuned_nn$tuning_results, "data.frame")
  expect_true(all(c("size", "decay", "error") %in% names(tuned_nn$tuning_results)))

  # Check if best size and decay are valid
  expect_true(tuned_nn$best_size %in% c(1, 3))
  expect_true(tuned_nn$best_decay %in% c(0, 0.01))
})

test_that("tl_plot_nn_tuning creates tuning plots", {
  skip_if_not_installed("nnet")
  skip_if_not_installed("ggplot2")

  # Create mock tuning results
  tuning_results <- data.frame(
    size = rep(c(1, 3, 5), each = 3),
    decay = rep(c(0, 0.01, 0.1), times = 3),
    error = runif(9, 0.1, 0.3)
  )

  # Create mock tuned model
  tuned_nn <- list(
    model = structure(list(), class = "nnet"),
    best_size = 3,
    best_decay = 0.01,
    tuning_results = tuning_results
  )

  # Test plot creation
  p <- tl_plot_nn_tuning(tuned_nn)
  expect_s3_class(p, "ggplot")
})

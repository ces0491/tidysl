context("Classification functionality tests")

# Sample data for testing
data(iris)

test_that("tl_fit_logistic correctly fits logistic regression", {
  # Create binary outcome dataset
  binary_data <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  binary_data$Species <- factor(binary_data$Species)

  # Test internal function
  fit <- tl_fit_logistic(binary_data, Species ~ Sepal.Length + Sepal.Width)

  # Check if fit is a glm model
  expect_s3_class(fit, "glm")

  # Check if family is binomial
  expect_equal(fit$family$family, "binomial")

  # Check if coefficients are calculated
  expect_named(coef(fit), c("(Intercept)", "Sepal.Length", "Sepal.Width"))

  # Check if predictions work
  preds <- predict(fit, binary_data[1:5, ], type = "response")
  expect_length(preds, 5)
  expect_true(all(preds >= 0 & preds <= 1))
})

test_that("tl_predict_logistic correctly predicts with different types", {
  # Create binary outcome dataset
  binary_data <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  binary_data$Species <- factor(binary_data$Species)

  # Create a model
  fit <- glm(Species ~ Sepal.Length + Sepal.Width, data = binary_data, family = binomial())

  model <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "logistic",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit,
    data = binary_data
  )
  class(model) <- c("tidylearn_logistic", "tidylearn_model")

  # Test probability predictions
  probs <- tl_predict_logistic(model, binary_data[1:5, ], type = "prob")
  expect_s3_class(probs, "data.frame")
  expect_equal(ncol(probs), 2)
  expect_true(all(rowSums(as.matrix(probs)) > 0.99 & rowSums(as.matrix(probs)) < 1.01))

  # Test class predictions
  classes <- tl_predict_logistic(model, binary_data[1:5, ], type = "class")
  expect_s3_class(classes, "factor")
  expect_equal(levels(classes), levels(binary_data$Species))

  # Test response predictions (raw probabilities)
  raw_probs <- tl_predict_logistic(model, binary_data[1:5, ], type = "response")
  expect_true(is.numeric(raw_probs))
  expect_true(all(raw_probs >= 0 & raw_probs <= 1))
})

test_that("tl_plot_roc creates ROC curve plots", {
  # Skip if required packages not installed
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("ROCR")

  # Create binary outcome dataset
  binary_data <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  binary_data$Species <- factor(binary_data$Species)

  # Create a model
  fit <- glm(Species ~ Sepal.Length + Sepal.Width, data = binary_data, family = binomial())

  model <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "logistic",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit,
    data = binary_data
  )
  class(model) <- c("tidylearn_logistic", "tidylearn_model")

  # Test ROC plot creation
  p <- tl_plot_roc(model)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_confusion creates confusion matrix plots", {
  # Skip if ggplot2 not installed
  skip_if_not_installed("ggplot2")

  # Create binary outcome dataset
  binary_data <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  binary_data$Species <- factor(binary_data$Species)

  # Create a model
  fit <- glm(Species ~ Sepal.Length + Sepal.Width, data = binary_data, family = binomial())

  model <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "logistic",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit,
    data = binary_data
  )
  class(model) <- c("tidylearn_logistic", "tidylearn_model")

  # Mock the predict function to avoid errors in testing
  mockery::stub(tl_plot_confusion, "predict", function(...) {
    list(prediction = sample(levels(binary_data$Species),
                             size = nrow(binary_data),
                             replace = TRUE,
                             prob = c(0.7, 0.3)))
  })

  # Test confusion matrix plot creation
  p <- tl_plot_confusion(model)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_precision_recall creates PR curve plots", {
  # Skip if required packages not installed
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("ROCR")

  # Create binary outcome dataset
  binary_data <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  binary_data$Species <- factor(binary_data$Species)

  # Create a model
  fit <- glm(Species ~ Sepal.Length + Sepal.Width, data = binary_data, family = binomial())

  model <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "logistic",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit,
    data = binary_data
  )
  class(model) <- c("tidylearn_logistic", "tidylearn_model")

  # Mock the predict function to avoid errors in testing
  mockery::stub(tl_plot_precision_recall, "predict", function(...) {
    list(prob = setNames(
      data.frame(
        runif(nrow(binary_data), 0.3, 0.7),
        runif(nrow(binary_data), 0.3, 0.7)
      ),
      levels(binary_data$Species)
    ))
  })

  # Test PR curve plot creation
  p <- tl_plot_precision_recall(model)
  expect_s3_class(p, "ggplot")
})

test_that("tl_plot_calibration creates calibration plots", {
  # Skip if ggplot2 not installed
  skip_if_not_installed("ggplot2")

  # Create binary outcome dataset
  binary_data <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  binary_data$Species <- factor(binary_data$Species)

  # Create a model
  fit <- glm(Species ~ Sepal.Length + Sepal.Width, data = binary_data, family = binomial())

  model <- list(
    spec = list(
      formula = Species ~ Sepal.Length + Sepal.Width,
      method = "logistic",
      is_classification = TRUE,
      response_var = "Species"
    ),
    fit = fit,
    data = binary_data
  )
  class(model) <- c("tidylearn_logistic", "tidylearn_model")

  # Mock the predict function to avoid errors in testing
  mockery::stub(tl_plot_calibration, "predict", function(...) {
    list(prob = setNames(
      data.frame(
        runif(nrow(binary_data), 0.3, 0.7),
        runif(nrow(binary_data), 0.3, 0.7)
      ),
      levels(binary_data$Species)
    ))
  })

  # Test calibration plot creation
  p <- tl_plot_calibration(model)
  expect_s3_class(p, "ggplot")
})

test_that("tl_calc_classification_metrics calculates metrics correctly", {
  # Create test data
  actuals <- factor(c("A", "A", "B", "B", "A", "B", "A", "B"))
  predicted <- factor(c("A", "A", "B", "A", "A", "B", "B", "B"))

  # Test basic metrics calculation
  metrics <- tl_calc_classification_metrics(
    actuals = actuals,
    predicted = predicted,
    metrics = c("accuracy", "precision", "recall")
  )

  # Check metrics structure
  expect_s3_class(metrics, "data.frame")
  expect_equal(nrow(metrics), 3)
  expect_equal(metrics$metric, c("accuracy", "precision", "recall"))
  expect_true(all(is.numeric(metrics$value)))

  # Check accuracy calculation
  accuracy <- metrics$value[metrics$metric == "accuracy"]
  expect_equal(accuracy, 0.75)  # 6 out of 8 correct
})

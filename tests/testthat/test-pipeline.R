context("Pipeline functionality tests")

# Sample data for testing
data(mtcars)
data(iris)

test_that("tl_pipeline creates a valid pipeline", {
  # Create a basic pipeline
  pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)

  # Check if pipeline is created with the right class
  expect_s3_class(pipeline, "tidysl_pipeline")

  # Check if pipeline contains required components
  expect_true(all(c("formula", "data", "preprocessing", "models", "evaluation") %in% names(pipeline)))

  # Check if formula is correctly stored
  expect_equal(as.character(pipeline$formula)[2], "mpg ~ hp + wt")

  # Check if default preprocessing is created
  expect_true(is.list(pipeline$preprocessing))
  expect_true(pipeline$preprocessing$impute_missing)

  # Check if default models are created
  expect_true(is.list(pipeline$models))
  expect_true(length(pipeline$models) >= 2)

  # Check if default evaluation is created
  expect_true(is.list(pipeline$evaluation))
  expect_true("metrics" %in% names(pipeline$evaluation))
})

test_that("tl_run_pipeline executes pipeline steps", {
  # Create a simple pipeline
  pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)

  # Run the pipeline
  pipeline_result <- tl_run_pipeline(pipeline, verbose = FALSE)

  # Check if result is still a pipeline
  expect_s3_class(pipeline_result, "tidysl_pipeline")

  # Check if results are stored
  expect_true("results" %in% names(pipeline_result))
  expect_false(is.null(pipeline_result$results))

  # Check if processed data is stored
  expect_true("processed_data" %in% names(pipeline_result$results))

  # Check if model results are stored
  expect_true("model_results" %in% names(pipeline_result$results))
  expect_true(length(pipeline_result$results$model_results) > 0)

  # Check if best model is identified
  expect_true("best_model_name" %in% names(pipeline_result$results))
  expect_true("best_model" %in% names(pipeline_result$results))
  expect_s3_class(pipeline_result$results$best_model, "tidysl_model")
})

test_that("tl_get_best_model extracts best model from pipeline", {
  # Create and run a pipeline
  pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)
  pipeline_result <- tl_run_pipeline(pipeline, verbose = FALSE)

  # Get best model
  best_model <- tl_get_best_model(pipeline_result)

  # Check if best model is a tidysl model
  expect_s3_class(best_model, "tidysl_model")

  # Check if best model has the same formula
  expect_equal(as.character(best_model$spec$formula)[2], "mpg ~ hp + wt")

  # Check for error when pipeline has not been run
  empty_pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)
  expect_error(tl_get_best_model(empty_pipeline))
})

test_that("tl_compare_pipeline_models creates comparison plots", {
  skip_if_not_installed("ggplot2")

  # Create and run a pipeline
  pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)
  pipeline_result <- tl_run_pipeline(pipeline, verbose = FALSE)

  # Create comparison plot
  p <- tl_compare_pipeline_models(pipeline_result)

  # Check if plot is created
  expect_s3_class(p, "ggplot")

  # Create comparison plot with specific metrics
  p2 <- tl_compare_pipeline_models(pipeline_result, metrics = c("rmse", "rsq"))

  # Check if plot is created
  expect_s3_class(p2, "ggplot")

  # Check for error when pipeline has not been run
  empty_pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)
  expect_error(tl_compare_pipeline_models(empty_pipeline))
})

test_that("tl_predict_pipeline makes predictions using pipeline models", {
  # Create and run a pipeline
  pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)
  pipeline_result <- tl_run_pipeline(pipeline, verbose = FALSE)

  # Make predictions using best model
  preds <- tl_predict_pipeline(pipeline_result, mtcars[1:5, ])

  # Check if predictions are generated
  expect_true(is.numeric(preds) || is.factor(preds))
  expect_length(preds, 5)

  # Make predictions using specific model
  model_names <- names(pipeline_result$results$model_results)

  if (length(model_names) > 0) {
    preds2 <- tl_predict_pipeline(pipeline_result, mtcars[1:5, ],
                                  model_name = model_names[1])

    # Check if predictions are generated
    expect_true(is.numeric(preds2) || is.factor(preds2))
    expect_length(preds2, 5)
  }

  # Check for error when pipeline has not been run
  empty_pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)
  expect_error(tl_predict_pipeline(empty_pipeline, mtcars[1:5, ]))
})

test_that("tl_save_pipeline and tl_load_pipeline work correctly", {
  skip_if_not(dir.exists(tempdir()))

  # Create a pipeline
  pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)

  # Create temp file path
  temp_file <- file.path(tempdir(), "test_pipeline.rds")

  # Save pipeline
  tl_save_pipeline(pipeline, temp_file)

  # Check if file exists
  expect_true(file.exists(temp_file))

  # Load pipeline
  loaded_pipeline <- tl_load_pipeline(temp_file)

  # Check if loaded pipeline is a tidysl_pipeline
  expect_s3_class(loaded_pipeline, "tidysl_pipeline")

  # Check if formulas match
  expect_equal(as.character(pipeline$formula), as.character(loaded_pipeline$formula))

  # Cleanup
  if (file.exists(temp_file)) {
    file.remove(temp_file)
  }

  # Check for errors with invalid inputs
  expect_error(tl_save_pipeline(mtcars, temp_file))

  invalid_file <- file.path(tempdir(), "nonexistent_dir", "invalid.rds")
  expect_error(tl_save_pipeline(pipeline, invalid_file))
})

test_that("print and summary methods work for pipelines", {
  # Create a pipeline
  pipeline <- tl_pipeline(mtcars, mpg ~ hp + wt)

  # Check print method
  expect_output(print(pipeline), "Tidylearn Pipeline")
  expect_output(print(pipeline), "Formula:")
  expect_output(print(pipeline), "Data:")

  # Run pipeline
  pipeline_result <- tl_run_pipeline(pipeline, verbose = FALSE)

  # Check print method for run pipeline
  expect_output(print(pipeline_result), "Tidylearn Pipeline")
  expect_output(print(pipeline_result), "Best model:")

  # Check summary method
  expect_output(summary(pipeline_result), "Detailed Results")
  expect_output(summary(pipeline_result), "Best Model Summary")
})

test_that("pipeline handles classification models correctly", {
  # Create binary outcome dataset
  mtcars_binary <- mtcars
  mtcars_binary$am <- as.factor(mtcars_binary$am)

  # Create a classification pipeline
  pipeline <- tl_pipeline(mtcars_binary, am ~ hp + wt)

  # Run the pipeline
  pipeline_result <- tl_run_pipeline(pipeline, verbose = FALSE)

  # Check if result is a pipeline
  expect_s3_class(pipeline_result, "tidysl_pipeline")

  # Check if best model is identified
  expect_true("best_model_name" %in% names(pipeline_result$results))
  expect_true("best_model" %in% names(pipeline_result$results))
  expect_s3_class(pipeline_result$results$best_model, "tidysl_model")

  # Make predictions using best model
  preds <- tl_predict_pipeline(pipeline_result, mtcars_binary[1:5, ], type = "class")

  # Check if predictions are factors
  expect_s3_class(preds, "factor")
  expect_length(preds, 5)
})

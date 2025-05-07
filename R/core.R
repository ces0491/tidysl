#' @title tidylearn: A Tidy Approach to Supervised Learning
#' @name tidylearn-core
#' @description Core functionality for the tidylearn package
#' @importFrom magrittr %>%
#' @importFrom rlang .data .env
#' @importFrom dplyr filter select mutate group_by summarize arrange
#' @importFrom tibble tibble as_tibble
#' @importFrom purrr map map_dbl map_lgl map2
#' @importFrom tidyr nest unnest
#' @importFrom stats predict model.matrix formula as.formula
NULL

#' Pipe operator
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#' @return The result of applying rhs to lhs.
#' @description See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
NULL

#' @export
#' @rdname pipe
`%>%` <- magrittr::`%>%`

#' Create a tidylearn model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param method The modeling method to use (e.g., "linear", "logistic", "tree", etc.)
#' @param ... Additional arguments to pass to the underlying model function
#' @return A tidylearn model object
#' @export
tl_model <- function(data, formula, method = "linear", ...) {
  # Validate inputs
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame", call. = FALSE)
  }

  if (!inherits(formula, "formula")) {
    formula <- as.formula(formula)
  }

  # Extract response variable
  response_var <- all.vars(formula)[1]

  # Determine if classification or regression
  y <- data[[response_var]]
  is_classification <- is.factor(y) || is.character(y) || (is.numeric(y) && length(unique(y)) <= 10)

  if (is_classification && is.numeric(y)) {
    warning("Response appears to be categorical but is stored as numeric. Consider converting to factor.")
  }

  # Create model specification
  model_spec <- list(
    formula = formula,
    method = method,
    is_classification = is_classification,
    response_var = response_var
  )

  # Fit the model based on method
  fitted_model <- switch(
    method,
    "linear" = tl_fit_linear(data, formula, ...),
    "polynomial" = tl_fit_polynomial(data, formula, ...),
    "logistic" = tl_fit_logistic(data, formula, ...),
    "tree" = tl_fit_tree(data, formula, is_classification, ...),
    "forest" = tl_fit_forest(data, formula, is_classification, ...),
    "boost" = tl_fit_boost(data, formula, is_classification, ...),
    "ridge" = tl_fit_ridge(data, formula, is_classification, ...),
    "lasso" = tl_fit_lasso(data, formula, is_classification, ...),
    "elastic_net" = tl_fit_elastic_net(data, formula, is_classification, ...),
    "svm" = tl_fit_svm(data, formula, is_classification, ...),
    "nn" = tl_fit_nn(data, formula, is_classification, ...),
    "deep" = tl_fit_deep(data, formula, is_classification, ...),
    stop("Unsupported method: ", method, call. = FALSE)
  )

  # Create and return tidylearn model object
  model <- structure(
    list(
      spec = model_spec,
      fit = fitted_model,
      data = data
    ),
    class = c(paste0("tidylearn_", method), "tidylearn_model")
  )

  return(model)
}

#' Predict using a tidylearn model
#'
#' @param object A tidylearn model object
#' @param new_data A data frame containing the new data for prediction
#' @param type Type of prediction: "response", "prob", "class", etc.
#' @param ... Additional arguments
#' @return A tibble of predictions
#' @export
predict.tidylearn_model <- function(object, new_data = NULL, type = "response", ...) {
  if (is.null(new_data)) {
    new_data <- object$data
  }

  method <- object$spec$method
  is_classification <- object$spec$is_classification

  # Get raw predictions
  preds <- switch(
    method,
    "linear" = tl_predict_linear(object, new_data, type, ...),
    "polynomial" = tl_predict_polynomial(object, new_data, type, ...),
    "logistic" = tl_predict_logistic(object, new_data, type, ...),
    "tree" = tl_predict_tree(object, new_data, type, ...),
    "forest" = tl_predict_forest(object, new_data, type, ...),
    "boost" = tl_predict_boost(object, new_data, type, ...),
    "ridge" = tl_predict_ridge(object, new_data, type, ...),
    "lasso" = tl_predict_lasso(object, new_data, type, ...),
    "elastic_net" = tl_predict_elastic_net(object, new_data, type, ...),
    "svm" = tl_predict_svm(object, new_data, type, ...),
    "nn" = tl_predict_nn(object, new_data, type, ...),
    "deep" = tl_predict_deep(object, new_data, type, ...),
    stop("Unsupported method: ", method, call. = FALSE)
  )

  # Convert to tibble with appropriate structure based on prediction type
  if (is_classification && type == "prob") {
    # For probability predictions, return one column per class
    return(as_tibble(preds))
  } else {
    # For response or class predictions, return a single column
    return(tibble(prediction = preds))
  }
}

#' Evaluate a tidylearn model
#'
#' @param model A tidylearn model object
#' @param new_data Optional data frame for evaluation (if NULL, uses training data)
#' @param metrics Character vector of metrics to compute
#' @param ... Additional arguments
#' @return A tibble of evaluation metrics
#' @export
tl_evaluate <- function(model, new_data = NULL,
                        metrics = NULL, ...) {
  if (is.null(new_data)) {
    new_data <- model$data
    message("Evaluating on training data. For model validation, provide separate test data.")
  }

  is_classification <- model$spec$is_classification
  response_var <- model$spec$response_var

  # Default metrics based on problem type
  if (is.null(metrics)) {
    if (is_classification) {
      metrics <- c("accuracy", "precision", "recall", "f1", "auc")
    } else {
      metrics <- c("rmse", "mae", "rsq", "mape")
    }
  }

  # Get actual values
  actuals <- new_data[[response_var]]

  # Get predictions
  if (is_classification) {
    pred_probs <- predict(model, new_data, type = "prob")
    pred_class <- predict(model, new_data, type = "class")$prediction

    # Calculate metrics
    results <- tl_calc_classification_metrics(
      actuals = actuals,
      predicted = pred_class,
      predicted_probs = pred_probs,
      metrics = metrics,
      ...
    )
  } else {
    predictions <- predict(model, new_data)$prediction

    # Calculate metrics
    results <- tl_calc_regression_metrics(
      actuals = actuals,
      predicted = predictions,
      metrics = metrics,
      ...
    )
  }

  return(results)
}

#' Cross-validate a tidylearn model
#'
#' @param data A data frame containing the training data
#' @param formula A formula specifying the model
#' @param method The modeling method to use
#' @param folds Number of cross-validation folds
#' @param metrics Character vector of metrics to compute
#' @param ... Additional arguments to pass to tl_model
#' @return A tibble with cross-validation results
#' @importFrom rsample vfold_cv
#' @export
tl_cv <- function(data, formula, method = "linear",
                  folds = 5, metrics = NULL, ...) {
  # Create cross-validation splits
  cv_splits <- rsample::vfold_cv(data, v = folds)

  # Extract response variable to determine if classification
  response_var <- all.vars(as.formula(formula))[1]
  y <- data[[response_var]]
  is_classification <- is.factor(y) || is.character(y) || (is.numeric(y) && length(unique(y)) <= 10)

  # Default metrics based on problem type
  if (is.null(metrics)) {
    if (is_classification) {
      metrics <- c("accuracy", "precision", "recall", "f1", "auc")
    } else {
      metrics <- c("rmse", "mae", "rsq", "mape")
    }
  }

  # For each fold, train model and evaluate
  cv_results <- purrr::map_dfr(1:folds, function(i) {
    # Get training and testing data for this fold
    train_data <- rsample::analysis(cv_splits$splits[[i]])
    test_data <- rsample::assessment(cv_splits$splits[[i]])

    # Train model
    model <- tl_model(train_data, formula, method = method, ...)

    # Evaluate model
    fold_metrics <- tl_evaluate(model, test_data, metrics = metrics)

    # Add fold number
    fold_metrics$fold <- i

    return(fold_metrics)
  })

  # Calculate average metrics across folds
  summary_metrics <- cv_results %>%
    dplyr::group_by(.data$metric) %>%
    dplyr::summarize(
      mean_value = mean(.data$value, na.rm = TRUE),
      sd_value = sd(.data$value, na.rm = TRUE),
      min_value = min(.data$value, na.rm = TRUE),
      max_value = max(.data$value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(method = method)

  # Return both detailed and summary results
  return(list(
    fold_metrics = cv_results,
    summary = summary_metrics
  ))
}

#' Plot a tidylearn model
#'
#' @param x A tidylearn model object
#' @param type Type of plot to create
#' @param ... Additional arguments
#' @return A ggplot object
#' @export
plot.tidylearn_model <- function(x, type = "diagnostics", ...) {
  model <- x
  method <- model$spec$method
  is_classification <- model$spec$is_classification

  if (is_classification) {
    # Classification plots
    switch(
      type,
      "roc" = tl_plot_roc(model, ...),
      "confusion" = tl_plot_confusion(model, ...),
      "importance" = tl_plot_importance(model, ...),
      "calibration" = tl_plot_calibration(model, ...),
      "precision_recall" = tl_plot_precision_recall(model, ...),
      stop("Unsupported plot type for classification: ", type, call. = FALSE)
    )
  } else {
    # Regression plots
    switch(
      type,
      "diagnostics" = tl_plot_diagnostics(model, ...),
      "residuals" = tl_plot_residuals(model, ...),
      "actual_vs_predicted" = tl_plot_actual_predicted(model, ...),
      "importance" = tl_plot_importance(model, ...),
      stop("Unsupported plot type for regression: ", type, call. = FALSE)
    )
  }
}

#' Print method for tidylearn models
#'
#' @param x A tidylearn model object
#' @param ... Additional arguments (not used)
#' @return Invisibly returns the model object
#' @export
print.tidylearn_model <- function(x, ...) {
  cat("Tidylearn", x$spec$method, "model\n")
  cat("Formula:", deparse(x$spec$formula), "\n")
  cat("Type:", ifelse(x$spec$is_classification, "Classification", "Regression"), "\n")

  # Print basic evaluation metrics on training data
  cat("\nTraining metrics:\n")
  metrics <- if (x$spec$is_classification) c("accuracy", "f1") else c("rmse", "rsq")
  eval_metrics <- tl_evaluate(x, metrics = metrics)
  print(as.data.frame(eval_metrics), row.names = FALSE)

  invisible(x)
}

#' Summary method for tidylearn models
#'
#' @param object A tidylearn model object
#' @param ... Additional arguments (not used)
#' @return Invisibly returns the model summary
#' @export
summary.tidylearn_model <- function(object, ...) {
  model <- object

  # Extract method-specific summary
  model_summary <- switch(
    model$spec$method,
    "linear" = summary(model$fit),
    "polynomial" = summary(model$fit),
    "logistic" = summary(model$fit),
    "tree" = summary(model$fit),
    "forest" = model$fit,  # Random forests don't have a summary method
    "boost" = model$fit,   # Neither do boosted models
    "ridge" = model$fit,
    "lasso" = model$fit,
    "elastic_net" = model$fit,
    "svm" = model$fit,
    "nn" = model$fit,
    "deep" = model$fit,
    model$fit
  )

  # Print common information
  cat("Tidylearn", model$spec$method, "model\n")
  cat("Formula:", deparse(model$spec$formula), "\n")
  cat("Type:", ifelse(model$spec$is_classification, "Classification", "Regression"), "\n")

  # Print evaluation metrics
  cat("\nEvaluation metrics:\n")
  metrics <- if (model$spec$is_classification) {
    c("accuracy", "precision", "recall", "f1", "auc")
  } else {
    c("rmse", "mae", "rsq", "mape")
  }
  eval_metrics <- tl_evaluate(model, metrics = metrics)
  print(as.data.frame(eval_metrics), row.names = FALSE)

  # Print method-specific summary or information
  cat("\nModel details:\n")
  if (model$spec$method %in% c("linear", "polynomial", "logistic")) {
    print(model_summary$coefficients)
    cat("\nResidual standard error:", format(model_summary$sigma, digits = 4),
        "on", model_summary$df[2], "degrees of freedom\n")
    if (!model$spec$is_classification) {
      cat("Multiple R-squared:", format(model_summary$r.squared, digits = 4),
          ", Adjusted R-squared:", format(model_summary$adj.r.squared, digits = 4), "\n")
    }
  } else if (model$spec$method %in% c("forest", "boost")) {
    if (model$spec$method == "forest") {
      cat("Number of trees:", model$fit$ntree, "\n")
      cat("No. of variables tried at each split:", model$fit$mtry, "\n")
      if (model$spec$is_classification) {
        cat("OOB estimate of error rate:", format(100 * model$fit$err.rate[model$fit$ntree, "OOB"], digits = 3), "%\n")
      } else {
        cat("% Var explained:", format(100 * model$fit$rsq[model$fit$ntree], digits = 3), "%\n")
      }
    } else {
      cat("Number of trees:", length(model$fit$trees), "\n")
      cat("Distribution:", model$fit$distribution$name, "\n")
    }
  } else {
    # For other models, try to extract important information
    cat("Model fitted with", model$spec$method, "method\n")
  }

  invisible(model_summary)
}

# Import helper functions that will be implemented in other files
# This ensures they're available even if files are loaded in the wrong order

#' Helper function to check if required packages are installed
#'
#' @param pkg Character vector of required packages
#' @param error Logical; whether to throw an error if packages are missing
#' @return Logical vector indicating if packages are installed
tl_check_packages <- function(pkg, error = TRUE) {
  is_installed <- purrr::map_lgl(pkg, requireNamespace, quietly = TRUE)

  if (error && any(!is_installed)) {
    missing_pkgs <- pkg[!is_installed]
    stop("The following packages are required but not installed: ",
         paste(missing_pkgs, collapse = ", "),
         ". Please install them with install.packages()", call. = FALSE)
  }

  return(is_installed)
}

# Re-export functions from other tidyverse packages that we'll use frequently
#' @importFrom tibble tibble
#' @export
tibble::tibble

#' @importFrom dplyr filter
#' @export
dplyr::filter

#' @importFrom dplyr select
#' @export
dplyr::select

#' @importFrom dplyr mutate
#' @export
dplyr::mutate

#' @importFrom dplyr group_by
#' @export
dplyr::group_by

#' @importFrom dplyr summarize
#' @export
dplyr::summarize

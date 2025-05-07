#' @title Integration Functions for tidylearn
#' @name tidylearn-integration
#' @description Functions for integrating with other R packages and ecosystems
#' @importFrom stats as.formula
#' @importFrom dplyr %>% select mutate
NULL

#' Convert a tidylearn model to a caret model
#'
#' @param model A tidylearn model object
#' @return A caret model object
#' @export
tl_to_caret <- function(model) {
  # Check if caret is installed
  tl_check_packages("caret")

  # Check input is a tidylearn model
  if (!inherits(model, "tidylearn_model")) {
    stop("Input must be a tidylearn model object", call. = FALSE)
  }

  # Extract components needed for caret model
  data <- model$data
  formula <- model$spec$formula
  method <- model$spec$method

  # Map tidylearn methods to caret methods
  caret_method <- switch(method,
                         "linear" = "lm",
                         "polynomial" = "lm",  # Polynomial handled through formula
                         "logistic" = "glm",
                         "ridge" = "glmnet",
                         "lasso" = "glmnet",
                         "elastic_net" = "glmnet",
                         "tree" = "rpart",
                         "forest" = "rf",
                         "boost" = "gbm",
                         "xgboost" = "xgbTree",
                         "svm" = "svmRadial",
                         "nn" = "nnet",
                         stop("Unsupported method for caret conversion: ", method, call. = FALSE)
  )

  # Extract specific parameters for each method
  if (method == "ridge" || method == "lasso" || method == "elastic_net") {
    # For glmnet, we need to specify alpha
    tuneGrid <- data.frame(
      alpha = if (method == "ridge") 0 else if (method == "lasso") 1 else 0.5,
      lambda = if (hasName(model$fit, "lambda")) model$fit$lambda else 0.1
    )

    # Train caret model
    caret_model <- caret::train(
      form = formula,
      data = data,
      method = caret_method,
      tuneGrid = tuneGrid,
      trControl = caret::trainControl(method = "none")
    )
  } else if (method == "tree") {
    # For rpart, extract complexity parameter
    cp <- model$fit$control$cp

    # Train caret model
    caret_model <- caret::train(
      form = formula,
      data = data,
      method = caret_method,
      tuneGrid = data.frame(cp = cp),
      trControl = caret::trainControl(method = "none")
    )
  } else if (method == "forest") {
    # For random forest, extract mtry
    mtry <- model$fit$mtry

    # Train caret model
    caret_model <- caret::train(
      form = formula,
      data = data,
      method = caret_method,
      tuneGrid = data.frame(mtry = mtry),
      trControl = caret::trainControl(method = "none")
    )
  } else if (method == "xgboost") {
    # For xgboost, extract parameters
    params <- model$fit$params

    # Create tuning grid
    tuneGrid <- data.frame(
      nrounds = model$fit$niter,
      max_depth = if (!is.null(params$max_depth)) params$max_depth else 6,
      eta = if (!is.null(params$eta)) params$eta else 0.3,
      gamma = if (!is.null(params$gamma)) params$gamma else 0,
      colsample_bytree = if (!is.null(params$colsample_bytree)) params$colsample_bytree else 1,
      min_child_weight = if (!is.null(params$min_child_weight)) params$min_child_weight else 1,
      subsample = if (!is.null(params$subsample)) params$subsample else 1
    )

    # Train caret model
    caret_model <- caret::train(
      form = formula,
      data = data,
      method = caret_method,
      tuneGrid = tuneGrid,
      trControl = caret::trainControl(method = "none")
    )
  } else {
    # For other methods, use default approach
    caret_model <- caret::train(
      form = formula,
      data = data,
      method = caret_method,
      trControl = caret::trainControl(method = "none")
    )
  }

  # Copy over any important attributes
  attr(caret_model, "tidylearn_origin") <- list(
    method = method,
    spec = model$spec
  )

  return(caret_model)
}

#' Convert a caret model to a tidylearn model
#'
#' @param model A caret model object
#' @return A tidylearn model object
#' @export
tl_from_caret <- function(model) {
  # Check if caret is installed
  tl_check_packages("caret")

  # Check input is a caret model
  if (!inherits(model, "train")) {
    stop("Input must be a caret train object", call. = FALSE)
  }

  # Extract components needed for tidylearn model
  method <- model$method
  formula <- model$call$form
  data <- model$trainingData

  # If there's no direct formula, try to reconstruct it
  if (is.null(formula)) {
    # Try to get the response variable
    y_name <- model$call$y
    if (!is.null(y_name)) {
      # Get predictors from x
      if (!is.null(model$call$x)) {
        x_names <- colnames(model$trainingData)[!colnames(model$trainingData) %in% ".outcome"]
        formula <- as.formula(paste(y_name, "~", paste(x_names, collapse = " + ")))
      } else {
        # Can't reconstruct formula
        stop("Cannot reconstruct formula from caret model", call. = FALSE)
      }
    } else {
      # Try to get from modelInfo
      if (!is.null(model$modelInfo$label)) {
        stop("Cannot reconstruct formula from caret model with no formula or y", call. = FALSE)
      }
    }
  }

  # Map caret methods to tidylearn methods
  tidylearn_method <- switch(method,
                             "lm" = "linear",
                             "glm" = if (model$modelInfo$parameter$family == "binomial") "logistic" else "linear",
                             "glmnet" = if (!is.null(model$bestTune$alpha)) {
                               if (model$bestTune$alpha == 0) "ridge"
                               else if (model$bestTune$alpha == 1) "lasso"
                               else "elastic_net"
                             } else {
                               "elastic_net"  # Default if alpha is not available
                             },
                             "rpart" = "tree",
                             "rf" = "forest",
                             "gbm" = "boost",
                             "xgbTree" = "xgboost",
                             "svmRadial" = "svm",
                             "nnet" = "nn",
                             stop("Unsupported caret method for tidylearn conversion: ", method, call. = FALSE)
  )

  # Extract fitted model
  fit <- model$finalModel

  # Create tidylearn model
  tidylearn_model <- structure(
    list(
      spec = list(
        formula = formula,
        method = tidylearn_method,
        is_classification = !model$modelInfo$type %in% c("Regression", "Clustering"),
        response_var = as.character(formula[[2]])
      ),
      fit = fit,
      data = model$trainingData
    ),
    class = c(paste0("tidylearn_", tidylearn_method), "tidylearn_model")
  )

  return(tidylearn_model)
}

#' Convert a tidylearn model to a tidymodels workflow
#'
#' @param model A tidylearn model object
#' @return A tidymodels workflow object
#' @export
tl_to_tidymodels <- function(model) {
  # Check if tidymodels packages are installed
  tl_check_packages(c("parsnip", "recipes", "workflows", "dplyr"))

  # Check input is a tidylearn model
  if (!inherits(model, "tidylearn_model")) {
    stop("Input must be a tidylearn model object", call. = FALSE)
  }

  # Extract components needed for tidymodels
  data <- model$data
  formula <- model$spec$formula
  method <- model$spec$method
  is_classification <- model$spec$is_classification
  response_var <- model$spec$response_var

  # Create recipe from formula
  recipe_obj <- recipes::recipe(formula, data = data)

  # Map tidylearn methods to parsnip models
  if (method == "linear" || method == "polynomial") {
    if (is_classification) {
      # For classification, use logistic regression
      parsnip_model <- parsnip::logistic_reg() %>%
        parsnip::set_engine("glm")
    } else {
      # For regression, use linear regression
      parsnip_model <- parsnip::linear_reg() %>%
        parsnip::set_engine("lm")
    }
  } else if (method == "logistic") {
    parsnip_model <- parsnip::logistic_reg() %>%
      parsnip::set_engine("glm")
  } else if (method %in% c("ridge", "lasso", "elastic_net")) {
    # Set alpha based on method
    alpha_val <- switch(method,
                        "ridge" = 0,
                        "lasso" = 1,
                        "elastic_net" = 0.5
    )

    if (is_classification) {
      parsnip_model <- parsnip::logistic_reg(penalty = 0.1, mixture = alpha_val) %>%
        parsnip::set_engine("glmnet")
    } else {
      parsnip_model <- parsnip::linear_reg(penalty = 0.1, mixture = alpha_val) %>%
        parsnip::set_engine("glmnet")
    }
  } else if (method == "tree") {
    if (is_classification) {
      parsnip_model <- parsnip::decision_tree() %>%
        parsnip::set_engine("rpart")
    } else {
      parsnip_model <- parsnip::decision_tree() %>%
        parsnip::set_engine("rpart")
    }
  } else if (method == "forest") {
    if (is_classification) {
      parsnip_model <- parsnip::rand_forest() %>%
        parsnip::set_engine("randomForest")
    } else {
      parsnip_model <- parsnip::rand_forest() %>%
        parsnip::set_engine("randomForest")
    }
  } else if (method == "boost" || method == "xgboost") {
    if (is_classification) {
      parsnip_model <- parsnip::boost_tree() %>%
        parsnip::set_engine(ifelse(method == "xgboost", "xgboost", "gbm"))
    } else {
      parsnip_model <- parsnip::boost_tree() %>%
        parsnip::set_engine(ifelse(method == "xgboost", "xgboost", "gbm"))
    }
  } else if (method == "svm") {
    if (is_classification) {
      parsnip_model <- parsnip::svm_rbf() %>%
        parsnip::set_engine("kernlab")
    } else {
      parsnip_model <- parsnip::svm_rbf() %>%
        parsnip::set_engine("kernlab")
    }
  } else if (method == "nn") {
    if (is_classification) {
      parsnip_model <- parsnip::mlp() %>%
        parsnip::set_engine("nnet")
    } else {
      parsnip_model <- parsnip::mlp() %>%
        parsnip::set_engine("nnet")
    }
  } else {
    stop("Unsupported method for tidymodels conversion: ", method, call. = FALSE)
  }

  # Create workflow
  workflow_obj <- workflows::workflow() %>%
    workflows::add_recipe(recipe_obj) %>%
    workflows::add_model(parsnip_model)

  # Try to fit the model with the same data
  fitted_workflow <- tryCatch({
    workflows::fit(workflow_obj, data = data)
  }, error = function(e) {
    warning("Could not automatically fit the tidymodels workflow: ", e$message,
            "\nReturning unfitted workflow instead.")
    return(workflow_obj)
  })

  # Add attribute to track original tidylearn model
  attr(fitted_workflow, "tidylearn_origin") <- list(
    method = method,
    spec = model$spec
  )

  return(fitted_workflow)
}

#' Convert a tidymodels workflow to a tidylearn model
#'
#' @param workflow A tidymodels workflow object
#' @return A tidylearn model object
#' @export
tl_from_tidymodels <- function(workflow) {
  # Check if tidymodels packages are installed
  tl_check_packages(c("parsnip", "recipes", "workflows"))

  # Check input is a tidymodels workflow
  if (!inherits(workflow, "workflow")) {
    stop("Input must be a tidymodels workflow object", call. = FALSE)
  }

  # Check if workflow is fitted
  if (!workflows::is_fitted(workflow)) {
    stop("Workflow must be fitted before conversion to tidylearn model", call. = FALSE)
  }

  # Extract model specification
  model_spec <- workflows::extract_spec_parsnip(workflow)

  # Extract fitted model
  fit <- workflows::extract_fit_parsnip(workflow)$fit

  # Extract model data
  if (!is.null(fit$call$data)) {
    data <- eval(fit$call$data)
  } else {
    # Try to extract from other places
    data <- tryCatch({
      # For some models, data is stored differently
      if (inherits(fit, "randomForest")) {
        # For random forest, try to reconstruct from model
        x <- fit$x
        y <- fit$y
        data <- data.frame(x, y = y)
      } else if (inherits(fit, "glmnet")) {
        # For glmnet, this gets complex
        warning("Cannot extract original data from glmnet model", call. = FALSE)
        data <- NULL
      } else {
        # Default approach
        data <- fit$data
      }
    }, error = function(e) {
      warning("Could not extract original data from model: ", e$message)
      return(NULL)
    })
  }

  # Extract formula
  formula <- NULL
  if (!is.null(workflow$pre$actions$recipe$recipe)) {
    # Get formula from recipe
    recipe_obj <- workflow$pre$actions$recipe$recipe
    formula <- recipe_obj$formula
  } else if (!is.null(fit$call$formula)) {
    # Get formula from model call
    formula <- fit$call$formula
  } else if (!is.null(fit$terms)) {
    # Try to reconstruct formula from terms
    formula <- stats::formula(fit$terms)
  } else {
    # Could not extract formula
    stop("Could not extract formula from workflow", call. = FALSE)
  }

  # Determine model type
  engine <- model_spec$engine

  # Map tidymodels models to tidylearn methods
  if (inherits(model_spec, "linear_reg")) {
    tidylearn_method <- "linear"
  } else if (inherits(model_spec, "logistic_reg")) {
    tidylearn_method <- "logistic"
  } else if (inherits(model_spec, "multinom_reg")) {
    tidylearn_method <- "logistic"  # Multinomial logistic regression
  } else if (inherits(model_spec, "decision_tree")) {
    tidylearn_method <- "tree"
  } else if (inherits(model_spec, "rand_forest")) {
    tidylearn_method <- "forest"
  } else if (inherits(model_spec, "boost_tree")) {
    tidylearn_method <- if (engine == "xgboost") "xgboost" else "boost"
  } else if (inherits(model_spec, "svm_rbf") || inherits(model_spec, "svm_poly") ||
             inherits(model_spec, "svm_linear")) {
    tidylearn_method <- "svm"
  } else if (inherits(model_spec, "mlp")) {
    tidylearn_method <- "nn"
  } else {
    stop("Unsupported tidymodels model type for conversion", call. = FALSE)
  }

  # Determine if classification or regression
  is_classification <- model_spec$mode == "classification"

  # Get response variable
  response_var <- all.vars(formula)[1]

  # Create tidylearn model
  tidylearn_model <- structure(
    list(
      spec = list(
        formula = formula,
        method = tidylearn_method,
        is_classification = is_classification,
        response_var = response_var
      ),
      fit = fit,
      data = data
    ),
    class = c(paste0("tidylearn_", tidylearn_method), "tidylearn_model")
  )

  return(tidylearn_model)
}

#' Export a tidylearn model to an external format
#'
#' @param model A tidylearn model object
#' @param format Export format: "rds", "onnx", "pmml", "json"
#' @param file Path to save the exported model (if NULL, returns the model object)
#' @param ... Additional arguments for the specific export format
#' @return The exported model or NULL if saved to file
#' @export
tl_export_model <- function(model, format = "rds", file = NULL, ...) {
  # Check input is a tidylearn model
  if (!inherits(model, "tidylearn_model")) {
    stop("Input must be a tidylearn model object", call. = FALSE)
  }

  # Export based on format
  if (format == "rds") {
    # Save as R object
    if (is.null(file)) {
      return(model)  # Return the model as is
    } else {
      saveRDS(model, file = file)
      return(invisible(NULL))
    }
  } else if (format == "onnx") {
    # Check if reticulate and onnx packages are installed
    tl_check_packages(c("reticulate", "onnx"))

    # Convert model to ONNX format
    if (model$spec$method %in% c("linear", "logistic", "ridge", "lasso", "elastic_net")) {
      # For linear models
      coefficients <- stats::coef(model$fit)

      # Create ONNX model using Python
      reticulate::source_python(system.file("python", "linear_to_onnx.py", package = "tidylearn"))
      onnx_model <- linear_model_to_onnx(
        coefficients = coefficients,
        is_classification = model$spec$is_classification,
        feature_names = names(coefficients)[-1]  # Exclude intercept
      )
    } else if (model$spec$method == "forest") {
      # For random forest models
      reticulate::source_python(system.file("python", "rf_to_onnx.py", package = "tidylearn"))
      onnx_model <- random_forest_to_onnx(model$fit)
    } else if (model$spec$method == "xgboost") {
      # For XGBoost models
      reticulate::source_python(system.file("python", "xgboost_to_onnx.py", package = "tidylearn"))
      onnx_model <- xgboost_to_onnx(model$fit)
    } else {
      stop("ONNX export not supported for model type: ", model$spec$method, call. = FALSE)
    }

    # Save or return the ONNX model
    if (is.null(file)) {
      return(onnx_model)
    } else {
      onnx::write_onnx(onnx_model, file)
      return(invisible(NULL))
    }
  } else if (format == "pmml") {
    # Check if pmml package is installed
    tl_check_packages("pmml")

    # Convert model to PMML format
    if (model$spec$method == "linear") {
      pmml_model <- pmml::pmml(model$fit)
    } else if (model$spec$method == "logistic") {
      pmml_model <- pmml::pmml(model$fit)
    } else if (model$spec$method == "tree") {
      pmml_model <- pmml::pmml(model$fit)
    } else if (model$spec$method == "forest") {
      pmml_model <- pmml::pmml(model$fit)
    } else {
      stop("PMML export not supported for model type: ", model$spec$method, call. = FALSE)
    }

    # Save or return the PMML model
    if (is.null(file)) {
      return(pmml_model)
    } else {
      saveXML(pmml_model, file)
      return(invisible(NULL))
    }
  } else if (format == "json") {
    # Check if jsonlite package is installed
    tl_check_packages("jsonlite")

    # Create a JSON representation of the model
    if (model$spec$method %in% c("linear", "logistic")) {
      # For linear/logistic models, export coefficients
      model_json <- list(
        method = model$spec$method,
        is_classification = model$spec$is_classification,
        formula = as.character(model$spec$formula),
        coefficients = as.list(stats::coef(model$fit)),
        response_var = model$spec$response_var
      )
    } else if (model$spec$method %in% c("ridge", "lasso", "elastic_net")) {
      # For regularized models, export coefficients at best lambda
      lambda <- attr(model$fit, "lambda_min")
      coef_matrix <- as.matrix(coef(model$fit, s = lambda))

      model_json <- list(
        method = model$spec$method,
        is_classification = model$spec$is_classification,
        formula = as.character(model$spec$formula),
        coefficients = as.list(coef_matrix[, 1]),
        lambda = lambda,
        response_var = model$spec$response_var
      )
    } else {
      # For other model types, create a simplified representation
      model_json <- list(
        method = model$spec$method,
        is_classification = model$spec$is_classification,
        formula = as.character(model$spec$formula),
        response_var = model$spec$response_var,
        model_details = "Complex model structure - not fully serializable to JSON"
      )
    }

    # Convert to JSON
    json_str <- jsonlite::toJSON(model_json, pretty = TRUE)

    # Save or return the JSON
    if (is.null(file)) {
      return(json_str)
    } else {
      writeLines(json_str, file)
      return(invisible(NULL))
    }
  } else {
    stop("Unsupported export format: ", format, call. = FALSE)
  }
}

#' Import a tidylearn model from an external format
#'
#' @param file Path to the model file
#' @param format Import format: "rds", "onnx", "pmml", "json"
#' @param ... Additional arguments for the specific import format
#' @return A tidylearn model object
#' @export
tl_import_model <- function(file, format = "rds", ...) {
  # Import based on format
  if (format == "rds") {
    # Load from R object
    model <- readRDS(file)

    # Check if it's a tidylearn model
    if (!inherits(model, "tidylearn_model")) {
      stop("Imported object is not a tidylearn model", call. = FALSE)
    }

    return(model)
  } else if (format == "onnx") {
    # Check if reticulate and onnx packages are installed
    tl_check_packages(c("reticulate", "onnx"))

    # Load ONNX model
    onnx_model <- onnx::read_onnx(file)

    # Convert to a tidylearn model (simplified)
    # This is a limited conversion as ONNX models don't contain all the original information

    # Detect model type and create a basic representation
    model_type <- "unknown"
    if (length(onnx_model$graph$node) == 1) {
      # Likely a linear model
      if (onnx_model$graph$node[[1]]$op_type == "LinearRegressor") {
        model_type <- "linear"
      } else if (onnx_model$graph$node[[1]]$op_type == "LinearClassifier") {
        model_type <- "logistic"
      }
    } else {
      # Try to determine from sequence of operations
      op_types <- sapply(onnx_model$graph$node, function(node) node$op_type)
      if (any(grepl("Tree", op_types))) {
        model_type <- "tree"
      } else if (any(grepl("Forest", op_types))) {
        model_type <- "forest"
      }
    }

    # Create a basic tidylearn model
    model <- structure(
      list(
        spec = list(
          method = model_type,
          is_classification = any(grepl("Classifier", sapply(onnx_model$graph$node, function(n) n$op_type))),
          formula = as.formula("y ~ ."),  # Placeholder
          response_var = "y"  # Placeholder
        ),
        fit = onnx_model,  # Store the ONNX model directly
        data = NULL  # No data available
      ),
      class = c(paste0("tidylearn_", model_type), "tidylearn_model", "tidylearn_onnx")
    )

    # Add custom predict method for ONNX models
    class(model) <- c("tidylearn_onnx", class(model))

    return(model)
  } else if (format == "pmml") {
    # Check if pmml package is installed
    tl_check_packages("pmml")

    # Load PMML model
    pmml_model <- XML::xmlParse(file)

    # Detect model type from PMML
    model_type <- "unknown"

    # Check for different model types in the PMML
    model_tags <- XML::xpathSApply(pmml_model, "//pmml:*", XML::xmlName)

    if ("RegressionModel" %in% model_tags) {
      model_type <- "linear"
    } else if ("TreeModel" %in% model_tags) {
      model_type <- "tree"
    } else if ("RandomForestModel" %in% model_tags) {
      model_type <- "forest"
    }

    # Create a basic tidylearn model
    model <- structure(
      list(
        spec = list(
          method = model_type,
          is_classification = XML::xpathSApply(
            pmml_model, "//pmml:MiningFunction", XML::xmlValue
          ) == "classification",
          formula = as.formula("y ~ ."),  # Placeholder
          response_var = "y"  # Placeholder
        ),
        fit = pmml_model,  # Store the PMML model directly
        data = NULL  # No data available
      ),
      class = c(paste0("tidylearn_", model_type), "tidylearn_model", "tidylearn_pmml")
    )

    # Add custom predict method for PMML models
    class(model) <- c("tidylearn_pmml", class(model))

    return(model)
  } else if (format == "json") {
    # Check if jsonlite package is installed
    tl_check_packages("jsonlite")

    # Load JSON model
    json_str <- readLines(file, warn = FALSE)
    model_json <- jsonlite::fromJSON(json_str)

    # Extract model information
    method <- model_json$method
    is_classification <- model_json$is_classification
    formula <- as.formula(model_json$formula)
    response_var <- model_json$response_var

    # Create a basic model representation
    if (method %in% c("linear", "logistic")) {
      # Create a basic lm/glm model from coefficients
      coefficients <- unlist(model_json$coefficients)

      # Create model matrix
      x_names <- names(coefficients)[-1]  # Exclude intercept
      x <- matrix(NA, nrow = 1, ncol = length(x_names))
      colnames(x) <- x_names

      if (method == "linear") {
        # Create a minimal lm object
        fit <- list(
          coefficients = coefficients,
          model = NULL,
          call = substitute(lm(formula = formula)),
          terms = terms(formula)
        )
        class(fit) <- "lm"
      } else {
        # Create a minimal glm object
        fit <- list(
          coefficients = coefficients,
          model = NULL,
          call = substitute(glm(formula = formula, family = binomial())),
          terms = terms(formula),
          family = binomial()
        )
        class(fit) <- c("glm", "lm")
      }
    } else {
      # For other model types, create a placeholder
      fit <- list(
        model_json = model_json
      )
      class(fit) <- "json_model"
    }

    # Create a tidylearn model
    model <- structure(
      list(
        spec = list(
          method = method,
          is_classification = is_classification,
          formula = formula,
          response_var = response_var
        ),
        fit = fit,
        data = NULL  # No data available
      ),
      class = c(paste0("tidylearn_", method), "tidylearn_model", "tidylearn_json")
    )

    # Add custom predict method for JSON models
    class(model) <- c("tidylearn_json", class(model))

    return(model)
  } else {
    stop("Unsupported import format: ", format, call. = FALSE)
  }
}

#' Predict method for ONNX models
#'
#' @param object A tidylearn_onnx model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction
#' @param ... Additional arguments
#' @return Predictions
#' @export
predict.tidylearn_onnx <- function(object, new_data, type = "response", ...) {
  # Check if reticulate and onnx packages are installed
  tl_check_packages(c("reticulate", "onnx", "onnxruntime"))

  # Get ONNX model
  onnx_model <- object$fit

  # Create input array from new_data
  # This is simplified and would need to be adapted based on the actual model
  x <- as.matrix(new_data[, setdiff(names(new_data), object$spec$response_var)])

  # Run inference using onnxruntime
  session <- onnxruntime::onnxruntime_session(onnx_model)
  preds <- onnxruntime::predict(session, x)

  # Format predictions based on type and model classification status
  if (object$spec$is_classification) {
    if (type == "prob") {
      # Return probabilities
      return(preds)
    } else {
      # Return class predictions
      return(factor(max.col(preds)))
    }
  } else {
    # Regression predictions
    return(preds[, 1])
  }
}

#' Predict method for PMML models
#'
#' @param object A tidylearn_pmml model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction
#' @param ... Additional arguments
#' @return Predictions
#' @export
predict.tidylearn_pmml <- function(object, new_data, type = "response", ...) {
  # Check if pmml package is installed
  tl_check_packages("pmml")

  # For PMML, we need a dedicated execution engine
  if (!requireNamespace("pmmlTransformations", quietly = TRUE)) {
    stop("Package 'pmmlTransformations' is required for PMML prediction", call. = FALSE)
  }

  # Get PMML model
  pmml_model <- object$fit

  # Predict using pmml
  preds <- pmmlTransformations::predict_pmml(pmml_model, newdata = new_data)

  # Format predictions based on type and model classification status
  if (object$spec$is_classification) {
    if (type == "prob") {
      # Return probabilities if available
      if ("probability" %in% names(preds)) {
        return(preds$probability)
      } else {
        stop("Probability predictions not available for this PMML model", call. = FALSE)
      }
    } else {
      # Return class predictions
      return(preds$prediction)
    }
  } else {
    # Regression predictions
    return(preds$prediction)
  }
}

#' Predict method for JSON models
#'
#' @param object A tidylearn_json model object
#' @param new_data A data frame containing the new data
#' @param type Type of prediction
#' @param ... Additional arguments
#' @return Predictions
#' @export
predict.tidylearn_json <- function(object, new_data, type = "response", ...) {
  # Get model information
  method <- object$spec$method
  is_classification <- object$spec$is_classification
  fit <- object$fit

  # For linear/logistic models, we can make predictions
  if (method %in% c("linear", "logistic")) {
    # Extract model matrix
    x <- stats::model.matrix(object$spec$formula, new_data)[, -1, drop = FALSE]  # Remove intercept

    # Get coefficients
    coef <- fit$coefficients

    # Make predictions
    linear_pred <- coef[1] + rowSums(t(t(x) * coef[-1]))

    if (method == "logistic") {
      # Apply logistic function
      prob <- 1 / (1 + exp(-linear_pred))

      if (type == "prob") {
        # Return probabilities
        return(data.frame(
          class0 = 1 - prob,
          class1 = prob
        ))
      } else {
        # Return class predictions
        return(factor(ifelse(prob > 0.5, 1, 0)))
      }
    } else {
      # Regression predictions
      return(linear_pred)
    }
  } else {
    # For other model types, we can't make predictions without the full model
    stop("Prediction not supported for JSON models of type: ", method, call. = FALSE)
  }
}

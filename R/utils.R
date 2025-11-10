#' @title Utility Functions for tidysl
#' @name tidysl-utils
#' @description General utility functions used across the tidysl package
#' @importFrom stats median quantile sd var cor aov coef fitted na.omit qqnorm reorder residuals runif terms update
#' @importFrom utils packageVersion installed.packages combn hasName
#' @importFrom tidyselect all_of
NULL

# Declare global variables used in NSE (non-standard evaluation)
utils::globalVariables(c(
  # rlang/tidyverse operators
  ":=",
  # Python functions loaded via reticulate (from inst/python/*.py)
  "linear_model_to_onnx", "random_forest_to_onnx", "xgboost_to_onnx",
  # ggplot2/dplyr NSE variables
  "Actual", "Assumption", "Details", "Freq", "Predicted", "Status",
  "abs_shap_value", "actual", "coefficient", "conf_lower", "conf_upper",
  "cooks_distance", "cost", "decay", "decile", "epoch", "error",
  "error_lower", "error_upper",
  "feature", "feature_value", "fitted", "fold", "fpr", "frac_pos",
  "interaction_value", "is_best", "is_cook_influential", "is_influential",
  "is_outlier", "is_top", "label", "lambda", "leverage", "mean_pred_prob",
  "mean_value", "metric", "model", "n", "observation", "percentage",
  "pred", "pred_lower", "pred_upper", "predicted", "residuals", "shap_value",
  "size", "sqrt_abs_residuals", "std_residual", "tpr", "value", "var_value",
  "variable", "x", "y"
))

#' Check if required packages are installed
#'
#' @param pkg Character vector of required packages
#' @param error Logical; whether to throw an error if packages are missing
#' @return Logical vector indicating if packages are installed
#' @export
tl_check_packages <- function(pkg, error = TRUE) {
  is_installed <- sapply(pkg, requireNamespace, quietly = TRUE)

  if (error && any(!is_installed)) {
    missing_pkgs <- pkg[!is_installed]
    stop("The following packages are required but not installed: ",
         paste(missing_pkgs, collapse = ", "),
         ". Please install them with install.packages()", call. = FALSE)
  }

  return(is_installed)
}

#' Get tidysl package version
#'
#' @return Character string with the version number
#' @export
tl_version <- function() {
  version <- as.character(utils::packageVersion("tidysl"))
  return(version)
}

#' Check if a variable has enough variance to be useful
#'
#' @param x Vector to check for variance
#' @param min_variance Minimum variance threshold (for numeric variables)
#' @param min_categories Minimum number of categories (for categorical variables)
#' @param min_frequency Minimum frequency per category (for categorical variables)
#' @return Logical indicating if variable has enough variance
#' @export
tl_has_variance <- function(x, min_variance = 1e-8, min_categories = 2, min_frequency = 1) {
  if (is.numeric(x)) {
    # For numeric variables, check variance
    return(stats::var(x, na.rm = TRUE) > min_variance)
  } else if (is.factor(x) || is.character(x)) {
    # For categorical variables, check number of unique values
    tab <- table(x)
    return(length(tab) >= min_categories && all(tab >= min_frequency))
  } else {
    # Other types, just check if there are at least two distinct values
    return(length(unique(x)) >= 2)
  }
}

#' Check for multicollinearity in data
#'
#' @param data A data frame containing predictors
#' @param threshold Correlation threshold to flag
#' @param variables Optional character vector of variables to check (if NULL, checks all numeric columns)
#' @return A data frame with highly correlated variable pairs
#' @export
tl_check_multicollinearity <- function(data, threshold = 0.8, variables = NULL) {
  # Select variables
  if (is.null(variables)) {
    variables <- names(data)[sapply(data, is.numeric)]
  } else {
    # Check if variables exist
    missing_vars <- setdiff(variables, names(data))
    if (length(missing_vars) > 0) {
      stop("Variables not found in data: ", paste(missing_vars, collapse = ", "), call. = FALSE)
    }

    # Check if variables are numeric
    non_numeric <- variables[!sapply(data[variables], is.numeric)]
    if (length(non_numeric) > 0) {
      stop("Non-numeric variables: ", paste(non_numeric, collapse = ", "), call. = FALSE)
    }
  }

  # Need at least 2 variables to check multicollinearity
  if (length(variables) < 2) {
    message("Need at least 2 variables to check multicollinearity")
    return(data.frame(var1 = character(), var2 = character(), correlation = numeric()))
  }

  # Calculate correlation matrix
  cor_matrix <- stats::cor(data[variables], use = "pairwise.complete.obs")

  # Identify highly correlated pairs
  # Pull out the upper triangle of the correlation matrix
  cor_upper <- cor_matrix
  cor_upper[lower.tri(cor_upper, diag = TRUE)] <- NA

  # Convert to data frame
  cor_df <- reshape2::melt(cor_upper, na.rm = TRUE)
  names(cor_df) <- c("var1", "var2", "correlation")

  # Filter by threshold
  high_cor <- cor_df[abs(cor_df$correlation) > threshold, ]

  # Sort by absolute correlation (descending)
  high_cor <- high_cor[order(abs(high_cor$correlation), decreasing = TRUE), ]

  return(high_cor)
}

#' Scale variables to a specified range
#'
#' @param x Vector to scale
#' @param lower Lower bound of range
#' @param upper Upper bound of range
#' @param na.rm Logical; whether to remove NA values
#' @return Scaled vector
#' @export
tl_scale_to_range <- function(x, lower = 0, upper = 1, na.rm = TRUE) {
  if (!is.numeric(x)) {
    stop("Input must be numeric", call. = FALSE)
  }

  x_min <- min(x, na.rm = na.rm)
  x_max <- max(x, na.rm = na.rm)

  # Check if all values are identical
  if (x_min == x_max) {
    warning("All values are identical, returning constant value", call. = FALSE)
    return(rep((lower + upper) / 2, length(x)))
  }

  # Perform scaling
  scaled <- lower + (upper - lower) * (x - x_min) / (x_max - x_min)

  return(scaled)
}

#' Detect and handle outliers in data
#'
#' @param x Vector to check for outliers
#' @param method Method for outlier detection: "iqr", "z-score", "percentile"
#' @param threshold Threshold for outlier detection
#' @param replace_with How to replace outliers: "NA", "median", "mean", "winsorize", "trim"
#' @param lower_only Logical; whether to only handle lower outliers
#' @param upper_only Logical; whether to only handle upper outliers
#' @param na.rm Logical; whether to remove NA values
#' @return Vector with handled outliers
#' @export
tl_handle_outliers <- function(x, method = "iqr", threshold = 1.5,
                               replace_with = "NA", lower_only = FALSE,
                               upper_only = FALSE, na.rm = TRUE) {
  if (!is.numeric(x)) {
    stop("Input must be numeric", call. = FALSE)
  }

  # Detect outliers
  outliers <- switch(
    method,
    "iqr" = {
      q1 <- stats::quantile(x, 0.25, na.rm = na.rm)
      q3 <- stats::quantile(x, 0.75, na.rm = na.rm)
      iqr <- q3 - q1
      lower_bound <- q1 - threshold * iqr
      upper_bound <- q3 + threshold * iqr

      if (lower_only) {
        x < lower_bound
      } else if (upper_only) {
        x > upper_bound
      } else {
        x < lower_bound | x > upper_bound
      }
    },
    "z-score" = {
      z_scores <- abs(scale(x, center = mean(x, na.rm = na.rm),
                            scale = stats::sd(x, na.rm = na.rm)))

      if (lower_only) {
        scale(x) < -threshold
      } else if (upper_only) {
        scale(x) > threshold
      } else {
        z_scores > threshold
      }
    },
    "percentile" = {
      lower_percentile <- threshold / 2
      upper_percentile <- 1 - lower_percentile

      lower_bound <- stats::quantile(x, lower_percentile, na.rm = na.rm)
      upper_bound <- stats::quantile(x, upper_percentile, na.rm = na.rm)

      if (lower_only) {
        x < lower_bound
      } else if (upper_only) {
        x > upper_bound
      } else {
        x < lower_bound | x > upper_bound
      }
    },
    stop("Invalid method. Use 'iqr', 'z-score', or 'percentile'.", call. = FALSE)
  )

  # Handle outliers
  if (any(outliers, na.rm = TRUE)) {
    result <- x

    switch(
      replace_with,
      "NA" = {
        result[outliers] <- NA
      },
      "median" = {
        result[outliers] <- stats::median(x[!outliers], na.rm = na.rm)
      },
      "mean" = {
        result[outliers] <- mean(x[!outliers], na.rm = na.rm)
      },
      "winsorize" = {
        if (method == "iqr") {
          q1 <- stats::quantile(x, 0.25, na.rm = na.rm)
          q3 <- stats::quantile(x, 0.75, na.rm = na.rm)
          iqr <- q3 - q1
          lower_bound <- q1 - threshold * iqr
          upper_bound <- q3 + threshold * iqr
        } else if (method == "z-score") {
          mu <- mean(x, na.rm = na.rm)
          sigma <- stats::sd(x, na.rm = na.rm)
          lower_bound <- mu - threshold * sigma
          upper_bound <- mu + threshold * sigma
        } else {  # percentile
          lower_percentile <- threshold / 2
          upper_percentile <- 1 - lower_percentile
          lower_bound <- stats::quantile(x, lower_percentile, na.rm = na.rm)
          upper_bound <- stats::quantile(x, upper_percentile, na.rm = na.rm)
        }

        if (!lower_only) {
          result[x > upper_bound] <- upper_bound
        }
        if (!upper_only) {
          result[x < lower_bound] <- lower_bound
        }
      },
      "trim" = {
        result <- x
        result[outliers] <- NA
      },
      stop("Invalid replacement method. Use 'NA', 'median', 'mean', 'winsorize', or 'trim'.", call. = FALSE)
    )

    return(result)
  } else {
    return(x)  # No outliers detected
  }
}

#' Compute variable importance for feature selection
#'
#' @param data A data frame containing the data
#' @param response_var Name of the response variable
#' @param method Method for importance calculation: "correlation", "relief", "information_gain"
#' @param top_n Number of top variables to return (if NULL, returns all)
#' @param ... Additional arguments passed to specific importance methods
#' @return A data frame with variable importance scores
#' @export
tl_variable_importance <- function(data, response_var, method = "correlation",
                                   top_n = NULL, ...) {
  # Check if response variable exists
  if (!response_var %in% names(data)) {
    stop("Response variable not found in data", call. = FALSE)
  }

  # Get predictor variables
  predictors <- setdiff(names(data), response_var)

  # Calculate importance based on method
  if (method == "correlation") {
    # Correlation-based feature selection
    if (!is.numeric(data[[response_var]])) {
      stop("Correlation method requires a numeric response variable", call. = FALSE)
    }

    # Calculate correlation with response for numeric predictors
    num_predictors <- predictors[sapply(data[predictors], is.numeric)]

    if (length(num_predictors) == 0) {
      stop("No numeric predictor variables found", call. = FALSE)
    }

    importance <- sapply(num_predictors, function(var) {
      abs(stats::cor(data[[var]], data[[response_var]], use = "pairwise.complete.obs"))
    })

    importance_df <- data.frame(
      variable = num_predictors,
      importance = importance
    )
  } else if (method == "relief") {
    # ReliefF algorithm
    tl_check_packages("CORElearn")

    # Prepare data
    data_subset <- data[, c(response_var, predictors)]

    # Remove rows with NA values
    data_subset <- na.omit(data_subset)

    # Run ReliefF algorithm
    attrEval <- CORElearn::attrEval(
      formula = as.formula(paste(response_var, "~ .")),
      data = data_subset,
      estimator = "ReliefFequalK",
      ...
    )

    importance_df <- data.frame(
      variable = names(attrEval),
      importance = as.numeric(attrEval)
    )
  } else if (method == "information_gain") {
    # Information gain
    tl_check_packages("FSelector")

    # Prepare data
    data_subset <- data[, c(response_var, predictors)]

    # Remove rows with NA values
    data_subset <- na.omit(data_subset)

    # Calculate information gain
    weights <- FSelector::information.gain(
      formula = as.formula(paste(response_var, "~ .")),
      data = data_subset
    )

    importance_df <- data.frame(
      variable = rownames(weights),
      importance = weights[, 1]
    )
  } else {
    stop("Invalid method. Use 'correlation', 'relief', or 'information_gain'.", call. = FALSE)
  }

  # Sort by importance (descending)
  importance_df <- importance_df[order(importance_df$importance, decreasing = TRUE), ]

  # Return top N if specified
  if (!is.null(top_n)) {
    importance_df <- importance_df[1:min(top_n, nrow(importance_df)), ]
  }

  return(importance_df)
}

#' Convert a numerical variable to bins
#'
#' @param x Vector to bin
#' @param n_bins Number of bins (default: 10)
#' @param method Method for binning: "equal_width", "equal_freq", "custom"
#' @param breaks Custom breaks (only used if method = "custom")
#' @param labels Custom labels (if NULL, defaults to range labels)
#' @param right Logical; whether the intervals should be closed on the right
#' @return A factor with the binned values
#' @export
tl_bin_variable <- function(x, n_bins = 10, method = "equal_width", breaks = NULL,
                            labels = NULL, right = TRUE) {
  if (!is.numeric(x)) {
    stop("Input must be numeric", call. = FALSE)
  }

  if (method == "custom" && is.null(breaks)) {
    stop("Custom breaks must be provided when method = 'custom'", call. = FALSE)
  }

  # Generate breaks based on method
  if (method == "equal_width") {
    breaks <- seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = n_bins + 1)
  } else if (method == "equal_freq") {
    breaks <- stats::quantile(x, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
  } # else use provided custom breaks

  # Create bin labels if not provided
  if (is.null(labels)) {
    labels <- paste0("[", signif(breaks[-length(breaks)], 3), ", ",
                     signif(breaks[-1], 3), ifelse(right, "]", ")"))
  }

  # Perform binning
  bins <- cut(x, breaks = breaks, labels = labels, include.lowest = TRUE, right = right)

  return(bins)
}

#' Create a data partition for training and testing
#'
#' @param data A data frame containing the data
#' @param prop Proportion of data to use for training
#' @param strata Optional variable name to use for stratified sampling
#' @param seed Random seed for reproducibility
#' @return A list with training and testing indices
#' @export
tl_partition_data <- function(data, prop = 0.75, strata = NULL, seed = NULL) {
  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Get number of observations
  n <- nrow(data)

  # Determine number of training samples
  train_size <- floor(prop * n)

  if (is.null(strata)) {
    # Simple random sampling
    train_indices <- sample(1:n, train_size)
  } else {
    # Stratified sampling
    if (!strata %in% names(data)) {
      stop("Stratification variable not found in data", call. = FALSE)
    }

    # Handle different variable types
    if (is.factor(data[[strata]]) || is.character(data[[strata]])) {
      # Categorical stratification
      strata_values <- as.factor(data[[strata]])
    } else if (is.numeric(data[[strata]])) {
      # For numeric, bin into quartiles
      strata_values <- cut(data[[strata]], breaks = 4)
    } else {
      stop("Unsupported data type for stratification", call. = FALSE)
    }

    # Perform stratified sampling
    levels <- unique(strata_values)
    train_indices <- numeric(0)

    for (lvl in levels) {
      lvl_indices <- which(strata_values == lvl)
      lvl_size <- length(lvl_indices)
      lvl_train_size <- floor(prop * lvl_size)

      if (lvl_train_size > 0) {
        lvl_train_indices <- sample(lvl_indices, lvl_train_size)
        train_indices <- c(train_indices, lvl_train_indices)
      }
    }
  }

  # Get test indices
  test_indices <- setdiff(1:n, train_indices)

  # Return indices
  return(list(
    train = train_indices,
    test = test_indices
  ))
}

#' Format numbers with standardized precision
#'
#' @param x Vector of numbers to format
#' @param digits Number of digits to round to
#' @param format Format to use: "standard", "scientific", "percent"
#' @param zero_pad Logical; whether to pad with zeros
#' @return Character vector with formatted numbers
#' @export
tl_format_numbers <- function(x, digits = 3, format = "standard", zero_pad = FALSE) {
  if (!is.numeric(x)) {
    stop("Input must be numeric", call. = FALSE)
  }

  if (format == "standard") {
    if (zero_pad) {
      result <- formatC(x, digits = digits, format = "f", flag = "0")
    } else {
      result <- formatC(x, digits = digits, format = "f")
    }
  } else if (format == "scientific") {
    result <- formatC(x, digits = digits, format = "e")
  } else if (format == "percent") {
    result <- paste0(formatC(x * 100, digits = digits, format = "f"), "%")
  } else {
    stop("Invalid format. Use 'standard', 'scientific', or 'percent'.", call. = FALSE)
  }

  return(result)
}

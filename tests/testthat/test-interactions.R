context("Interactions functionality tests")

# Sample data for testing
data(mtcars)

test_that("tl_test_interactions tests for significant interactions", {
  # Test for specific pair
  test_results <- tl_test_interactions(mtcars, mpg ~ hp + wt + cyl + disp,
                                       var1 = "hp", var2 = "wt", alpha = 0.05)

  # Check structure of results
  expect_s3_class(test_results, "data.frame")
  expect_true(all(c("var1", "var2", "p_value", "significant", "delta_r2") %in%
                    names(test_results)))

  # Check if test result is for the right variables
  expect_equal(test_results$var1[1], "hp")
  expect_equal(test_results$var2[1], "wt")

  # Test with var1 and all others
  test_all <- tl_test_interactions(mtcars, mpg ~ hp + wt + cyl + disp,
                                   var1 = "hp", alpha = 0.05)

  # Check if results include all pairs with hp
  expect_equal(nrow(test_all), 3)  # Should test hp with wt, cyl, and disp
  expect_true(all(test_all$var1 == "hp"))
  expect_true(all(sort(test_all$var2) == sort(c("wt", "cyl", "disp"))))

  # Test all pairs
  test_all_pairs <- tl_test_interactions(mtcars, mpg ~ hp + wt + cyl,
                                         all_pairs = TRUE, alpha = 0.05)

  # Check if results include all possible pairs
  expect_equal(nrow(test_all_pairs), 3)  # 3 possible pairs: hp-wt, hp-cyl, wt-cyl

  # Test filtering options
  # Create mixed data
  mtcars_mixed <- mtcars
  mtcars_mixed$cyl_factor <- factor(mtcars$cyl)

  # Test categorical interactions
  test_cat <- tl_test_interactions(mtcars_mixed, mpg ~ hp + wt + cyl_factor,
                                   all_pairs = TRUE, categorical_only = TRUE)

  # Should only include pairs with categorical variables
  expect_true(all(sapply(test_cat$var1, function(v) is.factor(mtcars_mixed[[v]])) |
                    sapply(test_cat$var2, function(v) is.factor(mtcars_mixed[[v]]))))

  # Test numeric interactions
  test_num <- tl_test_interactions(mtcars_mixed, mpg ~ hp + wt + cyl_factor,
                                   all_pairs = TRUE, numeric_only = TRUE)

  # Should only include pairs with numeric variables
  expect_true(all(sapply(test_num$var1, function(v) is.numeric(mtcars_mixed[[v]])) &
                    sapply(test_num$var2, function(v) is.numeric(mtcars_mixed[[v]]))))

  # Test mixed interactions
  test_mixed <- tl_test_interactions(mtcars_mixed, mpg ~ hp + wt + cyl_factor,
                                     all_pairs = TRUE, mixed_only = TRUE)

  # Should only include pairs with one numeric and one categorical variable
  if (nrow(test_mixed) > 0) {
    expect_true(all(
      (sapply(test_mixed$var1, function(v) is.numeric(mtcars_mixed[[v]])) &
         sapply(test_mixed$var2, function(v) is.factor(mtcars_mixed[[v]]))) |
        (sapply(test_mixed$var1, function(v) is.factor(mtcars_mixed[[v]])) &
           sapply(test_mixed$var2, function(v) is.numeric(mtcars_mixed[[v]])))
    ))
  }
})

test_that("tl_plot_interaction creates interaction plots", {
  skip_if_not_installed("ggplot2")

  # Create a model with interactions
  model <- tl_model(mtcars, mpg ~ hp * wt, method = "linear")

  # Test numeric x numeric interaction plot
  p1 <- tl_plot_interaction(model, var1 = "hp", var2 = "wt")
  expect_s3_class(p1, "ggplot")

  # Create model with categorical variable
  mtcars_mixed <- mtcars
  mtcars_mixed$cyl_factor <- factor(mtcars$cyl)

  model_mixed <- tl_model(mtcars_mixed, mpg ~ hp * cyl_factor, method = "linear")

  # Test numeric x categorical interaction plot
  p2 <- tl_plot_interaction(model_mixed, var1 = "hp", var2 = "cyl_factor")
  expect_s3_class(p2, "ggplot")

  # Test categorical x numeric interaction plot
  p3 <- tl_plot_interaction(model_mixed, var1 = "cyl_factor", var2 = "hp")
  expect_s3_class(p3, "ggplot")

  # Test with fixed values for other variables
  p4 <- tl_plot_interaction(model, var1 = "hp", var2 = "wt",
                            fixed_values = list(cyl = 6))
  expect_s3_class(p4, "ggplot")

  # Test with confidence intervals
  p5 <- tl_plot_interaction(model, var1 = "hp", var2 = "wt", confidence = TRUE)
  expect_s3_class(p5, "ggplot")

  # Test with different number of points
  p6 <- tl_plot_interaction(model, var1 = "hp", var2 = "wt", n_points = 10)
  expect_s3_class(p6, "ggplot")
})

test_that("tl_auto_interactions finds important interactions", {
  # Find important interactions
  model <- tl_auto_interactions(mtcars, mpg ~ hp + wt + cyl + disp, top_n = 2)

  # Check if result is a tidylearn model
  expect_s3_class(model, "tidylearn_model")

  # Check if interaction tests are stored
  expect_true(!is.null(attr(model, "interaction_tests")))
  expect_true(!is.null(attr(model, "selected_interactions")))

  # Check if selected interactions are a subset of all tests
  all_tests <- attr(model, "interaction_tests")
  selected <- attr(model, "selected_interactions")

  expect_true(nrow(selected) <= nrow(all_tests))
  expect_true(nrow(selected) <= 2)  # Should not exceed top_n

  # Check with minimum R-squared change and maximum p-value
  model2 <- tl_auto_interactions(mtcars, mpg ~ hp + wt + cyl + disp,
                                 min_r2_change = 0.05, max_p_value = 0.01)

  # Check if interaction tests are stored
  expect_true(!is.null(attr(model2, "interaction_tests")))

  # May not find any significant interactions with strict criteria
  # In that case, should return a model without interactions
  expect_s3_class(model2, "tidylearn_model")

  # Test with excluded variables
  model3 <- tl_auto_interactions(mtcars, mpg ~ hp + wt + cyl + disp,
                                 exclude_vars = c("cyl"))

  # Check if excluded variable is not in any interaction
  selected3 <- attr(model3, "selected_interactions")
  if (!is.null(selected3) && nrow(selected3) > 0) {
    expect_true(!any(selected3$var1 == "cyl" | selected3$var2 == "cyl"))
  }
})

test_that("tl_interaction_effects calculates partial effects", {
  # Create a model with interactions
  model <- tl_model(mtcars, mpg ~ hp * wt, method = "linear")

  # Calculate interaction effects
  effects <- tl_interaction_effects(model, var = "hp", by_var = "wt")

  # Check structure of results
  expect_type(effects, "list")
  expect_true(all(c("effects", "slopes") %in% names(effects)))

  # Check if effects dataframe contains the right columns
  expect_true(all(c("hp", "wt", "fit", "by_value", "by_label") %in%
                    names(effects$effects)))

  # Check if slopes dataframe contains the right columns
  expect_true(all(c("by_value", "by_label", "slope", "slope_se") %in%
                    names(effects$slopes)))

  # Test with different by_var values
  # Create a model with categorical variable
  mtcars_mixed <- mtcars
  mtcars_mixed$cyl_factor <- factor(mtcars$cyl)

  model_mixed <- tl_model(mtcars_mixed, mpg ~ hp * cyl_factor, method = "linear")

  # Calculate interaction effects with categorical by_var
  effects2 <- tl_interaction_effects(model_mixed, var = "hp", by_var = "cyl_factor")

  # For categorical variables, just get effects (no slopes list)
  expect_s3_class(effects2, "data.frame")
  expect_true(all(c("hp", "cyl_factor", "fit", "by_value", "by_label") %in%
                    names(effects2)))

  # Check if all levels of cyl_factor are included
  expect_equal(sort(unique(effects2$by_value)), sort(levels(mtcars_mixed$cyl_factor)))

  # Test with at_values
  effects3 <- tl_interaction_effects(model, var = "hp", by_var = "wt",
                                     at_values = list(cyl = 6))

  # Check structure of results
  expect_type(effects3, "list")

  # Test with intervals
  effects4 <- tl_interaction_effects(model, var = "hp", by_var = "wt",
                                     intervals = TRUE)

  # Check if intervals are included in effects
  expect_true(all(c("lower", "upper") %in% names(effects4$effects)))
})

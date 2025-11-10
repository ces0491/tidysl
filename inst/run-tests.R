#!/usr/bin/env Rscript

# Script to run all tests and check code coverage for tidysl package
# Usage: Rscript run-tests.R [all|unit|examples|vignettes|lint|coverage]

library(testthat)
library(covr)
library(devtools)
library(lintr)

# Get command line argument
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  test_type <- "all"
} else {
  test_type <- args[1]
}

# Function to print section headers
print_header <- function(text) {
  cat("\n", rep("=", 80), "\n", text, "\n", rep("=", 80), "\n", sep = "")
}

# Check that we're in the package directory
if (!file.exists("DESCRIPTION")) {
  stop("This script must be run from the package root directory")
}

# Run unit tests
run_unit_tests <- function() {
  print_header("Running unit tests")
  result <- test_dir("tests/testthat/")
  cat("\nUnit test summary:\n")
  print(result)
  invisible(result)
}

# Run examples
run_examples <- function() {
  print_header("Running examples")
  result <- devtools::run_examples()
  invisible(result)
}

# Run vignettes
run_vignettes <- function() {
  print_header("Building vignettes")
  result <- devtools::build_vignettes()
  invisible(result)
}

# Run linter
run_lint <- function() {
  print_header("Running lintr")
  result <- lintr::lint_package()
  if (length(result) == 0) {
    cat("No linting issues found!\n")
  } else {
    print(result)
  }
  invisible(result)
}

# Check code coverage
check_coverage <- function() {
  print_header("Checking code coverage")
  cov <- covr::package_coverage()
  print(cov)
  covr::report(cov)
  invisible(cov)
}

# Run specified tests
if (test_type == "all" || test_type == "unit") {
  run_unit_tests()
}

if (test_type == "all" || test_type == "examples") {
  run_examples()
}

if (test_type == "all" || test_type == "vignettes") {
  run_vignettes()
}

if (test_type == "all" || test_type == "lint") {
  run_lint()
}

if (test_type == "all" || test_type == "coverage") {
  check_coverage()
}

# Final message
if (test_type == "all") {
  print_header("All tests completed")
  cat("To see the coverage report, open the report created by covr::report()\n")
} else {
  print_header(paste(test_type, "tests completed"))
}

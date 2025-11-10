#' Medical Insurance Cost Dataset
#'
#' A dataset containing medical insurance charges for 1338 individuals along
#' with their demographic and health-related characteristics.
#'
#' @format A data frame with 1338 rows and 7 variables:
#' \describe{
#'   \item{age}{Age of the primary beneficiary (years)}
#'   \item{sex}{Gender of the insurance contractor (female, male)}
#'   \item{bmi}{Body mass index, providing an understanding of body weight
#'             relative to height (kg/m^2)}
#'   \item{children}{Number of children/dependents covered by health insurance}
#'   \item{smoker}{Smoking status of the beneficiary (yes, no)}
#'   \item{region}{The beneficiary's residential area in the US (northeast,
#'                 southeast, southwest, northwest)}
#'   \item{charges}{Individual medical costs billed by health insurance (USD)}
#' }
#'
#' @source \url{https://www.kaggle.com/datasets/mirichoi0218/insurance}
#'
#' @examples
#' \dontrun{
#' data(insurance)
#' head(insurance)
#' }
"insurance"

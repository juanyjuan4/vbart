#' Sample from the VBART... whatever that means
#'
#' This function runs the VBART sampler.
#'
#' @param vbart_obj
#'
#' @return A matrix of results.
#' @export
#' @examples
#' vbart_sampler(y, 1000, 50, 2, lon, lat)
get_yhat_mat <- function(vbart_obj) {
  .Call(.get_yhat_mat, vbart_obj)
}

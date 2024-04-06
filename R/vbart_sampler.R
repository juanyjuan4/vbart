#' Sample from the VBART... whatever that means
#'
#' This function runs the VBART sampler.
#'
#' @param y A vector of data points.
#' @param n_samples A scalar giving the number of samples.
#' @param n_trees A scalar giving the number of trees.
#' @param delta A scalar parameter used to parameterize the underlying Poisson distribution.
#' @param lon A vector of longitudes.
#' @param lat A vector of latitudes.
#'
#' @return A list of results.
#' @export
#' @examples
#' vbart_sampler(y, 1000, 50, 2, lon, lat)
vbart_sampler <- function(y, n_samples, n_trees, delta, lon, lat) {
  .Call(.vbart_sampler, y, n_samples, n_trees, delta, lon, lat)
}

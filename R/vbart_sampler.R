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

  expected_max_b <- delta * 6
  nrows = length(y)

  s_list <- vector(mode = "list", length = n_samples)
  mu_list <- vector(mode = "list", length = n_samples)

  for (scan_index in 1:n_samples) {
    s_list[[scan_index]] <- vector(mode = "list", length = n_trees)
    mu_list[[scan_index]] <- vector(mode = "list", length = n_trees)
    for (tree_index in 1:n_trees) {
      s_list[[scan_index]][[tree_index]] <- list(
        "s_mat" = matrix(rep(-99999, 2 * expected_max_b), ncol = 2),
        "dtr" = rep(-1, expected_max_b),
        "dtd" = rep(-1, expected_max_b),
        "mu_index" = rep(-1L, nrows),
        "log_crit" = 0.0
      )
      mu_list[[scan_index]][[tree_index]] <- rep(-1, expected_max_b);
    }
  }
  tau_vec <- rep(-1, n_samples);
  yhat_mat <- matrix(rep(-1, nrows * n_samples), ncol = nrows)
  
  .Call(.vbart_sampler, y, n_samples, n_trees, delta, lon, lat, s_list, mu_list, tau_vec, yhat_mat)
  
  for (si in 1:n_samples) {
    for (ti in 1:n_trees) {
      temp_s <- s_list[[si]][[ti]]$s_mat
      s_list[[si]][[ti]]$s_mat <- temp_s[temp_s[,1]!=-99999,temp_s[1,]!=-99999]
      temp_dtr <- s_list[[si]][[ti]]$dtr
      s_list[[si]][[ti]]$dtr <- temp_dtr[temp_dtr!=-1]
      temp_dtd <- s_list[[si]][[ti]]$dtd
      s_list[[si]][[ti]]$dtd <- temp_dtd[temp_dtd!=-1]
      temp_mu <- mu_list[[si]][[ti]]
      mu_list[[si]][[ti]] <- temp_mu[temp_mu!=-1]
    }
  }
  
  list("s_list" = s_list, "mu_list" = mu_list, "tau_vec" = tau_vec, "yhat_mat" = yhat_mat)
}

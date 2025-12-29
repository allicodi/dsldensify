make_hazard_dt <- function(
  n_id = 50L,
  max_bins = 4L,      # each obs has between 1 and max_bins rows
  seed = 1L,
  with_wts = TRUE
) {
  testthat::skip_if_not_installed("data.table")
  data.table::setDTthreads(1L)

  set.seed(seed)

  # number of rows per obs_id (discrete uniform 1..max_bins)
  n_rows <- sample.int(max_bins, size = n_id, replace = TRUE)

  dt <- data.table::data.table(
    obs_id = rep(seq_len(n_id), times = n_rows)
  )

  # bin index within obs (1..n_i)
  dt[, bin_id := seq_len(.N), by = obs_id]

  # simple numeric features (can include bin_id signal if you want)
  dt[, x1 := rnorm(.N)]
  dt[, x2 := rnorm(.N)]

  # hazard format: last row per obs is 1, others 0
  dt[, in_bin := as.integer(bin_id == max(bin_id)), by = obs_id]

  if (with_wts) {
    # constant within obs_id is typical
    w <- runif(n_id, 0.5, 2)
    dt[, wts := rep(w, times = n_rows)]
  }

  dt[]
}

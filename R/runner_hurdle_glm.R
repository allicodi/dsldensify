make_glm_hurdle_runner <- function(
  rhs_list,
  use_weights_col = TRUE,
  strip_fit = TRUE,
  ...
) {
  stopifnot(requireNamespace("stats", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))

  # Fixed hurdle conventions
  outcome_col <- "in_hurdle"
  weights_col <- "wts"

  # RHS parsing: formulas (~ ...) or strings
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  tune_grid <- data.frame(
    .tune = seq_along(rhs_chr),
    rhs = rhs_chr,
    stringsAsFactors = FALSE
  )

  clip01 <- function(p, eps) pmin(pmax(p, eps), 1 - eps)

  strip_glm <- function(fit) {
    if (!isTRUE(strip_fit)) return(fit)
    # keep enough for predict.glm(); drop heavy components
    fit$model <- NULL
    fit$y <- NULL
    fit$x <- NULL
    fit
  }

  predict_prob_one <- function(fit, newdata) {
    # type="response" returns P(in_hurdle=1 | W)
    as.numeric(stats::predict(fit, newdata = newdata, type = "response"))
  }

  # ---- NEW: shared probability prediction helper -------------------------
  # Returns n x K matrix of probabilities, aligned with fits ordering.
  predict_pi <- function(fits, newdata, eps = 1e-15, ...) {
    nd <- data.table::as.data.table(newdata)
    if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")

    n <- nrow(nd)
    K <- length(fits)
    out <- matrix(NA_real_, nrow = n, ncol = K)

    for (k in seq_len(K)) {
      p <- predict_prob_one(fits[[k]], nd)
      out[, k] <- clip01(p, eps)
    }
    out
  }

  list(
    method = "hurdle_glm",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      dat <- data.table::as.data.table(train_set)
      if (!(outcome_col %in% names(dat))) {
        stop("train_set must contain column '", outcome_col, "'.")
      }

      has_wts <- isTRUE(use_weights_col) && (weights_col %in% names(dat))
      wts <- if (has_wts) dat[[weights_col]] else NULL

      fits <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        rhs <- tune_grid$rhs[k]
        fml <- stats::as.formula(paste0(outcome_col, " ~ ", rhs))

        fit_k <- suppressWarnings(stats::glm(
          formula = fml,
          data = dat,
          family = stats::binomial(),
          weights = wts
        ))

        fits[[k]] <- strip_glm(fit_k)
      }

      list(
        fits = fits,
        rhs_chr = rhs_chr,
        tune = seq_len(nrow(tune_grid)),
        stripped = isTRUE(strip_fit)
      )
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      dat <- data.table::as.data.table(train_set)
      if (!(outcome_col %in% names(dat))) {
        stop("train_set must contain column '", outcome_col, "'.")
      }

      has_wts <- isTRUE(use_weights_col) && (weights_col %in% names(dat))
      wts <- if (has_wts) dat[[weights_col]] else NULL

      rhs <- tune_grid$rhs[k]
      fml <- stats::as.formula(paste0(outcome_col, " ~ ", rhs))

      fit_k <- suppressWarnings(stats::glm(
        formula = fml,
        data = dat,
        family = stats::binomial(),
        weights = wts
      ))

      list(
        fits = list(strip_glm(fit_k)),
        rhs_chr = rhs,
        tune = k,
        stripped = isTRUE(strip_fit)
      )
    },

    # log(pi(W)) as n x K
    logpi = function(fit_bundle, newdata, eps = 1e-15, ...) {
      p <- predict_pi(fits = fit_bundle$fits, newdata = newdata, eps = eps, ...)
      log(p)
    },

    # Bernoulli negative log-likelihood as n x K
    log_density = function(fit_bundle, newdata, eps = 1e-15, ...) {
      nd <- data.table::as.data.table(newdata)
      if (!(outcome_col %in% names(nd))) {
        stop("hurdle glm runner requires `", outcome_col, "` column in newdata")
      }

      y <- as.integer(nd[[outcome_col]])
      if (anyNA(y) || any(!(y %in% c(0L, 1L)))) {
        stop("`", outcome_col, "` must be coded 0/1 with no NA.")
      }

      p <- predict_pi(fits = fit_bundle$fits, newdata = nd, eps = eps, ...)
      y_mat <- matrix(y, nrow = length(y), ncol = ncol(p))

      -(y_mat * log(p) + (1L - y_mat) * log1p(-p))
    },

    # Sample in_hurdle ~ Bernoulli(pi(W)); returns n x n_samp matrix
    # Assumes post-selection K=1, same as hazard samplers.
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, eps = 1e-15, ...) {
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) {
        stop("n_samp must be a positive integer.")
      }
      n_samp <- as.integer(n_samp)
      if (!is.null(seed)) set.seed(seed)

      p_mat <- predict_pi(fits = fit_bundle$fits, newdata = newdata, eps = eps, ...)
      if (ncol(p_mat) != 1L) {
        stop("sample() assumes K=1: fit_bundle must contain exactly one selected model.")
      }

      p <- as.numeric(p_mat[, 1L])
      n <- length(p)
      matrix(stats::rbinom(n * n_samp, size = 1L, prob = p), nrow = n, ncol = n_samp)
    }
  )
}

#' Create a Gaussian (normal) conditional density runner with homoscedastic errors
#'
#' @description
#' Constructs a **direct density** runner that models
#' \eqn{A \mid W \sim \mathcal{N}(\mu(W), \sigma^2)} with **constant** \eqn{\sigma}
#' across observations. The conditional mean \eqn{\mu(W)} is fit via
#' weighted least squares using \code{stats::lm()} with one model per RHS in
#' \code{rhs_list}. The runner exposes \code{log_density()} (and optionally
#' \code{density()}) for use in \code{run_direct_setting()} and global selection.
#'
#' The runner also provides \code{sample()} to draw
#' \eqn{A^* \sim \hat f(\cdot \mid W)} from the fitted conditional normal model.
#' Sampling is intended to be called on a **selected** fit (i.e.,
#' \code{length(fit_bundle$fits) == 1}).
#'
#' @section Numeric-only requirement:
#' For stable behavior across CV folds and to avoid factor-level bookkeeping,
#' this runner is intended for **numeric predictors only**. RHS formulas should
#' not include factors/characters. No coercion is performed.
#'
#' @param rhs_list A list of RHS specifications, either:
#' \itemize{
#'   \item one-sided formulas such as \code{~ W1 + W2}, or
#'   \item character strings such as \code{"W1 + W2"}.
#' }
#' Each RHS is used to form \code{A ~ <rhs>} for fitting the mean model.
#'
#' @param use_weights_col Logical. If \code{TRUE} and \code{train_set} contains a
#' column named \code{wts}, fitting uses weighted least squares and \eqn{\sigma}
#' is estimated using the same weights.
#'
#' @param strip_fit Logical. If \code{TRUE}, attempt to reduce stored fit size by
#' removing large components from the \code{lm} object (keeps prediction working).
#'
#' @return A runner (named list) with elements:
#' \describe{
#'   \item{method}{Character string \code{"gaussian_homosked"}.}
#'   \item{tune_grid}{Data frame with columns \code{.tune} and \code{rhs}.}
#'   \item{fit}{Function \code{fit(train_set, ...)} returning \code{list(fits=...)}.}
#'   \item{log_density}{Function returning an \code{n x K} matrix of log densities.}
#'   \item{density}{Function returning an \code{n x K} matrix of densities.}
#'   \item{sample}{Function drawing samples \code{A*} given \code{W} (assumes \code{K=1}).}
#'   \item{fit_one}{Fit only the selected tune (returns minimal bundle).}
#' }
#'
#' @export
make_gaussian_homosced_direct_runner <- function(
  rhs_list,
  use_weights_col = TRUE,
  strip_fit = TRUE
) {

  # normalize rhs_list -> character vector of RHS strings
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) {
      paste(deparse(f[[2]]), collapse = "")
    }, character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  tune_grid <- data.frame(
    .tune = seq_along(rhs_chr),
    rhs = rhs_chr,
    stringsAsFactors = FALSE
  )

  strip_lm <- function(fit) {
    # keep enough for predict.lm(); drop heavy stuff
    fit$model <- NULL
    fit$y <- NULL
    fit$x <- NULL
    # fit$qr <- NULL
    fit
  }

  # estimate homoscedastic sigma using (optionally) weights
  # sigma^2 = sum(w * r^2) / sum(w)
  sigma_hat <- function(resid, w) {
    if (is.null(w)) {
      sqrt(mean(resid^2))
    } else {
      sqrt(sum(w * resid^2) / sum(w))
    }
  }

  list(
    method = "gaussian_homosked",
    tune_grid = tune_grid,
    positive_support = FALSE,
    
    fit = function(train_set, ...) {
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      fits <- vector("list", nrow(tune_grid))
      for (k in seq_len(nrow(tune_grid))) {
        f_k <- as.formula(paste0("A ~ ", tune_grid$rhs[k]))

        lm_fit <- stats::lm(
          f_k,
          data = train_set,
          weights = wts_vec,
          ...
        )

        r <- stats::residuals(lm_fit)
        sig <- sigma_hat(r, wts_vec)

        if (strip_fit) lm_fit <- strip_lm(lm_fit)

        fits[[k]] <- list(
          fit = lm_fit,
          sigma = sig
        )
      }

      list(fits = fits)
    },

    # n x K matrix of log densities at observed A in newdata
    log_density = function(fit_bundle, newdata, eps = 1e-12, ...) {
      nd <- as.data.frame(newdata)

      K <- length(fit_bundle$fits)
      out <- matrix(NA_real_, nrow = nrow(nd), ncol = K)

      for (k in seq_len(K)) {
        obj <- fit_bundle$fits[[k]]
        mu <- stats::predict(obj$fit, newdata = nd, ...)

        sd_k <- pmax(obj$sigma, sqrt(eps))
        out[, k] <- stats::dnorm(nd$A, mean = mu, sd = sd_k, log = TRUE)
      }

      out <- pmax(out, log(eps))
      out
    },

    density = function(fit_bundle, newdata, eps = 1e-12, ...) {
      exp(log_density(fit_bundle, newdata, eps = eps, ...))
    },

    # Draw A* ~ Normal(mu(W), sigma^2) for each row of newdata (wide W).
    # Assumes fit_bundle contains exactly one tuned fit (K = 1), which is the
    # intended usage after global selection or fit_one().
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, eps = 1e-12, ...) {
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)
      if (!is.null(seed)) set.seed(seed)

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      if (length(fits) != 1L) stop("sample() assumes K=1: fit_bundle$fits must have length 1 (selected model).")

      nd <- as.data.frame(newdata)
      obj <- fits[[1]]

      mu <- stats::predict(obj$fit, newdata = nd, ...)
      sd1 <- pmax(as.numeric(obj$sigma), sqrt(eps))
      if (!is.finite(sd1) || sd1 <= 0) stop("Non-positive or non-finite sigma in fit bundle.")

      n <- length(mu)
      # vectorized: generate n*n_samp normals, then reshape
      out <- matrix(
        stats::rnorm(n * n_samp, mean = rep(mu, times = n_samp), sd = sd1),
        nrow = n,
        ncol = n_samp
      )

      out
    },

    fit_one = function(train_set, tune, ...) {
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      f_k <- as.formula(paste0("A ~ ", tune_grid$rhs[tune]))

      lm_fit <- stats::lm(
        f_k,
        data = train_set,
        weights = wts_vec,
        ...
      )

      r <- stats::residuals(lm_fit)
      sig <- sigma_hat(r, wts_vec)

      if (strip_fit) lm_fit <- strip_lm(lm_fit)

      list(fits = list(list(fit = lm_fit, sigma = sig)), tune = as.integer(tune))
    }
  )
}

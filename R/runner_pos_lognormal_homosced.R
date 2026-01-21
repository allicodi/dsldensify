#' Create a log-normal conditional density runner with homoscedastic log-scale errors
#'
#' @description
#' Constructs a **direct density** runner for strictly positive outcomes,
#' modeling
#' \eqn{\log(A) \mid W \sim \mathcal{N}(\mu(W), \sigma^2)}
#' with **constant** \eqn{\sigma} across observations on the log scale.
#' The conditional mean \eqn{\mu(W)} is fit via weighted least squares using
#' \code{stats::lm()} with one model per RHS in \code{rhs_list}.
#'
#' The conditional density of \eqn{A \mid W} is log-normal:
#' \deqn{
#' f(A \mid W) = \frac{1}{A}\,\phi\left(\frac{\log A - \mu(W)}{\sigma}\right),
#' }
#' hence \code{log_density()} includes the Jacobian term \eqn{-\log(A)}.
#'
#' The runner also provides \code{sample()} to draw
#' \eqn{A^* \sim \hat f(\cdot \mid W)} from the fitted conditional log-normal
#' model. Sampling is intended to be called on a **selected** fit (i.e.,
#' \code{length(fit_bundle$fits) == 1}).
#'
#' @section Strict positivity requirement:
#' This runner requires \eqn{A > 0}. If any \eqn{A \le 0} appears in training data
#' (or in \code{newdata} passed to \code{log_density()}), the runner errors.
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
#' Each RHS is used to form \code{log(A) ~ <rhs>} for fitting the log-mean model.
#'
#' @param use_weights_col Logical. If \code{TRUE} and \code{train_set} contains a
#' column named \code{wts}, fitting uses weighted least squares and \eqn{\sigma}
#' is estimated using the same weights (on the log scale).
#'
#' @param strip_fit Logical. If \code{TRUE}, attempt to reduce stored fit size by
#' removing large components from the \code{lm} object (keeps prediction working).
#'
#' @return A runner (named list) with elements:
#' \describe{
#'   \item{method}{Character string \code{"lognormal_homosked"}.}
#'   \item{tune_grid}{Data frame with columns \code{.tune} and \code{rhs}.}
#'   \item{fit}{Function \code{fit(train_set, ...)} returning \code{list(fits=...)}.}
#'   \item{log_density}{Function returning an \code{n x K} matrix of log densities.}
#'   \item{density}{Function returning an \code{n x K} matrix of densities.}
#'   \item{sample}{Function drawing samples \code{A*} given \code{W} (assumes \code{K=1}).}
#'   \item{fit_one}{Fit only the selected tune (returns minimal bundle).}
#' }
#'
#' @export
make_lognormal_homosced_pos_runner <- function(
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

  # estimate homoscedastic sigma on log scale using (optionally) weights
  # sigma^2 = sum(w * r^2) / sum(w)
  sigma_hat <- function(resid, w) {
    if (is.null(w)) {
      sqrt(mean(resid^2))
    } else {
      sqrt(sum(w * resid^2) / sum(w))
    }
  }

  # shared helper: validate positivity + compute mu(log-scale mean) for each k
  predict_mu <- function(fits, nd, ...) {
    if (!("A" %in% names(nd))) stop("newdata must contain column 'A'.")
    if (any(!is.finite(nd$A))) stop("Non-finite values in A.")
    if (any(nd$A <= 0)) stop("lognormal runner requires A > 0 (found A <= 0).")

    K <- length(fits)
    mu_mat <- matrix(NA_real_, nrow = nrow(nd), ncol = K)
    for (k in seq_len(K)) {
      mu_mat[, k] <- stats::predict(fits[[k]]$fit, newdata = nd, ...)
    }
    mu_mat
  }

  list(
    method = "lognormal_homosked",
    tune_grid = tune_grid,
    positive_support = TRUE,

    fit = function(train_set, ...) {
      if (!("A" %in% names(train_set))) stop("train_set must contain column 'A'.")
      if (any(!is.finite(train_set$A))) stop("Non-finite values in A.")
      if (any(train_set$A <= 0)) stop("lognormal runner requires A > 0 (found A <= 0).")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      fits <- vector("list", nrow(tune_grid))
      for (k in seq_len(nrow(tune_grid))) {
        # Fit on log(A)
        f_k <- as.formula(paste0("log(A) ~ ", tune_grid$rhs[k]))

        lm_fit <- stats::lm(
          f_k,
          data = train_set,
          weights = wts_vec,
          ...
        )

        r <- stats::residuals(lm_fit)  # residuals on log scale
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

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      K <- length(fits)

      # strict positivity check happens inside predict_mu()
      logA <- log(nd$A)
      mu_mat <- predict_mu(fits, nd, ...)
      out <- matrix(NA_real_, nrow = nrow(nd), ncol = K)

      for (k in seq_len(K)) {
        obj <- fits[[k]]
        sd_k <- pmax(as.numeric(obj$sigma), sqrt(eps))
        # log f(A|W) = log f_Y(logA|W) - log(A)
        out[, k] <- stats::dnorm(logA, mean = mu_mat[, k], sd = sd_k, log = TRUE) - logA
      }

      out <- pmax(out, log(eps))
      out
    },

    density = function(fit_bundle, newdata, eps = 1e-12, ...) {
      exp(log_density(fit_bundle, newdata, eps = eps, ...))
    },

    # Draw A* ~ LogNormal(mu(W), sigma^2) for each row of newdata (wide W).
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
      if (!("A" %in% names(nd))) stop("newdata must contain column 'A'.")
      # For sampling, A isn't used, but we keep the interface consistent; require finite if present.
      if (any(!is.finite(nd$A))) stop("Non-finite values in A.")

      obj <- fits[[1]]

      mu <- stats::predict(obj$fit, newdata = nd, ...)
      sd1 <- pmax(as.numeric(obj$sigma), sqrt(eps))
      if (!is.finite(sd1) || sd1 <= 0) stop("Non-positive or non-finite sigma in fit bundle.")

      n <- length(mu)
      # sample on log scale, then exponentiate
      y <- stats::rnorm(n * n_samp, mean = rep(mu, times = n_samp), sd = sd1)
      out <- matrix(exp(y), nrow = n, ncol = n_samp)

      # should be strictly positive unless overflow produced Inf
      if (any(!is.finite(out))) stop("Non-finite draws produced in lognormal sample().")
      out
    },

    fit_one = function(train_set, tune, ...) {
      if (!("A" %in% names(train_set))) stop("train_set must contain column 'A'.")
      if (any(!is.finite(train_set$A))) stop("Non-finite values in A.")
      if (any(train_set$A <= 0)) stop("lognormal runner requires A > 0 (found A <= 0).")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      f_k <- as.formula(paste0("log(A) ~ ", tune_grid$rhs[tune]))

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

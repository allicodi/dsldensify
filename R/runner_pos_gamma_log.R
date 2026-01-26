#' Create a Gamma GLM conditional density runner with log link
#'
#' @description
#' Constructs a **direct density** runner for strictly positive outcomes,
#' modeling
#' \eqn{A \mid W \sim \mathrm{Gamma}(\mu(W), \phi)} with a log link:
#' \deqn{\log \mu(W) = X\beta.}
#' The mean model is fit via \code{stats::glm()} with
#' \code{family = stats::Gamma(link = "log")}, one model per RHS in
#' \code{rhs_list}.
#'
#' Density evaluation uses the Gamma GLM variance convention
#' \eqn{\mathrm{Var}(A \mid W) = \phi \mu(W)^2}. Under this convention, a Gamma
#' distribution with
#' \eqn{\text{shape} = 1/\phi} and \eqn{\text{scale} = \mu \phi}
#' has mean \eqn{\mu} and variance \eqn{\phi \mu^2}. Thus \code{log_density()}
#' computes
#' \code{dgamma(A, shape = 1/phi, scale = mu * phi, log = TRUE)}.
#'
#' The runner also provides \code{sample()} to draw
#' \eqn{A^* \sim \hat f(\cdot \mid W)} from the fitted conditional Gamma model.
#' Sampling is intended to be called on a **selected** fit (i.e.,
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
#' Each RHS is used to form \code{A ~ <rhs>} for fitting the mean model.
#'
#' @param use_weights_col Logical. If \code{TRUE} and \code{train_set} contains a
#' column named \code{wts}, fitting uses \code{weights = wts}.
#'
#' @param strip_fit Logical. If \code{TRUE}, attempt to reduce stored fit size by
#' removing large components from the \code{glm} object (keeps prediction working).
#'
#' @param maxit Integer. Maximum number of IRWLS iterations passed via
#' \code{glm.control(maxit = ...)}.
#'
#' @param epsilon Numeric. Convergence tolerance passed via
#' \code{glm.control(epsilon = ...)}.
#'
#' @return A runner (named list).
#'
#' @export
make_gamma_glm_log_pos_runner <- function(
  rhs_list,
  use_weights_col = TRUE,
  strip_fit = TRUE,
  maxit = 50L,
  epsilon = 1e-8
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

  strip_glm <- function(fit) {
    # keep enough for predict.glm(); drop heavy stuff
    fit$model <- NULL
    fit$y <- NULL
    fit$x <- NULL
    # fit$qr <- NULL
    fit
  }

  # shared helper: strict A>0 check for train/eval
  check_pos_A <- function(A, where = "data") {
    if (any(!is.finite(A))) stop("Non-finite values in A (", where, ").")
    if (any(A <= 0)) stop("gamma(log) runner requires A > 0 (found A <= 0 in ", where, ").")
    invisible(TRUE)
  }

  # shared helper: for a list of fits, get mu (n x K) and phi (length K)
  # mu uses predict(type="response"), which for log link returns exp(eta)
  predict_params <- function(fits, nd, ...) {
    if (!("A" %in% names(nd))) stop("newdata must contain column 'A'.")
    check_pos_A(nd$A, where = "newdata")

    K <- length(fits)
    n <- nrow(nd)

    mu_mat <- matrix(NA_real_, nrow = n, ncol = K)
    phi_vec <- rep(NA_real_, K)

    for (k in seq_len(K)) {
      fit_k <- fits[[k]]$fit

      mu_k <- stats::predict(fit_k, newdata = nd, type = "response", ...)
      if (any(!is.finite(mu_k))) stop("Non-finite predicted mean mu in model ", k, ".")
      if (any(mu_k <= 0)) stop("Non-positive predicted mean mu in model ", k, " (unexpected for log link).")

      # dispersion (phi) for Gamma GLM: summary(fit)$dispersion is the standard route
      phi_k <- fits[[k]]$phi
      if (!is.finite(phi_k) || phi_k <= 0) stop("Non-positive or non-finite dispersion phi in model ", k, ".")

      mu_mat[, k] <- mu_k
      phi_vec[k] <- phi_k
    }

    list(mu = mu_mat, phi = phi_vec)
  }

  list(
    method = "gamma_glm_log",
    tune_grid = tune_grid,
    positive_support = TRUE,

    fit = function(train_set, ...) {
      if (!("A" %in% names(train_set))) stop("train_set must contain column 'A'.")
      check_pos_A(train_set$A, where = "train_set")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      ctrl <- stats::glm.control(maxit = as.integer(maxit), epsilon = as.numeric(epsilon))

      fits <- vector("list", nrow(tune_grid))
      for (k in seq_len(nrow(tune_grid))) {
        f_k <- as.formula(paste0("A ~ ", tune_grid$rhs[k]))

        glm_fit <- stats::glm(
          f_k,
          data = train_set,
          weights = wts_vec,
          family = stats::Gamma(link = "log"),
          control = ctrl,
          ...
        )

        if (isFALSE(glm_fit$converged)) {
          stop("Gamma(log) GLM did not converge for tune ", k, " (rhs = ", tune_grid$rhs[k], ").")
        }

        phi_k <- summary(glm_fit)$dispersion
        if (!is.finite(phi_k) || phi_k <= 0) stop("Non-positive or non-finite dispersion phi for tune ", k, ".")

        if (strip_fit) glm_fit <- strip_glm(glm_fit)

        fits[[k]] <- list(
          fit = glm_fit,
          phi = as.numeric(phi_k)
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

      pars <- predict_params(fits, nd, ...)
      mu_mat <- pars$mu
      phi_vec <- pars$phi

      out <- matrix(NA_real_, nrow = nrow(nd), ncol = K)

      for (k in seq_len(K)) {
        mu_k <- mu_mat[, k]
        phi_k <- pmax(phi_vec[k], eps)

        # Gamma GLM variance convention: Var(A|W)=phi*mu^2
        # shape = 1/phi, scale_i = mu_i * phi
        shape_k <- 1 / phi_k
        scale_k <- mu_k * phi_k

        # stabilize in case of tiny mu (should be >0, but clamp anyway)
        scale_k <- pmax(scale_k, eps)

        out[, k] <- stats::dgamma(nd$A, shape = shape_k, scale = scale_k, log = TRUE)
      }

      out <- pmax(out, log(eps))
      out
    },

    density = function(fit_bundle, newdata, eps = 1e-12, ...) {
      exp(log_density(fit_bundle, newdata, eps = eps, ...))
    },

    # Draw A* ~ Gamma(shape=1/phi, scale=mu*phi) for each row of newdata (wide W).
    # Assumes fit_bundle contains exactly one tuned fit (K = 1).
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, eps = 1e-12, ...) {
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)
      if (!is.null(seed)) set.seed(seed)

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      if (length(fits) != 1L) stop("sample() assumes K=1: fit_bundle$fits must have length 1 (selected model).")

      nd <- as.data.frame(newdata)
      pars <- predict_params(fits, nd, ...)
      mu <- pars$mu[, 1]
      phi <- pars$phi[1]

      phi <- pmax(phi, eps)
      shape <- 1 / phi
      scale <- pmax(mu * phi, eps)

      if (!is.finite(shape) || shape <= 0) stop("Non-positive or non-finite shape in fit bundle.")
      if (any(!is.finite(scale)) || any(scale <= 0)) stop("Non-positive or non-finite scale in fit bundle.")

      n <- length(mu)
      draws <- stats::rgamma(n * n_samp, shape = shape, scale = rep(scale, times = n_samp))
      out <- matrix(draws, nrow = n, ncol = n_samp)

      if (any(!is.finite(out)) || any(out <= 0)) stop("Non-finite or non-positive draws produced in gamma sample().")
      out
    },

    fit_one = function(train_set, tune, ...) {
      if (!("A" %in% names(train_set))) stop("train_set must contain column 'A'.")
      check_pos_A(train_set$A, where = "train_set")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      ctrl <- stats::glm.control(maxit = as.integer(maxit), epsilon = as.numeric(epsilon))

      f_k <- as.formula(paste0("A ~ ", tune_grid$rhs[tune]))

      glm_fit <- stats::glm(
        f_k,
        data = train_set,
        weights = wts_vec,
        family = stats::Gamma(link = "log"),
        control = ctrl,
        ...
      )

      if (isFALSE(glm_fit$converged)) {
        stop("Gamma(log) GLM did not converge for tune ", tune, " (rhs = ", tune_grid$rhs[tune], ").")
      }

      phi_k <- summary(glm_fit)$dispersion
      if (!is.finite(phi_k) || phi_k <= 0) stop("Non-positive or non-finite dispersion phi for tune ", tune, ".")

      if (strip_fit) glm_fit <- strip_glm(glm_fit)

      list(fits = list(list(fit = glm_fit, phi = as.numeric(phi_k))), tune = as.integer(tune))
    }
  )
}

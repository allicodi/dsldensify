#' Create a glmnet runner for hurdle probability modeling
#'
#' Constructs a runner (learner adapter) for modeling the hurdle probability
#' \eqn{\pi(W) = P(A = a_0 \mid W)} using penalized logistic regression via
#' glmnet::glmnet(). The runner is compatible with the hurdle workflow in
#' dsldensify, where the hurdle component is fit on wide data with binary
#' outcome \code{in_hurdle}.
#'
#' Tuning grid
#'
#' Tuning is performed over a grid defined by:
#'   - rhs_list (RHS varies slowest),
#'   - alpha_grid,
#'   - lambda_grid (varies fastest).
#'
#' Design matrix handling
#'
#' Predictors are expanded using model.matrix() for each RHS. The runner stores
#' per-RHS design specifications and aligns newdata columns to the training
#' design by name, filling missing columns with zeros. This runner assumes
#' numeric predictors only (no factor handling is performed).
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All columns
#' referenced in rhs_list must already be numeric. Factors, characters, and
#' ordered factors are not supported and are not coerced internally.
#'
#' @param rhs_list A list of RHS specifications, either as one-sided formulas
#'   (for example, ~ W1 + W2) or as character strings (for example, "W1 + W2").
#'
#' @param alpha_grid Numeric vector of elastic net mixing parameters.
#'
#' @param lambda_grid Numeric vector of regularization strengths (must be
#'   strictly positive, length >= 2).
#'
#' @param use_weights_col Logical. If TRUE and the training data contain a
#'   column named wts, it is passed as case weights to glmnet::glmnet().
#'
#' @param standardize Logical. Passed to glmnet::glmnet() to standardize
#'   predictors.
#'
#' @param intercept Logical. Passed to glmnet::glmnet() to include an intercept.
#'
#' @param strip_fit Logical. If TRUE, optionally strip large objects from fits.
#'
#' @param ... Additional arguments forwarded to glmnet::glmnet().
#'
#' @return A named list (runner) with the following elements:
#'   method: Character string "hurdle_glmnet".
#'   tune_grid: Data frame describing the tuning grid, including .tune, rhs,
#'     alpha, and lambda.
#'   fit: Function fit(train_set, ...) returning a fit bundle.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected
#'     tuning index.
#'   logpi: Function logpi(fit_bundle, newdata, ...) returning log probabilities.
#'   log_density: Function log_density(fit_bundle, newdata, ...) returning
#'     Bernoulli negative log-likelihoods.
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing hurdle
#'     indicators (assumes K = 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata as wide data containing:
#'   - a binary outcome column in_hurdle,
#'   - covariates referenced in rhs_list,
#'   - an optional weight column wts.
#'
#' @examples
#' rhs_list <- list(~ W1 + W2)
#'
#' runner <- make_glmnet_hurdle_runner(
#'   rhs_list = rhs_list,
#'   alpha_grid = c(0, 1),
#'   lambda_grid = exp(seq(log(1e-4), log(1), length.out = 10))
#' )
#'
#' @export
make_glmnet_hurdle_runner <- function(
  rhs_list,
  alpha_grid,
  lambda_grid,
  use_weights_col = TRUE,
  standardize = TRUE,
  intercept = TRUE,
  strip_fit = FALSE,
  ...
) {
  stopifnot(requireNamespace("glmnet", quietly = TRUE))
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
  if (length(alpha_grid) < 1L) stop("alpha_grid must have length >= 1")
  if (is.null(lambda_grid) || length(lambda_grid) < 2L) stop("lambda_grid must have length >= 2")
  lambda_grid <- as.numeric(lambda_grid)
  if (any(!is.finite(lambda_grid)) || any(lambda_grid <= 0)) stop("lambda_grid must be finite and strictly positive")

  # Tune grid order: rhs major, then alpha, then lambda (lambda varies fastest)
  tune_grid <- expand.grid(
    lambda = lambda_grid,
    alpha  = alpha_grid,
    rhs    = rhs_chr,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  tune_grid$.tune <- seq_len(nrow(tune_grid))
  tune_grid <- tune_grid[, c(".tune", "rhs", "alpha", "lambda")]

  clip01 <- function(p, eps) pmin(pmax(p, eps), 1 - eps)

  # ---- design helpers -----------------------------------------------------

  build_design_train <- function(rhs_raw, train_set) {
    fml <- stats::as.formula(paste0(outcome_col, " ~ ", rhs_raw))
    tt <- stats::terms(fml, data = train_set)
    tt <- stats::delete.response(tt)

    X <- stats::model.matrix(tt, data = train_set)

    # glmnet intercept handled separately; drop explicit intercept column
    if ("(Intercept)" %in% colnames(X)) {
      X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
    }

    list(
      X = X,
      design_spec = list(tt = tt, x_cols = colnames(X), rhs = rhs_raw)
    )
  }

  build_design_new <- function(design_spec, newdata) {
    Xn <- stats::model.matrix(design_spec$tt, data = newdata)

    if ("(Intercept)" %in% colnames(Xn)) {
      Xn <- Xn[, colnames(Xn) != "(Intercept)", drop = FALSE]
    }

    x_cols <- design_spec$x_cols
    missing <- setdiff(x_cols, colnames(Xn))
    if (length(missing)) {
      Xn <- cbind(Xn, matrix(0, nrow(Xn), length(missing), dimnames = list(NULL, missing)))
    }

    Xn <- Xn[, x_cols, drop = FALSE]
    Xn
  }


  # Optional strip hook (kept intentionally simple for hurdle)
  strip_glmnet_fit <- function(fit) {
    if (!isTRUE(strip_fit)) return(fit)
    # For now: keep glmnet object (already compact enough for wide hurdle).
    fit
  }

  # Internal prediction helper (DO NOT call base::predict() on fit_bundle)
  predict_pi_matrix <- function(fit_bundle, nd, type = c("prob", "log_prob"), eps = 1e-15, ...) {
    type <- match.arg(type)
    nd <- data.table::as.data.table(nd)

    fits <- fit_bundle$fits
    if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
    n <- nrow(nd)

    # Selected bundle: fits is flat list length 1; lambda scalar in fit_bundle$lambda_grid
    if (!is.null(fit_bundle$selected) && isTRUE(fit_bundle$selected)) {
      fit1 <- fits[[1L]]
      rhs_raw <- fit_bundle$rhs_chr
      ds <- fit_bundle$design_specs[[rhs_raw]]
      Xn <- build_design_new(ds, nd)

      eta <- as.numeric(glmnet::predict.glmnet(
        fit1, newx = Xn, s = fit_bundle$lambda_grid, ...
      ))
      p <- plogis(eta)
      p <- clip01(p, eps)
      return(matrix(if (type == "prob") p else log(p), nrow = n, ncol = 1L))
    }

    # Full bundle: nested fits by rhs then alpha; each path gives n x L for lambda_grid
    K <- length(fit_bundle$rhs_chr) * length(fit_bundle$alpha_grid) * length(fit_bundle$lambda_grid)
    out <- matrix(NA_real_, nrow = n, ncol = K)
    col_idx <- 1L

    for (rhs_raw in fit_bundle$rhs_chr) {
      ds <- fit_bundle$design_specs[[rhs_raw]]
      Xn <- build_design_new(ds, nd)

      for (alpha in fit_bundle$alpha_grid) {
        fit_ra <- fit_bundle$fits[[rhs_raw]][[as.character(alpha)]]

        eta <- as.matrix(glmnet::predict.glmnet(
          fit_ra, newx = Xn, s = fit_bundle$lambda_grid, ...
        ))
        pmat <- plogis(eta)
        pmat <- clip01(pmat, eps)
        if (type == "log_prob") pmat <- log(pmat)

        L <- ncol(pmat)
        out[, col_idx:(col_idx + L - 1L)] <- pmat
        col_idx <- col_idx + L
      }
    }

    out
  }

  list(
    method = "hurdle_glmnet",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      dat <- data.table::as.data.table(train_set)
      if (!(outcome_col %in% names(dat))) {
        stop("train_set must contain column '", outcome_col, "'.")
      }

      has_wts <- isTRUE(use_weights_col) && (weights_col %in% names(dat))
      wts <- if (has_wts) as.numeric(dat[[weights_col]]) else NULL
      y <- as.numeric(dat[[outcome_col]])

      design_specs <- setNames(vector("list", length(rhs_chr)), rhs_chr)
      fits <- setNames(vector("list", length(rhs_chr)), rhs_chr)

      for (rhs_raw in rhs_chr) {
        built <- build_design_train(rhs_raw, dat)
        X <- built$X
        design_specs[[rhs_raw]] <- built$design_spec

        fits_r <- setNames(vector("list", length(alpha_grid)), as.character(alpha_grid))
        for (alpha in alpha_grid) {
          fit_ra <- glmnet::glmnet(
            x = X, y = y, weights = wts,
            family = "binomial",
            alpha = alpha,
            lambda = lambda_grid,
            standardize = standardize,
            intercept = intercept,
            ...
          )
          fits_r[[as.character(alpha)]] <- strip_glmnet_fit(fit_ra)
        }
        fits[[rhs_raw]] <- fits_r
      }

      list(
        fits = fits,
        design_specs = design_specs,
        rhs_chr = rhs_chr,
        alpha_grid = alpha_grid,
        lambda_grid = lambda_grid,
        stripped = isTRUE(strip_fit),
        selected = FALSE
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

      rhs_raw  <- tune_grid$rhs[k]
      alpha_k  <- tune_grid$alpha[k]
      lambda_k <- as.numeric(tune_grid$lambda[k])

      has_wts <- isTRUE(use_weights_col) && (weights_col %in% names(dat))
      wts <- if (has_wts) as.numeric(dat[[weights_col]]) else NULL
      y <- as.numeric(dat[[outcome_col]])

      built <- build_design_train(rhs_raw, dat)
      X <- built$X
      ds <- built$design_spec

      fit_ra <- glmnet::glmnet(
        x = X, y = y, weights = wts,
        family = "binomial",
        alpha = alpha_k,
        lambda = lambda_grid,
        standardize = standardize,
        intercept = intercept,
        ...
      )

      list(
        fits = list(strip_glmnet_fit(fit_ra)),
        design_specs = setNames(list(ds), rhs_raw),
        rhs_chr = rhs_raw,
        alpha_grid = alpha_k,
        lambda_grid = lambda_k,
        stripped = isTRUE(strip_fit),
        selected = TRUE,
        tune = k
      )
    },

    select_fit = function(fit_bundle, tune) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      rhs_k   <- tune_grid$rhs[k]
      alpha_k <- tune_grid$alpha[k]
      lam_k   <- as.numeric(tune_grid$lambda[k])

      fit_ra <- fit_bundle$fits[[rhs_k]][[as.character(alpha_k)]]
      ds <- fit_bundle$design_specs[[rhs_k]]
      if (is.null(ds)) stop("Missing design_specs for RHS: ", rhs_k)

      list(
        fits = list(fit_ra),
        design_specs = setNames(list(ds), rhs_k),
        rhs_chr = rhs_k,
        alpha_grid = alpha_k,
        lambda_grid = lam_k,
        stripped = isTRUE(fit_bundle$stripped),
        selected = TRUE,
        tune = k
      )
    },

    # Required: log(pi(W)); fold_id accepted for contract, ignored here
    logpi = function(fit_bundle, newdata, fold_id = NULL, eps = 1e-15, ...) {
      predict_pi_matrix(fit_bundle, newdata, type = "log_prob", eps = eps, ...)
    },

    # Bernoulli negative log-likelihood for in_hurdle (useful for scoring)
    log_density = function(fit_bundle, newdata, eps = 1e-15, ...) {
      nd <- data.table::as.data.table(newdata)
      if (!(outcome_col %in% names(nd))) {
        stop("hurdle glmnet runner requires `", outcome_col, "` column in newdata")
      }
      y <- as.integer(nd[[outcome_col]])

      p_mat <- predict_pi_matrix(fit_bundle, nd, type = "prob", eps = eps, ...)
      n <- nrow(nd)
      K <- ncol(p_mat)

      out <- matrix(NA_real_, nrow = n, ncol = K)
      for (k in seq_len(K)) {
        p <- p_mat[, k]
        out[, k] <- -(y * log(p) + (1L - y) * log1p(-p))
      }
      out
    },

    # Sample in_hurdle ~ Bernoulli(pi(W)); returns n x n_samp matrix
    # Assumes post-selection K=1.
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, eps = 1e-15, ...) {
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)
      if (!is.null(seed)) set.seed(seed)

      p_mat <- predict_pi_matrix(fit_bundle, newdata, type = "prob", eps = eps, ...)
      if (ncol(p_mat) != 1L) {
        stop("sample() assumes K=1: fit_bundle must contain exactly one selected model.")
      }

      p <- as.numeric(p_mat[, 1L])
      n <- length(p)
      matrix(stats::rbinom(n * n_samp, size = 1L, prob = p), nrow = n, ncol = n_samp)
    }
  )
}

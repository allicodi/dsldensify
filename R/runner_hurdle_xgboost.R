#' Create an xgboost runner for hurdle probability modeling
#'
#' Constructs a runner (learner adapter) for modeling the hurdle probability
#' \eqn{\pi(W) = P(A = a_0 \mid W)} using gradient boosting via
#' xgboost::xgboost(). The runner is compatible with the hurdle workflow in
#' dsldensify, where the hurdle component is fit on wide data with binary
#' outcome \code{in_hurdle}.
#'
#' Tuning grid
#'
#' Tuning is performed over a grid defined by:
#'   - rhs_list (RHS varies slowest),
#'   - tree depth, learning rate, and regularization parameters,
#'   - subsampling and column subsampling parameters.
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All columns
#' referenced by rhs_list must already be numeric. Factors, characters, and
#' ordered factors are not supported and should be encoded upstream.
#'
#' Early stopping
#'
#' If early_stopping_rounds is provided, an internal validation split is
#' created within each fold. When valid_by_id is TRUE, the split is performed
#' by id_col; otherwise it is performed at the row level.
#'
#' @param rhs_list A list of RHS specifications, either one-sided formulas
#'   (for example, ~ W1 + W2) or character strings (for example, "W1 + W2").
#'
#' @param max_depth_grid Integer vector of maximum tree depths.
#'
#' @param eta_grid Numeric vector of learning rates.
#'
#' @param min_child_weight_grid Numeric vector of minimum child weights.
#'
#' @param subsample_grid Numeric vector of subsample proportions.
#'
#' @param colsample_bytree_grid Numeric vector of column subsample proportions.
#'
#' @param gamma_grid Numeric vector of minimum loss reduction.
#'
#' @param reg_lambda_grid Numeric vector of L2 regularization strengths.
#'
#' @param reg_alpha_grid Numeric vector of L1 regularization strengths.
#'
#' @param nrounds_max Integer maximum number of boosting rounds.
#'
#' @param early_stopping_rounds Integer number of early-stopping rounds; set to
#'   NULL to disable early stopping.
#'
#' @param valid_frac Fraction of observations (or ids) used for validation when
#'   early stopping is enabled.
#'
#' @param valid_by_id Logical. If TRUE, validation split is made by id_col.
#'
#' @param id_col Column name used when valid_by_id = TRUE.
#'
#' @param use_weights_col Logical. If TRUE and weights_col is present, weights
#'   are passed to xgboost::xgboost() via the weight argument.
#'
#' @param weights_col Name of the weights column in the wide data.
#'
#' @param objective Objective passed to xgboost::xgboost() (defaults to
#'   binary:logistic).
#'
#' @param eval_metric Evaluation metric passed to xgboost::xgboost().
#'
#' @param verbose Verbosity level for xgboost::xgboost().
#'
#' @param nthread Number of threads used by xgboost::xgboost().
#'
#' @param eps Numeric stability parameter for clipping probabilities.
#'
#' @param strip_fit Logical. If TRUE, attempt to reduce stored fit size.
#'
#' @param strip_method Method for stripping fit objects.
#'
#' @param seed Optional integer seed for deterministic fitting across tuning
#'   rows.
#'
#' @return A named list (runner) with the following elements:
#'   method: Character string "hurdle_xgboost".
#'   tune_grid: Data frame describing the tuning grid.
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
#'   - an optional weight column wts (or weights_col).
#'
#' @examples
#' rhs_list <- list(~ W1 + W2)
#'
#' runner <- make_xgboost_hurdle_runner(
#'   rhs_list = rhs_list,
#'   max_depth_grid = c(2L, 4L),
#'   eta_grid = c(0.05, 0.1)
#' )
#'
#' @export
make_xgboost_hurdle_runner <- function(
  rhs_list,

  max_depth_grid = c(2L, 4L),
  eta_grid = c(0.05, 0.1),
  min_child_weight_grid = c(1, 5),
  subsample_grid = 0.8,
  colsample_bytree_grid = 0.8,
  gamma_grid = 0,
  reg_lambda_grid = 1,
  reg_alpha_grid = 0,

  nrounds_max = 2000L,
  early_stopping_rounds = 30L,      # set NULL to disable early stopping
  valid_frac = 0.2,
  valid_by_id = FALSE,
  id_col = "obs_id",

  use_weights_col = TRUE,
  weights_col = "wts",

  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 0L,
  nthread = 0L,

  eps = 1e-8,
  strip_fit = TRUE,
  strip_method = c("none", "best_iter_refit"),

  seed = NULL
) {
  stopifnot(requireNamespace("xgboost", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))
  stopifnot(requireNamespace("stats", quietly = TRUE))

  strip_method <- match.arg(strip_method)

  # Fixed hurdle conventions (align with make_hurdle_glm_runner)
  outcome_col <- "in_hurdle"

  # RHS parsing: formulas (~ ...) or strings
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  # ---- tune grid (rhs varies first; last varies fastest) ----
  tune_grid <- expand.grid(
    rhs = rhs_chr,
    max_depth = as.integer(max_depth_grid),
    eta = eta_grid,
    min_child_weight = min_child_weight_grid,
    subsample = subsample_grid,
    colsample_bytree = colsample_bytree_grid,
    gamma = gamma_grid,
    reg_lambda = reg_lambda_grid,
    reg_alpha = reg_alpha_grid,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  tune_grid$.tune <- seq_len(nrow(tune_grid))

  rhs_to_cols <- function(rhs) setdiff(all.vars(stats::as.formula(paste0("~", rhs))), outcome_col)

  # ---- numeric-only enforcement (xgboost matrix) ----
  make_X <- function(dt, cols) {
    missing_cols <- setdiff(cols, names(dt))
    if (length(missing_cols)) stop("Missing columns in data: ", paste(missing_cols, collapse = ", "))
    X <- as.matrix(dt[, ..cols])
    if (!is.numeric(X)) stop("Selected features must be numeric. Encode factors upstream.")
    X
  }

  # ---- strict probability semantics ----
  clamp01_strict <- function(p, tol = 1e-6) {
    p <- as.numeric(p)
    if (any(!is.finite(p))) stop("Non-finite probabilities returned by xgboost.")
    if (any(p < -tol | p > 1 + tol)) {
      rg <- range(p, finite = TRUE)
      stop("xgboost returned values outside [0,1]. Range: [", rg[1], ", ", rg[2], "].")
    }
    p[p < 0] <- 0
    p[p > 1] <- 1
    p
  }

  clip01 <- function(p) pmin(pmax(p, eps), 1 - eps)

  # ---- internal validation split (mirrors hazard xgboost runner) ----
  split_train_valid <- function(train_dt) {
    n <- nrow(train_dt)

    if (is.null(early_stopping_rounds)) {
      return(list(idx_trn = rep(TRUE, n), idx_val = rep(FALSE, n)))
    }
    if (!is.finite(valid_frac) || valid_frac <= 0 || valid_frac >= 1) {
      stop("valid_frac must be in (0,1) when early_stopping_rounds is not NULL.")
    }

    if (isTRUE(valid_by_id)) {
      if (!(id_col %in% names(train_dt))) stop("valid_by_id=TRUE but id_col not found in train_set.")
      ids <- unique(train_dt[[id_col]])
      n_ids <- length(ids)
      n_val_ids <- max(1L, floor(valid_frac * n_ids))
      val_ids <- sample(ids, size = n_val_ids)
      idx_val <- train_dt[[id_col]] %in% val_ids
      idx_trn <- !idx_val
    } else {
      idx_val <- rep(FALSE, n)
      idx_val[sample.int(n, size = max(1L, floor(valid_frac * n)))] <- TRUE
      idx_trn <- !idx_val
    }

    if (!any(idx_val) || !any(idx_trn)) {
      idx_trn <- rep(TRUE, n)
      idx_val <- rep(FALSE, n)
    }
    list(idx_trn = idx_trn, idx_val = idx_val)
  }

  params_from_row <- function(tune_row) {
    params <- list(
      objective = objective,
      eval_metric = eval_metric,
      max_depth = as.integer(tune_row$max_depth),
      eta = tune_row$eta,
      min_child_weight = tune_row$min_child_weight,
      subsample = tune_row$subsample,
      colsample_bytree = tune_row$colsample_bytree,
      gamma = tune_row$gamma,
      lambda = tune_row$reg_lambda,
      alpha = tune_row$reg_alpha
    )
    if (!is.null(nthread) && isTRUE(nthread > 0)) params$nthread <- as.integer(nthread)
    params
  }

  make_dmatrix <- function(dt, cols) {
    if (!(outcome_col %in% names(dt))) stop("train_set must contain outcome column '", outcome_col, "'.")
    X <- make_X(dt, cols)
    y <- dt[[outcome_col]]

    if (is.logical(y)) y <- as.integer(y)
    y <- as.integer(y)
    if (anyNA(y) || any(!(y %in% c(0L, 1L)))) stop(outcome_col, " must be coded 0/1 with no NA.")

    w <- NULL
    if (isTRUE(use_weights_col)) {
      if (!(weights_col %in% names(dt))) stop("use_weights_col=TRUE but weights_col '", weights_col, "' not found.")
      w <- dt[[weights_col]]
      if (!is.numeric(w) || any(!is.finite(w)) || any(w < 0)) stop("weights must be nonnegative finite numeric.")
    }

    xgboost::xgb.DMatrix(data = X, label = y, weight = w)
  }

  fit_one_row <- function(train_dt, tune_row) {
    cols <- rhs_to_cols(tune_row$rhs)
    if (length(cols) < 1L) stop("RHS selects no columns: rhs='", tune_row$rhs, "'")

    # split (if early stopping enabled)
    sp <- split_train_valid(train_dt)
    dt_trn <- train_dt[sp$idx_trn]
    dt_val <- train_dt[sp$idx_val]

    dtrain <- make_dmatrix(dt_trn, cols)
    dvalid <- if (any(sp$idx_val)) make_dmatrix(dt_val, cols) else NULL

    params <- params_from_row(tune_row)
    use_es <- !is.null(early_stopping_rounds) && !is.null(dvalid)

    bst <- xgboost::xgb.train(
      params = params,
      data = dtrain,
      nrounds = as.integer(nrounds_max),
      watchlist = if (use_es) list(train = dtrain, eval = dvalid) else NULL,
      early_stopping_rounds = if (use_es) as.integer(early_stopping_rounds) else NULL,
      maximize = FALSE,
      verbose = as.integer(verbose)
    )

    best_iter <- if (!is.null(bst$best_iteration) && is.finite(bst$best_iteration)) {
      as.integer(bst$best_iteration)
    } else {
      as.integer(nrounds_max)
    }

    # strip by refitting to best_iter (mirrors hazard runner behavior)
    if (isTRUE(strip_fit) && strip_method == "best_iter_refit" && use_es && best_iter < as.integer(nrounds_max)) {
      bst <- xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = best_iter,
        watchlist = NULL,
        maximize = FALSE,
        verbose = 0L
      )
      best_iter <- NA_integer_
    }

    if (isTRUE(strip_fit)) {
      list(model = bst, best_iter = best_iter, cols = cols)
    } else {
      list(model = bst, best_iter = best_iter, cols = cols, params = params)
    }
  }

  # ---- shared probability scoring helper (used by logpi/log_density/sample) ----
  # Returns n x K matrix of probabilities, aligned with fits ordering.
  predict_pi <- function(fits, newdata, ...) {
    nd <- data.table::as.data.table(newdata)
    if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")

    n <- nrow(nd)
    out <- matrix(NA_real_, nrow = n, ncol = length(fits))

    for (k in seq_along(fits)) {
      cols <- fits[[k]]$cols
      X <- make_X(nd, cols)
      dnew <- xgboost::xgb.DMatrix(data = X)

      # If model was refit to best_iter, best_iter is NA and full model is correct.
      if (is.na(fits[[k]]$best_iter)) {
        p <- predict(fits[[k]]$model, dnew)
      } else {
        it <- as.integer(fits[[k]]$best_iter)
        p <- predict(fits[[k]]$model, dnew, iterationrange = c(1L, it))
      }

      p <- clamp01_strict(p)
      out[, k] <- clip01(p)
    }

    out
  }

  list(
    method = "hurdle_xgboost",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      dt <- data.table::as.data.table(train_set)

      if (!(outcome_col %in% names(dt))) stop("train_set must contain column '", outcome_col, "'.")
      if (isTRUE(use_weights_col) && !(weights_col %in% names(dt))) {
        stop("use_weights_col=TRUE but weights_col '", weights_col, "' not found in train_set.")
      }

      fits <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        tr <- tune_grid[k, , drop = FALSE]
        if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(tr$.tune))
        fits[[k]] <- fit_one_row(train_dt = dt, tune_row = tr)
      }

      list(
        fits = fits,
        rhs_chr = rhs_chr,
        tune = seq_len(nrow(tune_grid)),
        stripped = isTRUE(strip_fit),
        strip_method = strip_method
      )
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      dt <- data.table::as.data.table(train_set)

      tr <- tune_grid[k, , drop = FALSE]
      if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(tr$.tune))
      one <- fit_one_row(train_dt = dt, tune_row = tr)

      list(
        fits = list(one),
        rhs_chr = tr$rhs,
        tune = k,
        stripped = isTRUE(strip_fit),
        strip_method = strip_method
      )
    },

    # ---- hurdle convention: return predict_pi (no predict()) ----
    predict_pi = function(fit_bundle, newdata, ...) {
      predict_pi(fits = fit_bundle$fits, newdata = newdata, ...)
    },

    # log(pi(W)) as n x K
    logpi = function(fit_bundle, newdata, ...) {
      p <- predict_pi(fits = fit_bundle$fits, newdata = newdata, ...)
      log(p)
    },

    # Bernoulli negative log-likelihood as n x K (matches hurdle_glm_runner)
    log_density = function(fit_bundle, newdata, ...) {
      nd <- data.table::as.data.table(newdata)
      if (!(outcome_col %in% names(nd))) {
        stop("hurdle xgboost runner requires `", outcome_col, "` column in newdata")
      }

      y <- nd[[outcome_col]]
      if (is.logical(y)) y <- as.integer(y)
      y <- as.integer(y)
      if (anyNA(y) || any(!(y %in% c(0L, 1L)))) stop("`", outcome_col, "` must be coded 0/1 with no NA.")

      p <- predict_pi(fits = fit_bundle$fits, newdata = nd, ...)
      y_mat <- matrix(y, nrow = length(y), ncol = ncol(p))

      -(y_mat * log(p) + (1L - y_mat) * log1p(-p))
    },

    # Sample in_hurdle ~ Bernoulli(pi(W)); returns n x n_samp matrix
    # Assumes post-selection K=1.
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, ...) {
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)
      if (!is.null(seed)) set.seed(seed)

      p_mat <- predict_pi(fits = fit_bundle$fits, newdata = newdata, ...)
      if (ncol(p_mat) != 1L) stop("sample() assumes K=1: fit_bundle must contain exactly one selected model.")

      p <- as.numeric(p_mat[, 1L])
      n <- length(p)
      matrix(stats::rbinom(n * n_samp, size = 1L, prob = p), nrow = n, ncol = n_samp)
    }
  )
}

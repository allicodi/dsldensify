#' Create an xgboost runner for discrete-time hazard modeling
#'
#' @description
#' Constructs a **runner** (learner adapter) compatible with the
#' \code{run_grid_setting()} / \code{summarize_and_select()} workflow used in
#' \code{dsl_densify}. The runner fits gradient-boosted decision tree models via
#' \code{xgboost::xgb.train()} on long-format discrete-time hazard data
#' (binary outcome \code{in_bin}), and supports a tuning grid over:
#' \itemize{
#'   \item multiple RHS feature specifications (\code{rhs_list}),
#'   \item tree depth and learning-rate parameters,
#'   \item node- and split-regularization parameters.
#' }
#'
#' The fitted models estimate per-bin discrete-time hazards
#' \eqn{P(T \in \text{bin}_j \mid T \ge \text{bin}_j, W)} using a logistic loss,
#' which is equivalent to maximizing the discrete-time hazard likelihood under
#' the long-data construction used by \code{dsl_densify}.
#'
#' @section RHS specifications (column selection only):
#' The \code{rhs_list} argument follows the same *interface* as the GLM and
#' GLMNET runners, but is interpreted more restrictively:
#' \itemize{
#'   \item RHS formulas are used **only to select columns** from the data.
#'   \item No transformations, interactions, or spline terms are evaluated.
#'   \item The variable names extracted via \code{all.vars()} define the feature set.
#' }
#'
#' This design keeps the runner lightweight and delegates all feature
#' engineering (encoding, interactions, splines, etc.) to upstream code.
#'
#' By default, \code{bin_id} is automatically included in each RHS specification
#' unless \code{require_bin_id = FALSE}.
#'
#' @section Numeric-only requirement:
#' This runner operates directly on numeric feature matrices passed to
#' \code{xgboost}. All columns referenced in \code{rhs_list} **must already be
#' numeric** (including \code{bin_id} and all covariates in \code{W}).
#' Factors, characters, and ordered factors are not supported and should be
#' encoded upstream. No coercion is performed internally.
#'
#' @section Tuning grid and prediction layout:
#' The internal \code{tune_grid} is constructed using \code{expand.grid()} with
#' **RHS varying first**, followed by tree hyperparameters:
#' \itemize{
#'   \item \code{rhs},
#'   \item \code{max_depth},
#'   \item \code{eta},
#'   \item \code{min_child_weight},
#'   \item \code{subsample},
#'   \item \code{colsample_bytree},
#'   \item \code{gamma},
#'   \item \code{reg_lambda},
#'   \item \code{reg_alpha}.
#' }
#'
#' Each row of \code{tune_grid} corresponds to **exactly one fitted xgboost model**.
#' During cross-validation, \code{predict()} returns an
#' \code{n_long x K} matrix of predicted hazards, where
#' \code{K = nrow(tune_grid)}, with columns aligned to \code{.tune}.
#'
#' Internally, \code{predict()} delegates to a local helper \code{predict_hazards()}
#' to avoid duplicated scoring logic and to ensure sampling and prediction use
#' the same hazard predictions.
#'
#' @section Sampling from the fitted hazard model:
#' The runner provides a \code{sample()} method that generates draws
#' \eqn{A^* \sim \hat f(\cdot \mid W)} from the implied conditional density under
#' the discrete-time hazard representation.
#'
#' Sampling assumes the \code{fit_bundle} contains **exactly one** tuned fit
#' (i.e., \code{length(fit_bundle$fits) == 1}). This is the intended usage after
#' model selection (e.g., after applying \code{select_fit_tune()} or fitting the
#' selected tuning index via \code{fit_one()}).
#'
#' IMPORTANT: This runner's \code{sample()} expects \code{newdata} in **long hazard
#' format**. Expansion of wide \code{W} to long (repeating rows across all bins,
#' attaching \code{bin_lower}/\code{bin_upper}) is handled upstream by the
#' hazard-grid orchestration utilities in \code{dsldensify}.
#'
#' @section Early stopping:
#' If \code{early_stopping_rounds} is not \code{NULL}, each model is trained with
#' early stopping using an internal validation split drawn from the training data.
#' Validation splits may be constructed at the subject level (via \code{obs_id})
#' or at the row level.
#'
#' Early stopping uses standard logistic log loss on the hazard outcome
#' \code{in_bin}. No custom objective or evaluation function is required, as this
#' loss is already equivalent to the discrete-time hazard likelihood.
#'
#' @section Lightweight fit objects:
#' When \code{strip_fit = TRUE}, fitted models are reduced to a minimal
#' representation sufficient for prediction:
#' \itemize{
#'   \item the fitted \code{xgboost} booster (optionally refit to the
#'         early-stopping iteration),
#'   \item the selected feature column names.
#' }
#'
#' If \code{strip_method = "best_iter_refit"}, models trained with early stopping
#' are refit to exactly the selected number of boosting iterations, discarding
#' unused trees and substantially reducing memory usage.
#'
#' @param rhs_list A list of one-sided RHS formulas, such as
#' \code{list(~ bin_id + W1 + W2, ~ bin_id + W1 + W2 + W3)}. Variable names are
#' extracted using \code{all.vars()} and used solely for column selection.
#'
#' @param max_depth_grid Integer vector of tree depths to tune over.
#'
#' @param eta_grid Numeric vector of learning rates.
#'
#' @param min_child_weight_grid Numeric vector controlling minimum node weight.
#'
#' @param subsample_grid Numeric vector of row subsampling fractions.
#'
#' @param colsample_bytree_grid Numeric vector of column subsampling fractions.
#'
#' @param gamma_grid Numeric vector of minimum split-loss reduction values.
#'
#' @param reg_lambda_grid Numeric vector of L2 regularization parameters.
#'
#' @param reg_alpha_grid Numeric vector of L1 regularization parameters.
#'
#' @param nrounds_max Integer maximum number of boosting iterations.
#'
#' @param early_stopping_rounds Integer number of rounds with no improvement
#' before early stopping. Set to \code{NULL} to disable early stopping.
#'
#' @param valid_frac Fraction of training data used for internal validation
#' when early stopping is enabled.
#'
#' @param valid_by_id Logical. If \code{TRUE}, validation splits are constructed
#' at the subject level using \code{obs_id}.
#'
#' @param id_col Name of the subject identifier column used when
#' \code{valid_by_id = TRUE}.
#'
#' @param bin_var Name of the time-bin variable. Automatically added to RHS
#' specifications when \code{require_bin_id = TRUE}.
#'
#' @param require_bin_id Logical. If \code{TRUE} (default), ensure \code{bin_id}
#' is included in all RHS feature sets.
#'
#' @param use_weights_col Logical. If \code{TRUE} and a column named \code{wts}
#' (or \code{weights}) is present, it is passed to \code{xgboost} as case weights.
#'
#' @param objective Character string passed to \code{xgboost::xgb.train()}.
#' Defaults to \code{"binary:logistic"}.
#'
#' @param eval_metric Evaluation metric passed to \code{xgboost}. Defaults to
#' \code{"logloss"}.
#'
#' @param verbose Integer verbosity level passed to \code{xgboost}.
#'
#' @param nthread Integer number of threads used by \code{xgboost}.
#'
#' @param eps Numeric tolerance used to clip predicted hazards away from
#' \code{0} and \code{1}.
#'
#' @param strip_fit Logical. If \code{TRUE}, store a lightweight representation
#' of each fitted model.
#'
#' @param strip_method Method used when \code{strip_fit = TRUE}. Currently
#' supports \code{"none"} and \code{"best_iter_refit"}.
#'
#' @return A named list (runner) with elements:
#' \describe{
#'   \item{method}{Character string \code{"xgboost"}.}
#'   \item{tune_grid}{Data frame describing the tuning grid, including
#'         \code{.tune} and hyperparameter columns.}
#'   \item{fit}{Function \code{fit(train_set, ...)} returning a fit bundle.}
#'   \item{predict}{Function \code{predict(fit_bundle, newdata, ...)} returning
#'         an \code{n_long x K} matrix of hazard predictions.}
#'   \item{fit_one}{Function \code{fit_one(train_set, tune, ...)} fitting only
#'         the selected tuning index.}
#'   \item{sample}{Function \code{sample(fit_bundle, newdata, n_samp, ...)}
#'         drawing samples from the implied conditional density (assumes
#'         \code{length(fit_bundle$fits)==1}).}
#' }
#'
#' @details
#' ## Data requirements
#' The runner expects \code{train_set} and \code{newdata} as
#' \code{data.table}s in the **long hazard format** produced by
#' \code{format_long_hazards()}, including:
#' \itemize{
#'   \item a binary outcome column \code{in_bin},
#'   \item a time-bin column \code{bin_id},
#'   \item covariates referenced in \code{rhs_list},
#'   \item an optional weight column \code{wts}.
#' }
#'
#' \code{newdata} passed to \code{sample()} must additionally include
#' \code{bin_lower} and \code{bin_upper}.
#'
#' ## Model selection
#' Each row of \code{tune_grid} corresponds to a distinct fitted model, so no
#' \code{select_fit()} method is required. The \code{.tune} index uniquely
#' identifies the fitted xgboost model.
#'
#' @examples
#' rhs_list <- list(
#'   ~ bin_id + W1 + W2,
#'   ~ bin_id + W1 + W2 + W3
#' )
#'
#' runner <- make_xgboost_runner(
#'   rhs_list = rhs_list,
#'   max_depth_grid = c(2L, 4L),
#'   eta_grid = c(0.05, 0.1),
#'   strip_fit = TRUE
#' )
#'
#' @export
make_xgboost_runner <- function(
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
  valid_by_id = TRUE,
  id_col = "obs_id",
  bin_var = "bin_id",
  require_bin_id = TRUE,
  use_weights_col = TRUE,
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 0L,
  nthread = 0L,
  eps = 1e-8,
  strip_fit = TRUE,
  strip_method = c("none", "best_iter_refit")
) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("Package 'xgboost' is required.")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required because train_set/newdata are data.table.")
  }
  strip_method <- match.arg(strip_method)

  # ---- rhs parsing (column selection only) ----
  if (!(is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula")))) {
    stop("rhs_list must be a list of one-sided formulas, e.g., list(~ x1 + x2, ~ x1 + x2 + x3).")
  }

  rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  rhs_vars <- lapply(rhs_list, function(f) all.vars(f))

  if (require_bin_id) {
    rhs_vars <- lapply(rhs_vars, function(v) if (!(bin_var %in% v)) c(bin_var, v) else v)
  }

  # ---- tune grid (rhs major; last varies fastest) ----
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

  rhs_index_of <- function(rhs_string) {
    m <- match(rhs_string, rhs_chr)
    if (is.na(m)) stop("Internal error: rhs not found in rhs_list.")
    m
  }

  clip01 <- function(p) pmin(pmax(p, eps), 1 - eps)

  split_train_valid <- function(train_set) {
    n <- nrow(train_set)
    if (is.null(early_stopping_rounds)) {
      return(list(idx_trn = rep(TRUE, n), idx_val = rep(FALSE, n)))
    }
    if (valid_frac <= 0 || valid_frac >= 1) {
      stop("valid_frac must be in (0,1) when early_stopping_rounds is not NULL.")
    }

    if (valid_by_id) {
      if (!(id_col %in% names(train_set))) stop("valid_by_id=TRUE but id_col not found in train_set.")
      ids <- unique(train_set[[id_col]])
      n_ids <- length(ids)
      n_val_ids <- max(1L, floor(valid_frac * n_ids))
      val_ids <- sample(ids, size = n_val_ids)
      idx_val <- train_set[[id_col]] %in% val_ids
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

  # data.table-safe feature matrix extraction: df[, ..cols]
  make_X <- function(df, cols) {
    missing_cols <- setdiff(cols, names(df))
    if (length(missing_cols)) {
      stop("Missing columns in data: ", paste(missing_cols, collapse = ", "))
    }
    X <- as.matrix(df[, ..cols])
    # If upstream leaves non-numeric cols, fail fast (xgboost expects numeric)
    if (!is.numeric(X)) {
      stop("Selected features must be numeric. Encode factors upstream.")
    }
    X
  }

  make_dmatrix <- function(df, cols, y_col = "in_bin") {
    if (!(y_col %in% names(df))) stop("train_set must contain outcome column '", y_col, "'.")

    X <- make_X(df, cols)
    y <- df[[y_col]]

    w <- NULL
    if (use_weights_col) {
      if ("wts" %in% names(df)) w <- df[["wts"]]
      else if ("weights" %in% names(df)) w <- df[["weights"]]
    }

    xgboost::xgb.DMatrix(data = X, label = y, weight = w)
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
    if (!is.null(nthread) && nthread > 0) params$nthread <- as.integer(nthread)
    params
  }

  fit_one_tune <- function(train_set, tune_row) {
    rhs_idx <- rhs_index_of(tune_row$rhs)
    cols <- rhs_vars[[rhs_idx]]

    sp <- split_train_valid(train_set)
    df_trn <- train_set[sp$idx_trn, ]
    df_val <- train_set[sp$idx_val, ]

    dtrain <- make_dmatrix(df_trn, cols = cols)
    watch <- list(train = dtrain)

    dvalid <- NULL
    if (any(sp$idx_val)) {
      dvalid <- make_dmatrix(df_val, cols = cols)
      watch$eval <- dvalid
    }

    params <- params_from_row(tune_row)

    use_es <- !is.null(early_stopping_rounds) && !is.null(dvalid)

    fit <- xgboost::xgb.train(
      params = params,
      data = dtrain,
      nrounds = as.integer(nrounds_max),
      evals = if (use_es) watch else NULL,
      early_stopping_rounds = if (use_es) as.integer(early_stopping_rounds) else NULL,
      maximize = FALSE,
      verbose = verbose
    )

    num_rounds <- xgboost::xgb.get.num.boosted.rounds(fit)

    best_iter <- NA_integer_

    # If early stopping was used, try to retrieve best_iteration robustly.
    # Note: xgb.attr() best_iteration is documented as 0-based; R attribute is base-1.
    if (use_es) {
      bi <- suppressWarnings(as.integer(xgboost::xgb.attr(fit, "best_iteration")))
      if (!is.na(bi)) {
        best_iter <- bi + 1L
      } else {
        bi2 <- suppressWarnings(as.integer(attr(fit, "best_iteration")))
        if (!is.na(bi2)) best_iter <- bi2
      }
    }
    # cap to what the model actually contains
    if (!is.na(best_iter)) best_iter <- min(best_iter, num_rounds)
    # If we still don't have a usable best_iter, just use the full model
    if (is.na(best_iter) || best_iter < 1L) best_iter <- NA_integer_


    # ---- strip: refit to best_iter to drop unnecessary trees ----
    if (strip_fit && strip_method == "best_iter_refit" && use_es && !is.na(best_iter) && best_iter < as.integer(nrounds_max)) {
      # refit exactly best_iter (no watchlist / no early stopping) => smaller model
      fit <- xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = best_iter,
        evals = NULL,
        maximize = FALSE,
        verbose = 0
      )
      best_iter <- NA_integer_  # no longer needed
    }

    if (strip_fit) {
      # keep only what we need to predict
      list(model = fit, best_iter = best_iter, cols = cols)
    } else {
      # keep extras for debugging/repro (still avoid holding dtrain/dvalid explicitly)
      list(model = fit, best_iter = best_iter, cols = cols, params = params)
    }
  }

  # ---- shared scoring helper (used by predict() and sample()) ----
  predict_hazards <- function(fits, newdata, ...) {
    if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table.")
    if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")

    n <- nrow(newdata)
    out <- matrix(NA_real_, nrow = n, ncol = length(fits))

    for (k in seq_along(fits)) {
      cols <- fits[[k]]$cols
      X <- make_X(newdata, cols)
      dnew <- xgboost::xgb.DMatrix(data = X)

      if (is.na(fits[[k]]$best_iter)) {
        # stripped by refit => full model is already best_iter length
        p <- predict(fits[[k]]$model, dnew)
      } else {
        it <- as.integer(fits[[k]]$best_iter)
        nr <- xgboost::xgb.get.num.boosted.rounds(fits[[k]]$model)
        it <- max(1L, min(it, nr))
        p <- predict(fits[[k]]$model, dnew, iterationrange = c(1L, it))
      }

      out[, k] <- clip01(p)
    }

    out
  }

  list(
    method = "xgboost",
    tune_grid = tune_grid,
    positive_support = TRUE,
    
    fit = function(train_set, ...) {
      if (!data.table::is.data.table(train_set)) stop("train_set must be a data.table.")

      fits <- vector("list", nrow(tune_grid))

      # Efficiency win: rhs-major loop so we can reuse the train/valid split per rhs
      # (still allows fair comparison across hyperparams for a given rhs)
      rhs_levels <- unique(tune_grid$rhs)
      for (rhs_str in rhs_levels) {
        idx_rows <- which(tune_grid$rhs == rhs_str)

        # Compute split ONCE per rhs within this fit() call
        sp <- split_train_valid(train_set)
        df_trn <- train_set[sp$idx_trn, ]
        df_val <- train_set[sp$idx_val, ]

        rhs_idx <- rhs_index_of(rhs_str)
        cols <- rhs_vars[[rhs_idx]]

        dtrain <- make_dmatrix(df_trn, cols = cols)
        dvalid <- if (any(sp$idx_val)) make_dmatrix(df_val, cols = cols) else NULL
        use_es_rhs <- !is.null(early_stopping_rounds) && !is.null(dvalid)

        for (i in idx_rows) {
          tune_row <- tune_grid[i, , drop = FALSE]
          params <- params_from_row(tune_row)

          fit <- xgboost::xgb.train(
            params = params,
            data = dtrain,
            nrounds = as.integer(nrounds_max),
            evals = if (use_es_rhs) list(train = dtrain, eval = dvalid) else NULL,
            early_stopping_rounds = if (use_es_rhs) as.integer(early_stopping_rounds) else NULL,
            maximize = FALSE,
            verbose = verbose
          )

          num_rounds <- xgboost::xgb.get.num.boosted.rounds(fit)

          best_iter <- NA_integer_

          # If early stopping was used, try to retrieve best_iteration robustly.
          # Note: xgb.attr() best_iteration is documented as 0-based; R attribute is base-1.
          if (use_es_rhs) {
            bi <- suppressWarnings(as.integer(xgboost::xgb.attr(fit, "best_iteration")))
            if (!is.na(bi)) {
              best_iter <- bi + 1L
            } else {
              bi2 <- suppressWarnings(as.integer(attr(fit, "best_iteration")))
              if (!is.na(bi2)) best_iter <- bi2
            }
          }
          # cap to what the model actually contains
          if (!is.na(best_iter)) best_iter <- min(best_iter, num_rounds)
          # If we still don't have a usable best_iter, just use the full model
          if (is.na(best_iter) || best_iter < 1L) best_iter <- NA_integer_


          if (strip_fit && strip_method == "best_iter_refit" && use_es_rhs && !is.na(best_iter) && best_iter < num_rounds) {
            fit <- xgboost::xgb.train(
              params = params,
              data = dtrain,
              nrounds = best_iter,
              evals = NULL,
              maximize = FALSE,
              verbose = 0
            )
            best_iter <- NA_integer_
          }

          fits[[i]] <- if (strip_fit) {
            list(model = fit, best_iter = best_iter, cols = cols)
          } else {
            list(model = fit, best_iter = best_iter, cols = cols, params = params)
          }
        }
      }

      list(fits = fits)
    },

    # NOTE: newdata is LONG hazard data (expanded upstream from wide W):
    # dsldensify's hazard-grid utilities handle repeating W across bins and
    # attaching bin metadata (bin_id, bin_lower, bin_upper, etc.).
    predict = function(fit_bundle, newdata, ...) {
      predict_hazards(fits = fit_bundle$fits, newdata = newdata, ...)
    },

    # NOTE: newdata is LONG hazard data (expanded upstream from wide W):
    # sampling requires hazards across all bins per subject, so the wide->long
    # expansion (and bin endpoint attachment) is handled upstream. This method
    # assumes fit_bundle contains exactly one selected fit (K = 1).
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, ...) {
      if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table.")
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      if (length(fits) != 1L) stop("sample() assumes K=1: fit_bundle$fits must have length 1 (selected model).")

      if (!("bin_lower" %in% names(newdata)) || !("bin_upper" %in% names(newdata))) {
        stop("newdata must contain 'bin_lower' and 'bin_upper' columns for sampling.")
      }
      if (!(id_col %in% names(newdata))) {
        stop("newdata must contain id_col='", id_col, "' for sampling.")
      }
      if (!(bin_var %in% names(newdata))) {
        stop("newdata must contain bin_var='", bin_var, "' for sampling.")
      }

      if (!is.null(seed)) set.seed(seed)

      # Predict hazards (n_long x 1). Hazards must be ordered consistently with dt.
      haz <- predict_hazards(fits = fits, newdata = newdata, ...)
      haz <- as.numeric(haz[, 1])

      dt <- data.table::as.data.table(newdata)
      dt[, .row_id__ := .I]
      data.table::setorderv(dt, c(id_col, bin_var))
      haz <- haz[dt$.row_id__]  # reorder hazards to match dt after sorting

      # per-subject row indices (already in bin order after setorderv)
      rows_by_id <- split(seq_len(nrow(dt)), dt[[id_col]])
      ids <- names(rows_by_id)

      out <- matrix(NA_real_, nrow = length(ids), ncol = n_samp)
      rownames(out) <- ids
      warned_zero_mass <- FALSE

      for (ii in seq_along(ids)) {
        rr <- rows_by_id[[ii]]
        m <- length(rr)
        if (m < 1L) next

        h <- clip01(haz[rr])

        lower <- dt$bin_lower[rr]
        upper <- dt$bin_upper[rr]
        if (any(!is.finite(lower)) || any(!is.finite(upper)) || any(upper <= lower)) {
          stop("Invalid bin_lower/bin_upper for id='", ids[[ii]], "'.")
        }

        # masses p_j = h_j * prod_{l<j}(1 - h_l)
        if (m == 1L) {
          mass <- h
        } else {
          s_prev <- c(1, cumprod(1 - h)[-m])
          mass <- h * s_prev
        }

        tot <- sum(mass)
        if (!is.finite(tot) || tot <= 0) {

          if (!warned_zero_mass) {
            warning(
              "Hazard-based sampling encountered zero or non-finite total mass for at least one observation.\n",
              "Falling back to uniform sampling over bins for those cases.\n",
              "This can occur if predicted hazards are numerically near zero across all bins\n",
              "or if survival past the grid has probability ~1."
            )
            warned_zero_mass <- TRUE
          }

          # fallback: uniform over bins
          j <- sample.int(m, size = n_samp, replace = TRUE)

        } else {
          pmf <- mass / tot
          j <- sample.int(m, size = n_samp, replace = TRUE, prob = pmf)
        }

        out[ii, ] <- stats::runif(n_samp, min = lower[j], max = upper[j])
      }

      out
    },

    fit_one = function(train_set, tune, ...) {
      if (!data.table::is.data.table(train_set)) stop("train_set must be a data.table.")
      if (length(tune) != 1L || is.na(tune) || tune < 1L || tune > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..nrow(tune_grid).")
      }
      one <- fit_one_tune(train_set, tune_grid[tune, , drop = FALSE])
      # Wrap as list-of-fits so predict() returns a 1-column matrix.
      list(fits = list(one))
    }
  )
}

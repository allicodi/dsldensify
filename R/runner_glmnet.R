#' Create a glmnet runner for discrete-time hazard modeling
#'
#' @description
#' Constructs a **runner** (learner adapter) compatible with the
#' \code{run_grid_setting()} / \code{summarize_and_select()} workflow used in
#' dsl_densify. The runner fits penalized logistic regression models via
#' \code{glmnet::glmnet()} on long-format discrete-time hazard data
#' (binary outcome \code{in_bin}), and supports a tuning grid over:
#' \itemize{
#'   \item multiple RHS model specifications (\code{rhs_list}),
#'   \item elastic net mixing parameters (\code{alpha_grid}),
#'   \item a fixed penalty grid (\code{lambda_grid}).
#' }
#'
#' This runner supports natural spline terms specified directly in the RHS
#' formulas using \code{ns()} or \code{splines::ns()}, under a restricted but
#' relatively robust spline policy:
#' \itemize{
#'   \item Only \code{ns(<symbol>, df = ...)} is supported (the first argument must
#'         be a bare variable name, i.e., not \code{ns(log(variable))}).
#'   \item \code{splines::ns()} is accepted but internally is rewritten to \code{ns()}.
#'   \item Knot locations are computed on the training data and then frozen for
#'         prediction by evaluating \code{model.matrix()} in an environment where
#'         \code{ns()} is replaced by a wrapper that injects the stored knots and
#'         boundary knots. This ensures reproducible behavior across training and 
#'         validation folds
#' }
#'
#' @section Numeric-only requirement:
#' The design matrix is constructed using \code{stats::model.matrix()}.
#' To ensure stable feature definitions across folds and avoid factor-level
#' bookkeeping, this runner is intended for use with **numeric predictors only**
#' (including \code{bin_id} and all covariates in \code{W}). RHS formulas should
#' not include factors, characters, or ordered factors. This function does not
#' coerce variables to numeric; it assumes inputs are already in numeric form.
#'
#' @section Tuning grid and prediction layout:
#' The internal \code{tune_grid} is ordered such that:
#' \itemize{
#'   \item RHS varies first,
#'   \item then \code{alpha},
#'   \item then \code{lambda}.
#' }
#' During cross-validation, \code{predict()} returns an
#' \code{n_long x K} matrix of predicted hazards, where
#' \code{K = length(rhs_list) * length(alpha_grid) * length(lambda_grid)},
#' with columns aligned to \code{tune_grid}.
#'
#' @section Lightweight fit objects:
#' When \code{strip_fit = TRUE}, each fitted \code{(rhs, alpha)} block is reduced
#' to a minimal representation sufficient for prediction:
#' \itemize{
#'   \item an intercept vector \code{a0} of length \code{length(lambda_grid)},
#'   \item a sparse coefficient matrix \code{beta}
#'         (\code{Matrix::dgCMatrix}, dimensions \code{p x L}),
#'   \item the training column names \code{x_cols}.
#' }
#' This saves memory relative to storing full \code{glmnet} fits
#'
#' @param rhs_list A list of RHS specifications, either:
#' \itemize{
#'   \item one-sided formulas such as \code{~ W1 + ns(bin_id, df = 5)}, or
#'   \item character strings such as \code{"W1 + splines::ns(bin_id, df = 5)"}.
#' }
#' These RHS are used to construct \code{in_bin ~ <rhs>} internally.
#'
#' @param alpha_grid Numeric vector of elastic net mixing parameters passed to
#' \code{glmnet::glmnet(alpha = ...)}. Typical values lie in \code{[0, 1]}, where
#' \code{alpha = 1} corresponds to the lasso and \code{alpha = 0} to ridge
#' regression.
#'
#' @param lambda_grid Numeric vector of strictly positive penalty values.
#' This grid is treated as fixed and is reused across folds and grid settings
#' to ensure tuning alignment. Must have length at least 2. Users should verify
#' that the grid passed in is diverse enough. If tuning parameters at the edge
#' of the grid are being selected by CV, then it could be an indication that a 
#' broader grid is required.
#'
#' @param use_weights_col Logical. If \code{TRUE} and the training data contain a
#' column named \code{wts}, it is passed as \code{weights = ...} to
#' \code{glmnet::glmnet()}. Otherwise, fitting is unweighted.
#'
#' @param standardize Logical. Passed directly to \code{glmnet::glmnet(standardize = ...)}.
#'
#' @param intercept Logical. Passed directly to \code{glmnet::glmnet(intercept = ...)}.
#'
#' @param strip_fit Logical. If \code{TRUE} (default), store a lightweight fit
#' representation (intercept and coefficient matrix only) rather than the full
#' \code{glmnet} object.
#'
#' @return A named list (runner) with elements:
#' \describe{
#'   \item{method}{Character string \code{"glmnet"}.}
#'   \item{tune_grid}{Data frame with columns \code{.tune}, \code{rhs},
#'         \code{alpha}, and \code{lambda}.}
#'   \item{fit}{Function \code{fit(train_set, ...)} returning a fit bundle.}
#'   \item{predict}{Function \code{predict(fit_bundle, newdata, ...)} returning
#'         an \code{n_long x K} matrix of hazard predictions.}
#'   \item{fit_one}{Function \code{fit_one(train_set, tune, ...)} fitting only
#'         the selected tuning index.}
#' }
#'
#' @details
#' ## Data requirements
#' The runner expects \code{train_set} and \code{newdata} in the **long hazard
#' format** produced by \code{format_long_hazards()}, including:
#' \itemize{
#'   \item a binary outcome column \code{in_bin},
#'   \item covariates referenced in \code{rhs_list},
#'   \item an optional \code{wts} column of observation weights.
#' }
#'
#' ## Spline handling
#' Spline knots are computed once per fold (inside \code{fit()}) using the
#' training data and stored in \code{design_specs}. Prediction uses the stored
#' knots to ensure consistent spline bases across training and validation data
#' within each fold.
#'
#' ## Interactions
#' Interactions between spline terms and other covariates (for example,
#' \code{W1 * ns(bin_id, df = 5)}) are supported and handled via
#' \code{model.matrix()}. Nested spline constructs are not supported.
#'
#' @examples
#' rhs_list <- list(
#'   ~ W1 + W2 + splines::ns(bin_id, df = 4),
#'   ~ (W1 + W2) * splines::ns(bin_id, df = 4)
#' )
#'
#' runner <- make_glmnet_runner(
#'   rhs_list = rhs_list,
#'   alpha_grid = c(0.5, 1),
#'   lambda_grid = exp(seq(log(1e-4), log(10), length.out = 50)),
#'   strip_fit = TRUE
#' )
#'
#' @export

make_glmnet_runner <- function(
  rhs_list,
  alpha_grid,
  lambda_grid,                 # REQUIRED fixed grid
  use_weights_col = TRUE,
  standardize = TRUE,
  intercept = TRUE,
  strip_fit = TRUE             # NEW: store only coef/intercept for prediction
) {
  stopifnot(requireNamespace("glmnet", quietly = TRUE))
  stopifnot(requireNamespace("splines", quietly = TRUE))
  stopifnot(requireNamespace("Matrix", quietly = TRUE))

  # Accept RHS formulas (~ ...) or RHS strings
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
    rhs = rhs_chr,
    alpha = alpha_grid,
    lambda = lambda_grid,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  tune_grid$.tune <- seq_len(nrow(tune_grid))

  # --- helpers -------------------------------------------------------------

  normalize_rhs <- function(rhs) gsub("splines::ns", "ns", rhs, fixed = TRUE)

  find_ns_calls <- function(expr) {
    out <- list()
    rec <- function(e) {
      if (is.call(e)) {
        fn <- e[[1L]]
        if (is.symbol(fn) && as.character(fn) == "ns") out[[length(out) + 1L]] <<- e
        for (j in seq_along(e)[-1L]) rec(e[[j]])
      }
    }
    rec(expr)
    out
  }


  compute_spline_specs <- function(rhs, train_set) {
    f <- stats::as.formula(paste0("in_bin ~ ", rhs))
    rhs_expr <- f[[3L]]
    calls <- find_ns_calls(rhs_expr)

    specs <- list()

    for (cl in calls) {
      # only ns(<symbol>, df=...) supported
      if (length(cl) < 2L || !is.symbol(cl[[2L]])) {
        stop("Only ns(<variable>, df=...) supported. Found: ", paste(deparse(cl), collapse = ""))
      }
      xvar <- as.character(cl[[2L]])
      if (!(xvar %in% names(train_set))) stop("Spline variable '", xvar, "' not found in data for RHS: ", rhs)
      if (!is.numeric(train_set[[xvar]])) stop("Spline variable '", xvar, "' must be numeric.")

      argn <- names(as.list(cl))
      if (!("df" %in% argn)) stop("ns(", xvar, ", ...) must specify df= for RHS: ", rhs)
      df <- eval(cl[["df"]], envir = train_set, enclos = parent.frame())

      intercept_flag <- FALSE
      if ("intercept" %in% argn) {
        intercept_flag <- isTRUE(eval(cl[["intercept"]], envir = train_set, enclos = parent.frame()))
      }

      key <- paste(xvar, df, as.integer(intercept_flag), sep = "|")
      if (is.null(specs[[key]])) {
        B <- splines::ns(train_set[[xvar]], df = df, intercept = intercept_flag)
        specs[[key]] <- list(
          xvar = xvar,
          df = df,
          intercept = intercept_flag,
          knots = attr(B, "knots"),
          Boundary.knots = attr(B, "Boundary.knots")
        )
      }
    }

    specs
  }

  make_ns_env <- function(specs, parent_env) {
    env <- new.env(parent = parent_env)

    env$ns <- function(x, df = NULL, intercept = FALSE, ...) {
      x_name <- deparse(substitute(x))  # bare var name (enforced)
      key <- paste(x_name, df, as.integer(isTRUE(intercept)), sep = "|")
      sp <- specs[[key]]
      if (is.null(sp)) {
        stop("No frozen spline spec found for ns(", x_name, ", df=", df,
             ", intercept=", intercept, ").")
      }
      splines::ns(
        x,
        df = sp$df,
        knots = sp$knots,
        Boundary.knots = sp$Boundary.knots,
        intercept = sp$intercept,
        ...
      )
    }

    env
  }

  build_design_train <- function(rhs_raw, train_set) {
    rhs <- normalize_rhs(rhs_raw)
    f <- stats::as.formula(paste0("in_bin ~ ", rhs))

    specs <- compute_spline_specs(rhs, train_set)

    env <- make_ns_env(specs, parent_env = environment(f))
    environment(f) <- env

    tt <- stats::terms(f, data = train_set)
    X <- stats::model.matrix(tt, data = train_set)
    if ("(Intercept)" %in% colnames(X)) X <- X[, colnames(X) != "(Intercept)", drop = FALSE]

    list(
      X = X,
      design_spec = list(rhs = rhs, specs = specs, x_cols = colnames(X))
    )
  }

  build_design_new <- function(design_spec, newdata) {
    f <- stats::as.formula(paste0("in_bin ~ ", design_spec$rhs))

    env <- make_ns_env(design_spec$specs, parent_env = environment(f))
    environment(f) <- env

    tt <- stats::terms(f, data = newdata)
    Xn <- stats::model.matrix(tt, data = newdata)
    if ("(Intercept)" %in% colnames(Xn)) Xn <- Xn[, colnames(Xn) != "(Intercept)", drop = FALSE]

    x_cols <- design_spec$x_cols
    missing <- setdiff(x_cols, colnames(Xn))
    if (length(missing)) {
      Xn <- cbind(Xn, matrix(0, nrow(Xn), length(missing), dimnames = list(NULL, missing)))
    }
    Xn <- Xn[, x_cols, drop = FALSE]
    Xn
  }

  # Strip a glmnet block fit down to intercept + coefficient matrix (sparse) for lambda_grid
  strip_glmnet_block <- function(fit_ra, x_cols, lambda_grid) {
    B <- glmnet::coef.glmnet(fit_ra, s = lambda_grid)  # (p+1) x L sparse
    rn <- rownames(B)

    a0_mat <- B[rn == "(Intercept)", , drop = FALSE]
    if (nrow(a0_mat) != 1L) stop("Could not extract intercept row from glmnet coef().")

    a0 <- as.numeric(a0_mat)

    idx <- match(x_cols, rn)
    beta <- B[idx, , drop = FALSE]
    miss <- is.na(idx)
    if (any(miss)) beta[miss, ] <- 0
    rownames(beta) <- x_cols

    out <- list(a0 = a0, beta = beta, x_cols = x_cols, lambda = lambda_grid)
    class(out) <- "glmnet_stripped"
    out
  }


  predict_stripped_block <- function(obj, Xnew) {
    eta <- sweep(as.matrix(Xnew %*% obj$beta), 2L, obj$a0, FUN = "+")
    plogis(eta)
  }

  is_stripped <- function(obj) is.list(obj) && !is.null(obj$beta) && !is.null(obj$a0)

  # --- runner --------------------------------------------------------------

  list(
    method = "glmnet",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      if (!("in_bin" %in% names(train_set))) stop("train_set must contain column 'in_bin'.")
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) as.numeric(train_set$wts) else NULL
      y <- as.numeric(train_set$in_bin)

      design_specs <- setNames(vector("list", length(rhs_chr)), rhs_chr)
      fits <- setNames(vector("list", length(rhs_chr)), rhs_chr)

      for (rhs_raw in rhs_chr) {
        built <- build_design_train(rhs_raw, train_set)
        X <- built$X
        design_specs[[rhs_raw]] <- built$design_spec

        fits_r <- setNames(vector("list", length(alpha_grid)), as.character(alpha_grid))
        for (alpha in alpha_grid) {
          fit_ra <- glmnet::glmnet(
            x = X, y = y, weights = wts_vec,
            family = "binomial",
            alpha = alpha,
            lambda = lambda_grid,
            standardize = standardize,
            intercept = intercept,
            ...
          )

          if (strip_fit) fit_ra <- strip_glmnet_block(fit_ra, x_cols = built$design_spec$x_cols, lambda_grid = lambda_grid)
          
          fits_r[[as.character(alpha)]] <- fit_ra
        }
        fits[[rhs_raw]] <- fits_r
      }

      list(
        fits = fits,
        design_specs = design_specs,
        rhs_chr = rhs_chr,
        alpha_grid = alpha_grid,
        lambda_grid = lambda_grid,
        stripped = isTRUE(strip_fit)
      )
    },

    predict = function(fit_bundle, newdata, ...) {
      nd <- as.data.frame(newdata)

      preds_blocks <- vector("list", length(fit_bundle$rhs_chr) * length(fit_bundle$alpha_grid))
      idx <- 1L

      for (rhs_raw in fit_bundle$rhs_chr) {
        Xn <- build_design_new(fit_bundle$design_specs[[rhs_raw]], nd)

        for (alpha in fit_bundle$alpha_grid) {
          fit_ra <- fit_bundle$fits[[rhs_raw]][[as.character(alpha)]]
          if (inherits(fit_ra, "glmnet_stripped")) {
            eta <- sweep(as.matrix(Xn %*% fit_ra$beta), 2L, fit_ra$a0, FUN = "+")
            p <- plogis(eta)
          } else if (inherits(fit_ra, "glmnet")) {
            p <- as.matrix(glmnet::predict.glmnet(fit_ra, newx = Xn, type = "response", ...))
          } else {
            stop("Unexpected fit object for glmnet runner: class = ", paste(class(fit_ra), collapse = ", "))
          }
          preds_blocks[[idx]] <- p
          idx <- idx + 1L
        }
      }

      do.call(cbind, preds_blocks)
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      rhs_raw   <- tune_grid$rhs[k]
      alpha_k   <- tune_grid$alpha[k]
      lambda_k  <- as.numeric(tune_grid$lambda[k])

      if (!("in_bin" %in% names(train_set))) stop("train_set must contain column 'in_bin'.")
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) as.numeric(train_set$wts) else NULL
      y <- as.numeric(train_set$in_bin)

      # Build design on this train_set and freeze spline specs for later prediction
      built <- build_design_train(rhs_raw, train_set)
      X <- built$X
      design_spec <- built$design_spec

      # Fit the full path on the fixed lambda_grid (needed so lambda_k is meaningful)
      fit_ra <- glmnet::glmnet(
        x = X, y = y, weights = wts_vec,
        family = "binomial",
        alpha = alpha_k,
        lambda = lambda_grid,
        standardize = standardize,
        intercept = intercept,
        ...
      )

      # Reduce to the single selected lambda
      if (strip_fit) {
        # Get coefficient matrix at the single lambda (p+1) x 1
        B <- glmnet::coef.glmnet(fit_ra, s = lambda_k)
        rn <- rownames(B)

        a0_mat <- B[rn == "(Intercept)", , drop = FALSE]
        if (nrow(a0_mat) != 1L) stop("Could not extract intercept row from glmnet coef().")
        a0 <- as.numeric(a0_mat)  # length 1

        idx <- match(design_spec$x_cols, rn)
        beta <- B[idx, , drop = FALSE]
        miss <- is.na(idx)
        if (any(miss)) beta[miss, ] <- 0
        rownames(beta) <- design_spec$x_cols

        fit_sel <- list(a0 = a0, beta = beta, x_cols = design_spec$x_cols, lambda = lambda_k)
        class(fit_sel) <- "glmnet_stripped"
      } else {
        # Keep the full glmnet object but record which lambda we want at predict time
        fit_sel <- fit_ra
        attr(fit_sel, "lambda_selected") <- lambda_k
      }

      # Return a minimal bundle compatible with predict():
      # - predict() will produce ONE column because rhs_chr and alpha_grid are length-1,
      #   and the stripped object stores a single lambda.
      list(
        fits = setNames(
          list(setNames(list(fit_sel), as.character(alpha_k))),
          rhs_raw
        ),
        design_specs = setNames(list(design_spec), rhs_raw),
        rhs_chr = rhs_raw,
        alpha_grid = alpha_k,
        lambda_grid = lambda_k,
        stripped = isTRUE(strip_fit),
        tune = k
      )
    },

    select_fit <- function(fit_bundle, tune) {
      k <- as.integer(tune)
      rhs_k   <- tune_grid$rhs[k]
      alpha_k <- tune_grid$alpha[k]

      fit_sel <- fit_bundle$fits[[rhs_k]][[as.character(alpha_k)]]
      list(
        fits = list(setNames(list(setNames(list(fit_sel), as.character(alpha_k))), rhs_k)),
        design_specs = setNames(list(fit_bundle$design_specs[[rhs_k]]), rhs_k),
        rhs_chr = rhs_k,
        alpha_grid = alpha_k,
        lambda_grid = fit_bundle$lambda_grid,
        stripped = isTRUE(fit_bundle$stripped),
        tune = k
      )
    }
  )
}

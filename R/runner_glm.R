#' Create a GLM runner for discrete-time hazard modeling (with frozen spline bases)
#'
#' @description
#' Constructs a **runner** (learner adapter) compatible with the
#' \code{run_grid_setting()} / \code{summarize_and_select()} workflow used in
#' dsl_densify. The runner fits one or more logistic regression models via
#' \code{stats::glm.fit()} on long-format discrete-time hazard data with binary
#' outcome \code{in_bin}. Tuning is performed over a list of RHS model
#' specifications (\code{rhs_list}); one model is fit per RHS.
#'
#' This GLM runner supports natural spline terms specified directly in the RHS
#' using \code{ns()} or \code{splines::ns()}, under a restricted but reproducible
#' spline policy that avoids storing large formula environments:
#' \itemize{
#'   \item Only \code{ns(<symbol>, df = ...)} is supported (the first argument must
#'         be a bare variable name).
#'   \item \code{splines::ns()} is accepted but internally rewritten to \code{ns()}.
#'   \item Knot locations are computed on the training data and then frozen for
#'         prediction by evaluating \code{model.matrix()} in an environment where
#'         \code{ns()} is replaced by a wrapper that injects stored knots and
#'         boundary knots. This ensures consistent spline bases within each
#'         fold and between training and validation data.
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
#' @section Tuning and prediction layout:
#' The runner creates a \code{tune_grid} with one row per RHS, with columns
#' \code{.tune} and \code{rhs}. During cross-validation, \code{predict()} returns
#' an \code{n_long x K} matrix of predicted hazards, where \code{K = length(rhs_list)}.
#' Columns are aligned to the runner's \code{tune_grid} ordering.
#'
#' @section Lightweight fit objects:
#' When \code{strip_fit = TRUE}, each fitted GLM is reduced to a minimal
#' representation sufficient for prediction:
#' \itemize{
#'   \item a numeric coefficient vector aligned to the frozen design columns
#'         \code{x_cols},
#'   \item the inverse link function \code{linkinv} for mapping linear predictors
#'         to hazard probabilities,
#'   \item the training column names \code{x_cols}.
#' }
#' This can substantially reduce memory usage when storing fold-specific CV fits.
#'
#' @param rhs_list A list of RHS specifications, either:
#' \itemize{
#'   \item one-sided formulas such as \code{~ W1 + ns(bin_id, df = 5)}, or
#'   \item character strings such as \code{"W1 + splines::ns(bin_id, df = 5)"}.
#' }
#' These RHS are used to construct \code{in_bin ~ <rhs>} internally.
#'
#' @param use_weights_col Logical. If \code{TRUE} and the training data contain a
#' column named \code{wts}, it is passed as \code{weights = ...} to
#' \code{stats::glm.fit()}. Otherwise, fitting is unweighted.
#'
#' @param strip_fit Logical. If \code{TRUE} (default), store a lightweight fit
#' representation (coefficients + link inverse only) rather than a full \code{glm}
#' object. If \code{FALSE}, models are fit and stored as full \code{glm} objects.
#'
#' @param ... Additional arguments forwarded to \code{stats::glm.fit()} when
#' \code{strip_fit = TRUE}, and to \code{stats::glm()} when \code{strip_fit = FALSE}.
#'
#' @return A named list (runner) with elements:
#' \describe{
#'   \item{method}{Character string \code{"glm"}.}
#'   \item{tune_grid}{Data frame with columns \code{.tune} and \code{rhs}.}
#'   \item{fit}{Function \code{fit(train_set, ...)} returning a fit bundle. The
#'   fit bundle contains \code{fits}, a list of length \code{K} of fitted models
#'   (stripped or full), and \code{design_specs} that freeze spline knots/columns
#'   for each RHS.}
#'   \item{predict}{Function \code{predict(fit_bundle, newdata, ...)} returning an
#'   \code{n_long x K} matrix of hazard predictions.}
#'   \item{fit_one}{Function \code{fit_one(train_set, tune, ...)} fitting only the
#'   selected tuning index and returning a minimal fit bundle compatible with
#'   \code{predict()}.}
#' }
#'
#' @details
#' ## Data requirements
#' The runner expects \code{train_set} and \code{newdata} in the **long hazard
#' format** produced by \code{format_long_hazards()}, including:
#' \itemize{
#'   \item a binary outcome column \code{in_bin},
#'   \item covariates referenced in \code{rhs_list},
#'   \item an optional \code{wts} column of observation weights (repeated on long rows).
#' }
#'
#' ## Spline handling
#' Spline knots (and boundary knots) are computed once per fold (inside \code{fit()})
#' using the training data and stored in \code{design_specs}. Prediction uses the
#' stored spline specifications to ensure consistent spline bases across training
#' and validation data within each fold.
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
#' runner <- make_glm_runner(
#'   rhs_list = rhs_list,
#'   use_weights_col = TRUE,
#'   strip_fit = TRUE
#' )
#'
#' @export

make_glm_runner <- function(
  rhs_list,
  use_weights_col = TRUE,
  strip_fit = TRUE,
  ...
) {
  stopifnot(requireNamespace("splines", quietly = TRUE))

  # Accept RHS formulas (~ ...) or RHS strings
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

  # --- helpers (copied/adapted from glmnet runner) -------------------------

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
      x_name <- deparse(substitute(x))
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

    list(
      X = X,
      terms = tt,
      design_spec = list(rhs = rhs, specs = specs, x_cols = colnames(X))
    )
  }

  build_design_new <- function(design_spec, newdata) {
    f <- stats::as.formula(paste0("in_bin ~ ", design_spec$rhs))

    env <- make_ns_env(design_spec$specs, parent_env = environment(f))
    environment(f) <- env

    tt <- stats::terms(f, data = newdata)
    Xn <- stats::model.matrix(tt, data = newdata)

    x_cols <- design_spec$x_cols
    missing <- setdiff(x_cols, colnames(Xn))
    if (length(missing)) {
      Xn <- cbind(Xn, matrix(0, nrow(Xn), length(missing), dimnames = list(NULL, missing)))
    }
    Xn <- Xn[, x_cols, drop = FALSE]
    Xn
  }

  strip_glm_coef <- function(glm_fit, x_cols) {
    coefs <- stats::coef(glm_fit)
    coefs[is.na(coefs)] <- 0

    # ensure all columns present (pad with zeros)
    miss <- setdiff(x_cols, names(coefs))
    if (length(miss)) {
      coefs <- c(coefs, stats::setNames(rep(0, length(miss)), miss))
    }
    coefs <- coefs[x_cols]

    out <- list(
      coefficients = coefs,
      # store linkinv for response-scale hazards
      linkinv = glm_fit$family$linkinv,
      x_cols = x_cols
    )
    class(out) <- "glm_stripped"
    out
  }

  predict_glm_stripped <- function(obj, Xnew) {
    eta <- as.numeric(Xnew %*% obj$coefficients)
    as.numeric(obj$linkinv(eta))
  }

  # --- runner --------------------------------------------------------------

  list(
    method = "glm",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      if (!("in_bin" %in% names(train_set))) stop("train_set must contain column 'in_bin'.")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      design_specs <- setNames(vector("list", length(rhs_chr)), rhs_chr)
      fits <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        rhs_raw <- tune_grid$rhs[k]
        built <- build_design_train(rhs_raw, train_set)
        X <- built$X
        y <- as.numeric(train_set$in_bin)

        fit_k <- suppressWarnings(stats::glm.fit(
          x = X, y = y,
          weights = wts_vec,
          family = stats::binomial(),
          ...
        ))

        if (strip_fit) {
          fits[[k]] <- strip_glm_coef(fit_k, x_cols = built$design_spec$x_cols)
        } else {
          # fall back to full glm object (larger)
          fits[[k]] <- suppressWarnings(stats::glm(
            stats::as.formula(paste0("in_bin ~ ", normalize_rhs(rhs_raw))),
            data = train_set,
            family = stats::binomial(),
            weights = wts_vec,
            ...
          ))
        }

        design_specs[[rhs_raw]] <- built$design_spec
      }

      list(
        fits = fits,
        design_specs = design_specs,
        rhs_chr = rhs_chr,
        stripped = isTRUE(strip_fit)
      )
    },

    predict = function(fit_bundle, newdata, ...) {
      nd <- as.data.frame(newdata)

      K <- length(fit_bundle$fits)
      preds <- matrix(NA_real_, nrow = nrow(nd), ncol = K)

      for (k in seq_len(K)) {
        rhs_raw <- tune_grid$rhs[k]
        Xn <- build_design_new(fit_bundle$design_specs[[rhs_raw]], nd)

        fit_k <- fit_bundle$fits[[k]]
        if (inherits(fit_k, "glm_stripped")) {
          preds[, k] <- predict_glm_stripped(fit_k, Xn)
        } else {
          # full glm fallback
          preds[, k] <- stats::predict(fit_k, newdata = nd, type = "response", ...)
        }
      }

      preds
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      rhs_raw <- tune_grid$rhs[k]

      if (!("in_bin" %in% names(train_set))) stop("train_set must contain column 'in_bin'.")
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      built <- build_design_train(rhs_raw, train_set)
      X <- built$X
      y <- as.numeric(train_set$in_bin)

      fit_k <- suppressWarnings(stats::glm.fit(
        x = X, y = y,
        weights = wts_vec,
        family = stats::binomial(),
        ...
      ))

      fit_store <- if (strip_fit) strip_glm_coef(fit_k, x_cols = built$design_spec$x_cols) else fit_k

      list(
        fits = list(fit_store),
        design_specs = setNames(list(built$design_spec), rhs_raw),
        rhs_chr = rhs_raw,
        stripped = isTRUE(strip_fit),
        tune = k
      )
    }
  )
}

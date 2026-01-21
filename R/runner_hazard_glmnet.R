#' Create a glmnet runner for discrete-time hazard modeling
#'
#' Constructs a runner (learner adapter) compatible with the
#' run_grid_setting() / summarize_and_select() workflow used in dsl_densify.
#' The runner fits penalized logistic regression models via glmnet::glmnet()
#' on long-format discrete-time hazard data with binary outcome in_bin.
#'
#' Tuning is performed over a grid defined by:
#'   - multiple RHS model specifications (rhs_list),
#'   - elastic net mixing parameters (alpha_grid),
#'   - a fixed penalty grid (lambda_grid).
#'
#' Each fitted model estimates per-bin discrete-time hazards
#'   P(T in bin_j | T >= bin_j, W)
#' under the discrete-time hazard likelihood induced by the long-data
#' construction used in dsl_densify.
#'
#' Spline handling
#'
#' Natural spline terms may be specified directly in the RHS using ns() or
#' splines::ns(), subject to the following restrictions:
#'   - Only ns(<symbol>, df = ...) is supported; the first argument must be a
#'     bare variable name.
#'   - splines::ns() is accepted but internally rewritten to ns().
#'   - Knot locations and boundary knots are computed once on the training
#'     data within each fold and then frozen.
#'
#' Frozen spline bases are enforced by evaluating model.matrix() in an
#' environment where ns() is replaced by a wrapper that injects the stored
#' knot information. This ensures consistent spline bases within each fold
#' and between training, validation, and sampling.
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All variables
#' referenced in rhs_list (including bin_id and all covariates in W) must
#' already be numeric. Factors, characters, and ordered factors are not
#' supported and are not coerced internally.
#'
#' Tuning grid and prediction layout
#'
#' The internal tune_grid is ordered such that RHS varies first, then alpha,
#' then lambda (lambda varies fastest). During cross-validation, predict()
#' returns an n_long x K matrix of predicted hazards, where
#'   K = length(rhs_list) * length(alpha_grid) * length(lambda_grid),
#' with columns aligned to tune_grid (and .tune).
#'
#' Prediction delegates to an internal predict_hazards() helper so that
#' prediction and sampling always use identical hazard estimates.
#'
#' Sampling from the fitted hazard model
#'
#' The runner provides a sample() method that generates draws
#'   A* ~ f_hat(· | W)
#' from the implied conditional density under the discrete-time hazard
#' representation.
#'
#' Sampling assumes the fit_bundle contains exactly one fitted model
#' (length(fit_bundle$fits) == 1). This is the intended usage after model
#' selection, for example via select_fit_tune() or fit_one().
#'
#' IMPORTANT: The sample() method expects newdata in long hazard format.
#' Expansion of wide W to long form (repeating rows across all bins and
#' attaching bin_lower and bin_upper) is handled upstream by
#' sample.dsldensify(). The runner itself never constructs hazard grids.
#'
#' For each subject, sampling proceeds by:
#'   - predicting hazards h_j for all bins,
#'   - computing implied bin masses
#'       p_j = h_j * prod_{l < j} (1 - h_l),
#'   - normalizing the masses to sum to one,
#'   - sampling a bin index according to p_j,
#'   - sampling uniformly within the selected bin.
#'
#' If the total mass is non-finite or non-positive for any subject, a single
#' warning is issued and sampling for those subjects falls back to uniform
#' sampling over bins.
#'
#' Lightweight fit objects
#'
#' When strip_fit = TRUE, each fitted (rhs, alpha) block is reduced to a
#' minimal representation sufficient for prediction:
#'   - an intercept vector a0 of length length(lambda_grid),
#'   - a sparse coefficient matrix beta (Matrix::dgCMatrix), dimensions p x L,
#'   - the training column names x_cols.
#'
#' This saves memory relative to storing full glmnet objects.
#'
#' @param rhs_list A list of RHS specifications, either as one-sided formulas
#'   (for example, ~ W1 + ns(bin_id, df = 5)) or as character strings
#'   (for example, "W1 + splines::ns(bin_id, df = 5)").
#'
#' @param alpha_grid Numeric vector of elastic net mixing parameters passed to
#'   glmnet::glmnet(alpha = ...). Typical values lie in [0, 1], where alpha = 1
#'   corresponds to the lasso and alpha = 0 to ridge regression.
#'
#' @param lambda_grid Numeric vector of strictly positive penalty values.
#'   This grid is treated as fixed and reused across folds and grid settings
#'   to ensure tuning alignment. Must have length at least 2.
#'
#' @param use_weights_col Logical. If TRUE and the training data contain a
#'   column named wts, it is passed as case weights to glmnet::glmnet().
#'
#' @param standardize Logical. Passed to glmnet::glmnet(standardize = ...).
#'
#' @param intercept Logical. Passed to glmnet::glmnet(intercept = ...).
#'
#' @param strip_fit Logical. If TRUE (default), store a lightweight
#'   coefficient-based representation of each fitted model.
#'
#' @param ... Additional arguments forwarded to glmnet::glmnet().
#'
#' @return A named list (runner) with the following elements:
#'   method: Character string "glmnet".
#'   tune_grid: Data frame describing the tuning grid, including .tune, rhs,
#'     alpha, and lambda.
#'   fit: Function fit(train_set, ...) returning a fit bundle.
#'   predict: Function predict(fit_bundle, newdata, ...) returning an
#'     n_long x K matrix of predicted hazards.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected
#'     tuning index.
#'   select_fit: Function select_fit(fit_bundle, tune) returning a reduced
#'     fit bundle for a single tuning index (K = 1).
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing samples
#'     from the implied conditional density (assumes K = 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata in long hazard format, including:
#'   - a binary outcome column in_bin,
#'   - a time-bin column bin_id,
#'   - covariates referenced in rhs_list,
#'   - an optional weight column wts.
#'
#' newdata passed to sample() must additionally include:
#'   - obs_id,
#'   - bin_lower,
#'   - bin_upper.
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
make_glmnet_hazard_runner <- function(
  rhs_list,
  alpha_grid,
  lambda_grid,
  use_weights_col = TRUE,
  standardize = TRUE,
  intercept = TRUE,
  strip_fit = TRUE,
  ...
) {
  stopifnot(requireNamespace("glmnet", quietly = TRUE))
  stopifnot(requireNamespace("splines", quietly = TRUE))
  stopifnot(requireNamespace("Matrix", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))

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
    lambda = lambda_grid,
    alpha  = alpha_grid,
    rhs    = rhs_chr,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  tune_grid$.tune <- seq_len(nrow(tune_grid))
  tune_grid <- tune_grid[, c(".tune", "rhs", "alpha", "lambda")]

  # ---- hazard conventions (match dsldensify + other hazard runners) -------
  id_col <- "obs_id"
  bin_var <- "bin_id"
  eps <- 1e-15
  clip01 <- function(p) pmin(pmax(p, eps), 1 - eps)

  # --- helpers: frozen splines via model.matrix ---------------------------

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

    # glmnet handles intercept separately; remove intercept column if present
    if ("(Intercept)" %in% colnames(X)) {
      X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
    }

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

  # Strip a glmnet block down to intercept + beta for a fixed lambda_grid
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

  # Predict hazards for all tuned models in a fit bundle
  predict_hazards <- function(fit_bundle, newdata, ...) {
    if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table.")
    if (is.null(fit_bundle$fits) || !length(fit_bundle$fits)) stop("fit_bundle does not contain fits.")

    nd <- as.data.frame(newdata)
    n <- nrow(nd)

    # Selected model bundle: fits is a flat list of length 1
    if (!is.null(fit_bundle$selected) && isTRUE(fit_bundle$selected)) {
      fit1 <- fit_bundle$fits[[1L]]
      rhs_raw <- fit_bundle$rhs_chr
      ds <- fit_bundle$design_specs[[rhs_raw]]
      Xn <- build_design_new(ds, nd)

      if (inherits(fit1, "glmnet_stripped")) {
        p <- as.numeric(predict_stripped_block(fit1, Xn))
        return(matrix(clip01(p), nrow = n, ncol = 1L))
      }

      if (inherits(fit1, "glmnet")) {
        s <- fit_bundle$lambda_grid
        eta <- as.matrix(glmnet::predict.glmnet(fit1, newx = Xn, s = s, ...))
        p <- plogis(eta)
        return(matrix(clip01(p[, 1L]), nrow = n, ncol = 1L))
      }

      stop("Unexpected fit object for selected glmnet runner: class = ", paste(class(fit1), collapse = ", "))
    }

    # Full bundle: nested fits by rhs then alpha (each a path over lambda_grid)
    K <- length(fit_bundle$rhs_chr) * length(fit_bundle$alpha_grid) * length(fit_bundle$lambda_grid)
    out <- matrix(NA_real_, nrow = n, ncol = K)
    col_idx <- 1L

    for (rhs_raw in fit_bundle$rhs_chr) {
      ds <- fit_bundle$design_specs[[rhs_raw]]
      Xn <- build_design_new(ds, nd)

      for (alpha in fit_bundle$alpha_grid) {
        fit_ra <- fit_bundle$fits[[rhs_raw]][[as.character(alpha)]]

        if (inherits(fit_ra, "glmnet_stripped")) {
          pmat <- predict_stripped_block(fit_ra, Xn)  # n x L
        } else if (inherits(fit_ra, "glmnet")) {
          etamat <- as.matrix(glmnet::predict.glmnet(
            fit_ra, newx = Xn, s = fit_bundle$lambda_grid, ...
          ))
          pmat <- plogis(etamat)  # n x L
        } else {
          stop("Unexpected fit object for glmnet runner: class = ", paste(class(fit_ra), collapse = ", "))
        }

        L <- ncol(pmat)
        out[, col_idx:(col_idx + L - 1L)] <- clip01(pmat)
        col_idx <- col_idx + L
      }
    }

    out
  }

  # --- runner --------------------------------------------------------------

  list(
    method = "glmnet",
    tune_grid = tune_grid,
    positive_support = TRUE,
    
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

          if (strip_fit) {
            fit_ra <- strip_glmnet_block(fit_ra, x_cols = built$design_spec$x_cols, lambda_grid = lambda_grid)
          }
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
        stripped = isTRUE(strip_fit),
        selected = FALSE
      )
    },

    predict = function(fit_bundle, newdata, ...) {
      dt <- data.table::as.data.table(newdata)
      predict_hazards(fit_bundle = fit_bundle, newdata = dt, ...)
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      rhs_raw  <- tune_grid$rhs[k]
      alpha_k  <- tune_grid$alpha[k]
      lambda_k <- as.numeric(tune_grid$lambda[k])

      if (!("in_bin" %in% names(train_set))) stop("train_set must contain column 'in_bin'.")
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) as.numeric(train_set$wts) else NULL
      y <- as.numeric(train_set$in_bin)

      built <- build_design_train(rhs_raw, train_set)
      X <- built$X
      design_spec <- built$design_spec

      # Fit full path (fixed lambda_grid), then reduce to single lambda
      fit_ra <- glmnet::glmnet(
        x = X, y = y, weights = wts_vec,
        family = "binomial",
        alpha = alpha_k,
        lambda = lambda_grid,
        standardize = standardize,
        intercept = intercept,
        ...
      )

      if (strip_fit) {
        # extract coefficients at single lambda_k
        B <- glmnet::coef.glmnet(fit_ra, s = lambda_k)  # (p+1) x 1
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
        fit_sel <- fit_ra
      }

      list(
        fits = list(fit_sel),
        design_specs = setNames(list(design_spec), rhs_raw),
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

      if (inherits(fit_ra, "glmnet_stripped")) {
        # reduce block to single lambda
        lam_all <- fit_ra$lambda
        jj <- match(lam_k, lam_all)
        if (is.na(jj)) stop("Selected lambda not found in stripped lambda grid.")
        fit_sel <- list(
          a0 = fit_ra$a0[jj],
          beta = fit_ra$beta[, jj, drop = FALSE],
          x_cols = fit_ra$x_cols,
          lambda = lam_k
        )
        class(fit_sel) <- "glmnet_stripped"
      } else if (inherits(fit_ra, "glmnet")) {
        fit_sel <- fit_ra
      } else {
        stop("Unexpected fit object for glmnet runner: class = ", paste(class(fit_ra), collapse = ", "))
      }

      list(
        fits = list(fit_sel),
        design_specs = setNames(list(ds), rhs_k),
        rhs_chr = rhs_k,
        alpha_grid = alpha_k,
        lambda_grid = lam_k,
        stripped = isTRUE(fit_bundle$stripped),
        selected = TRUE,
        tune = k
      )
    },

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
      haz <- predict_hazards(fit_bundle = fit_bundle, newdata = newdata, ...)
      haz <- as.numeric(haz[, 1L])

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

          j <- sample.int(m, size = n_samp, replace = TRUE)

        } else {
          pmf <- mass / tot
          j <- sample.int(m, size = n_samp, replace = TRUE, prob = pmf)
        }

        out[ii, ] <- stats::runif(n_samp, min = lower[j], max = upper[j])
      }

      out
    }
  )
}

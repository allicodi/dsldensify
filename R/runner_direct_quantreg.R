#' Create a quantile-based direct density runner
#'
#' Constructs a runner (learner adapter) compatible with the dsldensify() /
#' summarize_and_select() workflow for direct conditional density estimation of a
#' continuous outcome A given covariates W using an estimated conditional quantile
#' function Q(p | W).
#'
#' This runner fits conditional quantile models on a fixed grid of probability
#' levels p in (0, 1). The implied conditional density is recovered by
#' differentiating the fitted quantile function in p and applying the identity
#'   f(a | W) = 1 / (dQ(p | W) / dp) evaluated at p satisfying Q(p | W) = a.
#'
#' Sampling is straightforward: draw p ~ Uniform(0, 1) and return A* = Q(p | W).
#'
#' The runner is designed to be lightweight and robust in cross-validation:
#'   - quantile levels are fixed per tuning row,
#'   - monotonicity in p can be enforced by rearrangement,
#'   - density evaluation uses either finite differences or a p-smoother.
#'
#' Numeric-only requirement
#'
#' Covariates referenced in RHS specifications are assumed numeric. Factor handling
#' is not supported.
#'
#' Tuning grid and prediction layout
#'
#' The internal tune_grid is the Cartesian product of:
#'   - rhs (from rhs_list),
#'   - n_quantiles (from n_quantiles_grid),
#'   - smoother (from smoother_grid),
#'   - bandwidth / df parameters for the chosen smoother (when applicable).
#'
#' Each row of tune_grid corresponds to exactly one fitted quantile learner.
#' During cross-validation, log_density() returns an n x K matrix of log-densities
#' aligned to .tune.
#'
#' Sampling from the fitted model
#'
#' The runner provides a sample() method that generates draws A* ~ f_hat(· | W).
#' Sampling assumes the fit_bundle contains exactly one tuned fit
#' (length(fit_bundle$fits) == 1). It expects newdata in wide format containing W
#' only and returns an nrow(newdata) x n_samp matrix.
#'
#' @param rhs_list A list of RHS specifications, either one-sided formulas
#'   (for example, ~ x1 + x2) or character strings (for example, "x1 + x2").
#'   These RHS are used to build the mean structure for the conditional quantile
#'   function.
#'
#' @param n_quantiles_grid Integer vector giving the number of quantile levels to
#'   use in the p grid (for example, 50, 100, 200).
#'
#' @param p_min,p_max Endpoints for the internal p grid. The grid is
#'   seq(p_min, p_max, length.out = n_quantiles). Defaults avoid exactly 0/1.
#'
#' @param smoother_grid Character vector specifying how to obtain dQ/dp from the
#'   fitted quantile curve:
#'   - "diff": finite differences on the p grid
#'   - "spline": smoothing spline in p for each observation
#'   - "local": local linear regression in p for each observation
#'
#' @param spline_df_grid Integer vector of degrees of freedom for smoother_grid = "spline".
#'
#' @param local_span_grid Numeric vector of spans for smoother_grid = "local".
#'
#' @param enforce_monotone Logical. If TRUE, apply a rearrangement step to ensure
#'   the predicted quantile function is nondecreasing in p for each observation.
#'
#' @param use_weights_col Logical. If TRUE and the training data contain a column
#'   named wts, it is passed to the quantile fitting routine as weights when
#'   supported.
#'
#' @param strip_fit Logical. If TRUE, store a lightweight fit representation.
#'
#' @param eps Small positive constant used to bound densities away from zero and
#'   log-densities away from -Inf, and to clip p and derivatives.
#'
#' @param ... Additional arguments passed to the underlying quantile learner.
#'   This sketch uses quantreg::rq() by default.
#'
#' @return A named list (runner) with elements:
#'   method: Character string "quantile".
#'   tune_grid: Data frame describing the tuning grid, including .tune.
#'   fit: Function fit(train_set, ...) returning a fit bundle.
#'   log_density: Function log_density(fit_bundle, newdata, ...) returning an
#'     n x K matrix of log-densities.
#'   density: Function density(fit_bundle, newdata, ...) returning densities.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected tuning index.
#'   select_fit: Function select_fit(fit_bundle, tune) extracting a single tuning configuration.
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing samples
#'     (assumes length(fit_bundle$fits) == 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata in wide format containing:
#'   - a numeric outcome column A,
#'   - covariates referenced in rhs_list,
#'   - an optional weight column wts.
#'
#' @examples
#' runner <- make_quantile_runner(
#'   rhs_list = list(~ x1 + x2),
#'   n_quantiles_grid = c(50L, 100L),
#'   smoother_grid = c("diff", "spline"),
#'   spline_df_grid = c(7L, 11L)
#' )
#'
#' @export
make_quantreg_direct_runner <- function(
  rhs_list,
  n_quantiles_grid = c(50L, 100L),
  p_min = 0.01,
  p_max = 0.99,
  smoother_grid = c("diff", "spline"),
  spline_df_grid = c(7L),
  local_span_grid = c(0.25),
  enforce_monotone = TRUE,
  use_weights_col = TRUE,
  strip_fit = TRUE,
  eps = 1e-12,
  ...
) {
  stopifnot(requireNamespace("quantreg", quietly = TRUE))

  # rhs parsing
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  n_quantiles_grid <- as.integer(n_quantiles_grid)
  if (any(is.na(n_quantiles_grid)) || any(n_quantiles_grid < 5L)) stop("n_quantiles_grid must be integers >= 5")

  smoother_grid <- unique(as.character(smoother_grid))
  if (!all(smoother_grid %in% c("diff", "spline", "local"))) {
    stop("smoother_grid must be subset of c('diff','spline','local')")
  }

  # tune grid (rhs major)
  tune_grid <- expand.grid(
    rhs = rhs_chr,
    n_quantiles = n_quantiles_grid,
    smoother = smoother_grid,
    stringsAsFactors = FALSE,
    KEEP.OUT.ATTRS = FALSE
  )
  # add smoother-specific params; keep a single column each with NA when irrelevant
  tune_grid$spline_df <- NA_integer_
  tune_grid$local_span <- NA_real_
  for (i in seq_len(nrow(tune_grid))) {
    if (tune_grid$smoother[i] == "spline") tune_grid$spline_df[i] <- spline_df_grid[1L]
    if (tune_grid$smoother[i] == "local") tune_grid$local_span[i] <- local_span_grid[1L]
  }
  # expand for spline_df and local_span if multiple provided
  if (any(tune_grid$smoother == "spline") && length(spline_df_grid) > 1L) {
    tg0 <- tune_grid[tune_grid$smoother != "spline", , drop = FALSE]
    tg1 <- tune_grid[tune_grid$smoother == "spline", , drop = FALSE]
    tg1 <- merge(tg1[, setdiff(names(tg1), "spline_df"), drop = FALSE],
                 data.frame(spline_df = as.integer(spline_df_grid)), by = NULL)
    tune_grid <- rbind(tg0, tg1)
  }
  if (any(tune_grid$smoother == "local") && length(local_span_grid) > 1L) {
    tg0 <- tune_grid[tune_grid$smoother != "local", , drop = FALSE]
    tg1 <- tune_grid[tune_grid$smoother == "local", , drop = FALSE]
    tg1 <- merge(tg1[, setdiff(names(tg1), "local_span"), drop = FALSE],
                 data.frame(local_span = as.numeric(local_span_grid)), by = NULL)
    tune_grid <- rbind(tg0, tg1)
  }

  tune_grid <- tune_grid[order(match(tune_grid$rhs, rhs_chr), tune_grid$n_quantiles, tune_grid$smoother), , drop = FALSE]
  rownames(tune_grid) <- NULL
  tune_grid$.tune <- seq_len(nrow(tune_grid))

  clip01 <- function(p) pmin(pmax(p, eps), 1 - eps)

  p_grid <- function(nq) clip01(seq(p_min, p_max, length.out = nq))

  enforce_monotone_vec <- function(q) {
    # simple rearrangement: sort in p
    sort(q, decreasing = FALSE)
  }

  # Build design matrices (keep simple, numeric-only)
  build_design_train <- function(rhs_raw, train_set) {
    f <- stats::as.formula(paste0("A ~ ", rhs_raw))
    tt <- stats::terms(f, data = train_set)
    X <- stats::model.matrix(tt, data = train_set)
    list(X = X, terms = tt, x_cols = colnames(X), rhs = rhs_raw)
  }

  build_design_new <- function(design_spec, newdata) {
    f <- stats::as.formula(paste0("A ~ ", design_spec$rhs))
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

  # Fit rq at multiple taus and store a compact representation:
  # coefficient matrix B: p x M, and x_cols
  fit_rq_path <- function(X, y, taus, wts_vec, ...) {
    # quantreg::rq can fit multiple taus at once when tau is a vector
    # Use method = "fn" by default; users can override via ...
    fit <- quantreg::rq(x = X, y = y, tau = taus, weights = wts_vec, ...)
    # coef() returns p x M when multiple taus
    B <- as.matrix(stats::coef(fit))
    # Make sure orientation is p x M
    if (nrow(B) != ncol(X) && ncol(B) == ncol(X)) B <- t(B)
    list(B = B, taus = taus, x_cols = colnames(X))
  }

  predict_q <- function(fit_obj, Xnew) {
    # returns n x M quantiles
    Q <- as.matrix(Xnew %*% fit_obj$B)
    Q
  }

  # Given predicted Q (n x M) on taus, compute derivative dQ/dp at grid points.
  # Return list(dq = n x M, Q = n x M, taus = M)
  derive_dQ <- function(Q, taus, smoother, spline_df = NA_integer_, local_span = NA_real_) {
    n <- nrow(Q); M <- length(taus)
    dq <- matrix(NA_real_, n, M)

    if (smoother == "diff") {
      # central diffs interior, forward/backward at ends
      dt <- diff(taus)
      for (j in seq_len(M)) {
        if (j == 1L) {
          dq[, j] <- (Q[, 2L] - Q[, 1L]) / (taus[2L] - taus[1L])
        } else if (j == M) {
          dq[, j] <- (Q[, M] - Q[, M - 1L]) / (taus[M] - taus[M - 1L])
        } else {
          dq[, j] <- (Q[, j + 1L] - Q[, j - 1L]) / (taus[j + 1L] - taus[j - 1L])
        }
      }
      dq <- pmax(dq, eps)
      return(list(Q = Q, dq = dq, taus = taus))
    }

    if (smoother == "spline") {
      df <- as.integer(spline_df)
      if (is.na(df) || df < 3L) stop("spline_df must be >= 3 for smoother='spline'")
      for (i in seq_len(n)) {
        sp <- stats::smooth.spline(x = taus, y = Q[i, ], df = df)
        pr <- stats::predict(sp, x = taus, deriv = 1L)$y
        dq[i, ] <- pmax(as.numeric(pr), eps)
      }
      return(list(Q = Q, dq = dq, taus = taus))
    }

    if (smoother == "local") {
      span <- as.numeric(local_span)
      if (!is.finite(span) || span <= 0 || span > 1) stop("local_span must be in (0, 1] for smoother='local'")
      # local linear: approximate derivative by fitting weighted local regression around each tau_j
      # Keep it simple: for each i, each j, use loess to get derivative-like slope
      for (i in seq_len(n)) {
        # loess returns fitted values; derivative not direct. Use finite diff on loess-smoothed Q.
        lo <- stats::loess(Q[i, ] ~ taus, span = span, degree = 1L)
        qhat <- stats::predict(lo, newdata = data.frame(taus = taus))
        qhat[is.na(qhat)] <- Q[i, is.na(qhat)]
        # then diff on smoothed curve
        for (j in seq_len(M)) {
          if (j == 1L) dq[i, j] <- (qhat[2L] - qhat[1L]) / (taus[2L] - taus[1L])
          else if (j == M) dq[i, j] <- (qhat[M] - qhat[M - 1L]) / (taus[M] - taus[M - 1L])
          else dq[i, j] <- (qhat[j + 1L] - qhat[j - 1L]) / (taus[j + 1L] - taus[j - 1L])
        }
      }
      dq <- pmax(dq, eps)
      return(list(Q = Q, dq = dq, taus = taus))
    }

    stop("Unknown smoother: ", smoother)
  }

  # For each (a_i, Q_i(.)), find p location by interpolation.
  # Return list(p_hat = n, dq_at = n) where dq_at approximates dQ/dp at p_hat.
  invert_and_deriv <- function(a, Q, dq, taus) {
    n <- length(a); M <- length(taus)
    p_hat <- rep(NA_real_, n)
    dq_at <- rep(NA_real_, n)

    for (i in seq_len(n)) {
      qi <- Q[i, ]
      dqi <- dq[i, ]

      # handle monotone assumption; if not monotone, caller should have enforced it
      # clamp a into [min(q), max(q)] for interpolation (density eps outside)
      if (!is.finite(a[i])) {
        p_hat[i] <- NA_real_
        dq_at[i] <- NA_real_
        next
      }
      if (a[i] <= qi[1L]) {
        p_hat[i] <- taus[1L]
        dq_at[i] <- dqi[1L]
        next
      }
      if (a[i] >= qi[M]) {
        p_hat[i] <- taus[M]
        dq_at[i] <- dqi[M]
        next
      }

      j <- max(which(qi <= a[i]))
      j <- min(j, M - 1L)

      # linear interpolation in quantile space
      q0 <- qi[j]; q1 <- qi[j + 1L]
      t0 <- taus[j]; t1 <- taus[j + 1L]
      w <- if (q1 > q0) (a[i] - q0) / (q1 - q0) else 0
      p_hat[i] <- t0 + w * (t1 - t0)

      # interpolate derivative as well
      dq0 <- dqi[j]; dq1 <- dqi[j + 1L]
      dq_at[i] <- pmax(dq0 + w * (dq1 - dq0), eps)
    }

    list(p_hat = p_hat, dq_at = dq_at)
  }

  list(
    method = "quantile",
    tune_grid = tune_grid,
    positive_support = TRUE,
    
    fit = function(train_set, ...) {
      if (!("A" %in% names(train_set))) stop("train_set must contain column 'A'.")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) as.numeric(train_set$wts) else NULL

      fits <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        tg <- tune_grid[k, , drop = FALSE]
        built <- build_design_train(tg$rhs, train_set)
        X <- built$X
        y <- as.numeric(train_set$A)

        taus <- p_grid(as.integer(tg$n_quantiles))

        fit_obj <- fit_rq_path(X = X, y = y, taus = taus, wts_vec = wts_vec, ...)
        fit_obj$design_spec <- list(rhs = built$rhs, x_cols = built$x_cols)

        # keep tuning info needed at score time
        fit_obj$smoother <- as.character(tg$smoother)
        fit_obj$spline_df <- as.integer(tg$spline_df)
        fit_obj$local_span <- as.numeric(tg$local_span)
        fit_obj$enforce_monotone <- isTRUE(enforce_monotone)

        if (strip_fit) {
          # rq object itself can be large; keep only coef matrix + grid + design spec + tuning knobs
          fit_obj <- list(
            B = fit_obj$B,
            taus = fit_obj$taus,
            x_cols = fit_obj$x_cols,
            design_spec = fit_obj$design_spec,
            smoother = fit_obj$smoother,
            spline_df = fit_obj$spline_df,
            local_span = fit_obj$local_span,
            enforce_monotone = fit_obj$enforce_monotone
          )
          class(fit_obj) <- "quantile_stripped"
        } else {
          class(fit_obj) <- c("quantile_full", class(fit_obj))
        }

        fits[[k]] <- fit_obj
      }

      list(fits = fits, tune_grid = tune_grid)
    },

    log_density = function(fit_bundle, newdata, eps = eps, ...) {
      nd <- as.data.frame(newdata)
      if (!("A" %in% names(nd))) stop("newdata must contain column 'A'.")

      K <- length(fit_bundle$fits)
      n <- nrow(nd)
      out <- matrix(log(eps), nrow = n, ncol = K)

      a <- as.numeric(nd$A)

      for (k in seq_len(K)) {
        obj <- fit_bundle$fits[[k]]

        Xn <- build_design_new(obj$design_spec, nd)
        Q <- predict_q(obj, Xn)          # n x M
        taus <- obj$taus

        # enforce monotonicity in p if requested (row-wise)
        if (isTRUE(obj$enforce_monotone)) {
          for (i in seq_len(n)) Q[i, ] <- enforce_monotone_vec(Q[i, ])
        }

        dd <- derive_dQ(Q = Q, taus = taus, smoother = obj$smoother,
                        spline_df = obj$spline_df, local_span = obj$local_span)

        inv <- invert_and_deriv(a = a, Q = dd$Q, dq = dd$dq, taus = dd$taus)

        dens <- 1 / pmax(inv$dq_at, eps)
        dens[!is.finite(dens)] <- eps
        out[, k] <- log(pmax(dens, eps))
      }

      out
    },

    density = function(fit_bundle, newdata, eps = eps, ...) {
      exp(log_density(fit_bundle, newdata, eps = eps, ...))
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..nrow(tune_grid).")
      }

      tg <- tune_grid[k, , drop = FALSE]
      if (!("A" %in% names(train_set))) stop("train_set must contain column 'A'.")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) as.numeric(train_set$wts) else NULL

      built <- build_design_train(tg$rhs, train_set)
      X <- built$X
      y <- as.numeric(train_set$A)
      taus <- p_grid(as.integer(tg$n_quantiles))

      fit_obj <- fit_rq_path(X = X, y = y, taus = taus, wts_vec = wts_vec, ...)
      fit_obj$design_spec <- list(rhs = built$rhs, x_cols = built$x_cols)
      fit_obj$smoother <- as.character(tg$smoother)
      fit_obj$spline_df <- as.integer(tg$spline_df)
      fit_obj$local_span <- as.numeric(tg$local_span)
      fit_obj$enforce_monotone <- isTRUE(enforce_monotone)

      if (strip_fit) {
        fit_obj <- list(
          B = fit_obj$B,
          taus = fit_obj$taus,
          x_cols = fit_obj$x_cols,
          design_spec = fit_obj$design_spec,
          smoother = fit_obj$smoother,
          spline_df = fit_obj$spline_df,
          local_span = fit_obj$local_span,
          enforce_monotone = fit_obj$enforce_monotone
        )
        class(fit_obj) <- "quantile_stripped"
      } else {
        class(fit_obj) <- c("quantile_full", class(fit_obj))
      }

      list(fits = list(fit_obj), tune = k, tune_grid = tune_grid[k, , drop = FALSE])
    },

    select_fit = function(fit_bundle, tune) {
      k <- as.integer(tune)
      if (!is.null(fit_bundle$fits) && length(fit_bundle$fits) >= k) {
        fit_bundle$fits <- fit_bundle$fits[k]
      }
      fit_bundle
    },

    sample = function(fit_bundle, newdata, n_samp, seed = NULL, ...) {
      nd <- as.data.frame(newdata)

      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) {
        stop("n_samp must be a positive integer.")
      }
      n_samp <- as.integer(n_samp)

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      if (length(fits) != 1L) {
        stop("sample() assumes K=1: fit_bundle$fits must have length 1 (selected model).")
      }
      if (!is.null(seed)) set.seed(seed)

      obj <- fits[[1L]]
      n <- nrow(nd)
      if (n < 1L) stop("newdata must have at least one row.")

      Xn <- build_design_new(obj$design_spec, nd)
      Q <- predict_q(obj, Xn)          # n x M
      taus <- obj$taus                 # length M
      M <- length(taus)

      if (M < 2L) stop("taus grid must have length >= 2 for interpolation.")

      if (isTRUE(obj$enforce_monotone)) {
        # rearrangement: enforce monotonicity in p row-wise
        # (loop is fine here; the expensive part is interpolation/sampling)
        for (i in seq_len(n)) Q[i, ] <- sort(Q[i, ], decreasing = FALSE)
      }

      # Draw all p's at once: n x n_samp
      P <- matrix(stats::runif(n * n_samp, min = eps, max = 1 - eps), nrow = n, ncol = n_samp)
      pvec <- as.vector(P)  # length n*n_samp

      # For each p, find bracketing interval in taus (shared across rows)
      j <- findInterval(pvec, taus, rightmost.closed = TRUE, all.inside = TRUE)
      j <- pmin(pmax(j, 1L), M - 1L)  # ensure 1..(M-1)

      t0 <- taus[j]
      t1 <- taus[j + 1L]
      w  <- (pvec - t0) / (t1 - t0)

      # Row indices aligned with pvec ordering (column-major vectorization)
      # as.vector(P) stacks columns, so row ids repeat for each column
      row_id <- rep(seq_len(n), times = n_samp)

      # Pull Q at (row_id, j) and (row_id, j+1) without loops
      q0 <- Q[cbind(row_id, j)]
      q1 <- Q[cbind(row_id, j + 1L)]

      out_vec <- q0 + w * (q1 - q0)
      out <- matrix(out_vec, nrow = n, ncol = n_samp)

      out
    }

  )
}

#' Create a GAMLSS beta runner for bounded outcomes using an affine transform
#'
#' Constructs a runner (learner adapter) compatible with the dsldensify() /
#' summarize_and_select() workflow for direct conditional density estimation of a
#' bounded continuous outcome A given covariates W.
#'
#' This runner assumes A lies in known bounds (lower, upper) and models a
#' transformed outcome Y in (0, 1) using a Beta family in GAMLSS. The transform is:
#'   Y = (A - lower) / (upper - lower)
#' After fitting a Beta model for Y | W, the implied conditional density for A | W
#' is obtained via a change of variables:
#'   f_A(a | W) = f_Y(y | W) / (upper - lower),
#' where y is the transformed value corresponding to a.
#'
#' The runner supports tuning over formulas for the location parameter (mu) and
#' optionally the scale parameter (sigma) of the Beta family.
#'
#' Robustness and fallback behavior
#'
#' GAMLSS fits can fail to converge or error for some folds or formula choices.
#' This runner supports an optional fallback:
#'   - fallback = "beta_glm": if a GAMLSS fit fails, fall back to a mean model for
#'     Y using a (quasi)binomial GLM with logit link, and estimate a single global
#'     precision parameter phi for the Beta distribution using a method-of-moments
#'     style estimate based on weighted residual variance.
#'   - fallback = "none": failed fits return log(eps) in log_density() and NA samples.
#'
#' Numeric-only requirement
#'
#' This runner assumes all covariates referenced in RHS formulas are numeric.
#' Factors and character predictors are not supported.
#'
#' Tuning grid and prediction layout
#'
#' The internal tune_grid is the Cartesian product of mu_rhs_list and sigma_rhs_list.
#' Each row corresponds to a distinct GAMLSS specification. During cross-validation,
#' log_density() returns an n x K matrix of log-densities aligned to .tune.
#'
#' Sampling from the fitted direct density model
#'
#' The runner provides a sample() method that generates draws A* ~ f_hat(· | W).
#' Sampling assumes the fit_bundle contains exactly one tuned fit
#' (length(fit_bundle$fits) == 1). It expects newdata in wide format containing only W
#' and returns an nrow(newdata) x n_samp numeric matrix.
#'
#' Lightweight fit objects
#'
#' When strip_fit = TRUE, fitted gamlss objects are stripped of large components
#' (responses, fitted values, residuals) before storage, reducing memory usage.
#'
#' @param mu_rhs_list RHS specifications for the location parameter mu. May be
#'   one-sided formulas (for example, ~ W1 + W2) or character strings
#'   (for example, "W1 + W2"). Must have length at least 1.
#'
#' @param sigma_rhs_list RHS specifications for the scale parameter sigma.
#'   Defaults to "1". May be formulas or character strings.
#'
#' @param lower Known lower bound for A (finite numeric scalar).
#'
#' @param upper Known upper bound for A (finite numeric scalar), must satisfy upper > lower.
#'
#' @param use_weights_col Logical. If TRUE and the training data contain a column
#'   named wts, it is passed to gamlss::gamlss() via weights. Otherwise, fitting
#'   is unweighted.
#'
#' @param strip_fit Logical. If TRUE, strip fitted gamlss objects before storing them.
#'
#' @param control A gamlss.control() object controlling the fitting procedure.
#'
#' @param eps Small positive constant used for clipping transformed values away from
#'   0 and 1, and for bounding log-densities away from -Inf.
#'
#' @param eps_fit Small positive constant used for clipping transformed values during
#'   fitting. Larger than eps is often more stable. Defaults to 1e-6.
#'
#' @param fallback Character string specifying fallback behavior when a GAMLSS fit fails.
#'   One of "beta_glm" or "none".
#'
#' @param ... Additional arguments forwarded to gamlss::gamlss().
#'
#' @return A named list (runner) with elements:
#'   method: Character string "gamlss_beta".
#'   tune_grid: Data frame enumerating mu_rhs / sigma_rhs combinations, including .tune.
#'   fit: Function fit(train_set, ...) returning a fit bundle containing all fits.
#'   log_density: Function log_density(fit_bundle, newdata, ...) returning an n x K
#'     matrix of log-densities for A | W.
#'   density: Function density(fit_bundle, newdata, ...) returning densities on the
#'     original scale for A | W.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected tuning index.
#'   select_fit: Function select_fit(fit_bundle, tune) extracting a single tuning configuration.
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing samples from the
#'     implied A | W distribution (assumes length(fit_bundle$fits) == 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata in wide format containing:
#'   - a numeric outcome column A,
#'   - covariates referenced in the RHS specifications,
#'   - an optional weight column wts.
#'
#' @examples
#' mu_rhs <- list(~ x1 + x2, ~ x1)
#' runner <- make_gamlss_beta_bounded_runner(
#'   mu_rhs_list = mu_rhs,
#'   sigma_rhs_list = c("1", "~ x1"),
#'   lower = -5,
#'   upper = 5,
#'   fallback = "beta_glm"
#' )
#'
#' @export
make_bounded_gamlss_runner <- function(
  mu_rhs_list,
  sigma_rhs_list = c("1"),
  lower,
  upper,
  use_weights_col = TRUE,
  strip_fit = TRUE,
  control = gamlss::gamlss.control(n.cyc = 50, trace = FALSE),
  eps = 1e-12,
  eps_fit = 1e-6,
  fallback = c("beta_glm", "none"),
  ...
) {
  fallback <- match.arg(fallback)

  if (!requireNamespace("gamlss", quietly = TRUE)) stop("Package 'gamlss' is required.")
  if (!requireNamespace("gamlss.dist", quietly = TRUE)) stop("Package 'gamlss.dist' is required.")

  if (length(lower) != 1L || !is.finite(lower)) stop("lower must be a finite numeric scalar.")
  if (length(upper) != 1L || !is.finite(upper)) stop("upper must be a finite numeric scalar.")
  if (!(upper > lower)) stop("upper must be greater than lower.")

  width <- as.numeric(upper - lower)
  log_jac <- log(width)

  rhs_to_chr <- function(x) {
    if (is.list(x) && all(vapply(x, inherits, logical(1), "formula"))) {
      vapply(x, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
    } else {
      as.character(x)
    }
  }

  mu_rhs_chr <- rhs_to_chr(mu_rhs_list)
  if (length(mu_rhs_chr) < 1L) stop("mu_rhs_list must have length >= 1")

  sigma_rhs_chr <- rhs_to_chr(sigma_rhs_list)
  if (length(sigma_rhs_chr) < 1L) stop("sigma_rhs_list must have length >= 1")

  tune_grid <- expand.grid(
    mu_rhs = mu_rhs_chr,
    sigma_rhs = sigma_rhs_chr,
    stringsAsFactors = FALSE
  )
  tune_grid$.tune <- seq_len(nrow(tune_grid))

  strip_gamlss <- function(fit) {
    fit$y <- NULL
    fit$residuals <- NULL
    fit$mu.fv <- NULL
    fit$sigma.fv <- NULL
    fit$nu.fv <- NULL
    fit$tau.fv <- NULL
    fit
  }

  fix_family_call <- function(fit, fam_chr) {
    fit$call$family <- substitute(
      get(FAM, envir = asNamespace("gamlss.dist")),
      list(FAM = fam_chr)
    )
    fit
  }

  fix_data_call <- function(fit, dat) {
    fit$.train_data <- dat
    fit$call$data <- quote(object$.train_data)
    fit
  }

  # BE parameterization: mu in (0,1), sigma > 0
  dfun_BE <- get("dBE", envir = asNamespace("gamlss.dist"))
  rfun_BE <- get("rBE", envir = asNamespace("gamlss.dist"))

  to_unit <- function(a) (as.numeric(a) - lower) / width
  from_unit <- function(y) lower + width * as.numeric(y)
  clip01_eps <- function(y, e) pmin(pmax(y, e), 1 - e)

  # ---- beta_glm fallback helpers ------------------------------------------
  normalize_rhs <- function(rhs) gsub("splines::ns", "ns", rhs, fixed = TRUE)

  build_design_train_glm <- function(mu_rhs_raw, dat) {
    rhs <- normalize_rhs(mu_rhs_raw)
    f <- stats::as.formula(paste0("A_unit ~ ", rhs))
    tt <- stats::terms(f, data = dat)
    X <- stats::model.matrix(tt, data = dat)
    list(X = X, terms = tt, x_cols = colnames(X), rhs = rhs)
  }

  build_design_new_glm <- function(design_spec, newdata) {
    f <- stats::as.formula(paste0("A_unit ~ ", design_spec$rhs))
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

    miss <- setdiff(x_cols, names(coefs))
    if (length(miss)) coefs <- c(coefs, stats::setNames(rep(0, length(miss)), miss))
    coefs <- coefs[x_cols]

    out <- list(coefficients = coefs, x_cols = x_cols)
    class(out) <- "glm_stripped"
    out
  }

  predict_glm_stripped_mu <- function(obj, Xnew, eps_mu) {
    eta <- as.numeric(Xnew %*% obj$coefficients)
    mu <- stats::plogis(eta)
    clip01_eps(mu, eps_mu)
  }

  estimate_phi_beta <- function(y, mu, wts, eps_phi) {
    # Var(Y) approx mu(1-mu)/(phi+1). Estimate dispersion d = 1/(phi+1) via
    # weighted residual variance relative to mu(1-mu).
    mu <- clip01_eps(mu, eps_fit)
    denom <- mu * (1 - mu)

    if (is.null(wts)) {
      num <- sum((y - mu)^2)
      den <- sum(denom)
    } else {
      w <- as.numeric(wts)
      num <- sum(w * (y - mu)^2)
      den <- sum(w * denom)
    }

    if (!is.finite(num) || !is.finite(den) || den <= 0) return(1 / eps_phi)
    d_hat <- num / den
    if (!is.finite(d_hat) || d_hat <= 0) return(1 / eps_phi)

    phi_hat <- (1 / d_hat) - 1
    if (!is.finite(phi_hat) || phi_hat <= 0) phi_hat <- 1 / eps_phi
    pmax(phi_hat, eps_phi)
  }

  # ---- core fitters --------------------------------------------------------
  fit_gamlss_one <- function(train_set, tr, wts_vec, ...) {
    dat <- as.data.frame(train_set)
    if (!("A" %in% names(dat))) stop("train_set must contain column 'A'.")

    y0 <- to_unit(dat$A)
    if (any(!is.finite(y0))) stop("Non-finite values encountered in transformed outcome.")
    dat$A_unit <- clip01_eps(y0, eps_fit)

    mu_f  <- stats::reformulate(tr$mu_rhs, response = "A_unit")
    sig_f <- stats::reformulate(tr$sigma_rhs)

    fit <- NULL
    tmp <- utils::capture.output({
      fit <- gamlss::gamlss(
        formula = mu_f,
        sigma.formula = sig_f,
        family = "BE",
        data = dat,
        weights = wts_vec,
        control = control,
        ...
      )
    }, type = "output")

    fit <- fix_family_call(fit, "BE")
    fit <- fix_data_call(fit, dat)
    fit
  }

  safe_fit_gamlss <- function(train_set, tr, wts_vec, ...) {
    tryCatch(
      list(ok = TRUE, fit = fit_gamlss_one(train_set = train_set, tr = tr, wts_vec = wts_vec, ...), err = NULL),
      error = function(e) list(ok = FALSE, fit = NULL, err = e)
    )
  }

  fit_fallback_beta_glm <- function(train_set, mu_rhs_raw, wts_vec) {
    dat <- as.data.frame(train_set)
    if (!("A" %in% names(dat))) stop("train_set must contain column 'A'.")

    y0 <- to_unit(dat$A)
    dat$A_unit <- clip01_eps(y0, eps_fit)

    built <- build_design_train_glm(mu_rhs_raw, dat)
    X <- built$X

    # Use glm.fit for speed and small objects; quasi helps with variance but we are
    # only using the mean model. The link is logit by default.
    fit <- suppressWarnings(stats::glm.fit(
      x = X,
      y = dat$A_unit,
      weights = wts_vec,
      family = stats::quasibinomial()
    ))

    fit_store <- strip_glm_coef(fit, built$x_cols)

    mu_hat <- predict_glm_stripped_mu(fit_store, X, eps_mu = eps_fit)
    phi_hat <- estimate_phi_beta(y = dat$A_unit, mu = mu_hat, wts = wts_vec, eps_phi = eps_fit)

    list(
      kind = "fallback_beta_glm",
      mu_fit = fit_store,
      design_spec = list(rhs = built$rhs, x_cols = built$x_cols),
      phi = as.numeric(phi_hat)
    )
  }

  list(
    method = "gamlss_beta",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) as.numeric(train_set$wts) else NULL

      fits <- vector("list", nrow(tune_grid))
      errors <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        tr <- tune_grid[k, , drop = FALSE]

        res <- safe_fit_gamlss(train_set = train_set, tr = tr, wts_vec = wts_vec, ...)
        if (!res$ok) errors[[k]] <- res$err

        fit_k <- res$fit
        if (!is.null(fit_k) && strip_fit) fit_k <- strip_gamlss(fit_k)

        if (!is.null(fit_k)) {
          fits[[k]] <- list(kind = "gamlss", gamlss_fit = fit_k)
        } else if (fallback == "beta_glm") {
          fb <- tryCatch(
            fit_fallback_beta_glm(train_set = train_set, mu_rhs_raw = tr$mu_rhs, wts_vec = wts_vec),
            error = function(e) list(kind = "failed", error = conditionMessage(e))
          )
          fits[[k]] <- fb
        } else {
          fits[[k]] <- list(
            kind = "failed",
            error = if (!is.null(errors[[k]])) conditionMessage(errors[[k]]) else NA_character_
          )
        }
      }

      list(fits = fits, errors = errors)
    },

    log_density = function(fit_bundle, newdata, eps = eps, ...) {
      nd <- as.data.frame(newdata)
      if (!("A" %in% names(nd))) stop("newdata must contain column 'A'.")

      a <- as.numeric(nd$A)
      y <- clip01_eps(to_unit(a), eps_fit)

      K <- length(fit_bundle$fits)
      out <- matrix(log(eps), nrow = nrow(nd), ncol = K)

      for (k in seq_len(K)) {
        obj <- fit_bundle$fits[[k]]

        if (identical(obj$kind, "gamlss")) {
          par <- tryCatch(
            gamlss::predictAll(obj$gamlss_fit, newdata = nd, type = "response"),
            error = function(e) NULL
          )
          if (is.null(par) || is.null(par$mu) || is.null(par$sigma)) next

          mu <- clip01_eps(as.numeric(par$mu), eps_fit)
          sigma <- pmax(as.numeric(par$sigma), sqrt(eps))

          ll_y <- tryCatch(
            dfun_BE(x = y, mu = mu, sigma = sigma, log = TRUE),
            error = function(e) rep(log(eps), length(y))
          )

          out[, k] <- pmax(ll_y - log_jac, log(eps))
          next
        }

        if (identical(obj$kind, "fallback_beta_glm")) {
          Xn <- build_design_new_glm(obj$design_spec, nd)
          mu <- predict_glm_stripped_mu(obj$mu_fit, Xn, eps_mu = eps_fit)
          phi <- pmax(as.numeric(obj$phi), eps_fit)

          a1 <- pmax(mu * phi, eps_fit)
          a2 <- pmax((1 - mu) * phi, eps_fit)

          ll_y <- tryCatch(
            stats::dbeta(y, shape1 = a1, shape2 = a2, log = TRUE),
            error = function(e) rep(log(eps), length(y))
          )

          out[, k] <- pmax(ll_y - log_jac, log(eps))
          next
        }

        # failed -> leave log(eps)
      }

      out
    },

    density = function(fit_bundle, newdata, eps = eps, ...) {
      exp(log_density(fit_bundle, newdata, eps = eps, ...))
    },

    fit_one = function(train_set, tune, ...) {
      if (length(tune) != 1L || is.na(tune) || tune < 1L || tune > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..nrow(tune_grid).")
      }

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) as.numeric(train_set$wts) else NULL

      tr <- tune_grid[as.integer(tune), , drop = FALSE]
      res <- safe_fit_gamlss(train_set = train_set, tr = tr, wts_vec = wts_vec, ...)
      fit_k <- res$fit
      err_k <- res$err

      if (!is.null(fit_k) && strip_fit) fit_k <- strip_gamlss(fit_k)

      fit_obj <- NULL
      if (!is.null(fit_k)) {
        fit_obj <- list(kind = "gamlss", gamlss_fit = fit_k)
      } else if (fallback == "beta_glm") {
        fit_obj <- tryCatch(
          fit_fallback_beta_glm(train_set = train_set, mu_rhs_raw = tr$mu_rhs, wts_vec = wts_vec),
          error = function(e) list(kind = "failed", error = conditionMessage(e))
        )
      } else {
        fit_obj <- list(kind = "failed", error = if (!is.null(err_k)) conditionMessage(err_k) else NA_character_)
      }

      list(fits = list(fit_obj), errors = list(err_k), tune = as.integer(tune))
    },

    select_fit = function(fit_bundle, tune) {
      if (!is.null(fit_bundle$fits) && length(fit_bundle$fits) >= tune) {
        fit_bundle$fits <- fit_bundle$fits[tune]
      }
      if (!is.null(fit_bundle$errors) && length(fit_bundle$errors) >= tune) {
        fit_bundle$errors <- fit_bundle$errors[tune]
      }
      fit_bundle
    },

    sample = function(fit_bundle, newdata, n_samp, seed = NULL, ...) {
      nd <- as.data.frame(newdata)
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      if (length(fits) != 1L) stop("sample() assumes K=1: fit_bundle$fits must have length 1 (selected model).")
      if (!is.null(seed)) set.seed(seed)

      obj <- fits[[1L]]
      n <- nrow(nd)
      out <- matrix(NA_real_, nrow = n, ncol = n_samp)

      if (identical(obj$kind, "gamlss")) {
        par <- tryCatch(
          gamlss::predictAll(obj$gamlss_fit, newdata = nd, type = "response"),
          error = function(e) NULL
        )
        if (is.null(par) || is.null(par$mu) || is.null(par$sigma)) {
          warning("gamlss_beta sample(): predictAll() failed; returning NA samples.")
          return(out)
        }

        mu <- clip01_eps(as.numeric(par$mu), eps_fit)
        sigma <- pmax(as.numeric(par$sigma), sqrt(eps))

        for (s in seq_len(n_samp)) {
          y_s <- tryCatch(
            rfun_BE(n = n, mu = mu, sigma = sigma),
            error = function(e) rep(NA_real_, n)
          )
          y_s <- clip01_eps(y_s, eps_fit)
          out[, s] <- from_unit(y_s)
        }
        return(out)
      }

      if (identical(obj$kind, "fallback_beta_glm")) {
        Xn <- build_design_new_glm(obj$design_spec, nd)
        mu <- predict_glm_stripped_mu(obj$mu_fit, Xn, eps_mu = eps_fit)
        phi <- pmax(as.numeric(obj$phi), eps_fit)

        a1 <- pmax(mu * phi, eps_fit)
        a2 <- pmax((1 - mu) * phi, eps_fit)

        for (s in seq_len(n_samp)) {
          y_s <- stats::rbeta(n, shape1 = a1, shape2 = a2)
          y_s <- clip01_eps(y_s, eps_fit)
          out[, s] <- from_unit(y_s)
        }
        return(out)
      }

      warning("gamlss_beta sample(): selected fit is not available; returning NA samples.")
      out
    }
  )
}

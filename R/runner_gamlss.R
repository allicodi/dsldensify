#' Create a GAMLSS runner for direct conditional density estimation
#'
#' Constructs a runner (learner adapter) compatible with the
#' dsldensify() / summarize_and_select() workflow for direct conditional density
#' estimation of a continuous outcome A given covariates W using GAMLSS.
#'
#' The runner fits one or more GAMLSS models via gamlss::gamlss(), allowing
#' specification of regression formulas for distribution parameters (mu, sigma,
#' and optionally nu and tau) and supports tuning over:
#'   - distribution family,
#'   - mu model (mu_rhs_list),
#'   - sigma model (sigma_rhs_list),
#'   - optional nu model (nu_rhs_list),
#'   - optional tau model (tau_rhs_list).
#'
#' This is a direct density learner. log_density() evaluates log f(A | W)
#' directly and is used for cross-validated model selection on the negative
#' log-likelihood scale.
#'
#' Robustness and fallback behavior
#'
#' GAMLSS fits can fail to converge or error for certain combinations of
#' families and formulas. This runner supports fallback behavior:
#'   - fallback = "normal_lm": if a GAMLSS fit fails, fall back to a homoscedastic
#'     normal linear model for the mean with an empirical residual standard
#'     deviation.
#'   - fallback = "none": retain the failure and return finite log-densities
#'     bounded below by log(eps). Error objects are stored for debugging.
#'
#' Numeric-only requirement
#'
#' This runner assumes all covariates referenced in the RHS formulas are
#' numeric. Factor handling and categorical smooth terms are not supported.
#'
#' Tuning grid and prediction layout
#'
#' The internal tune_grid is the Cartesian product of family_list, mu_rhs_list,
#' sigma_rhs_list, and optionally nu_rhs_list and tau_rhs_list. Each row
#' corresponds to a distinct specification. During cross-validation,
#' log_density() returns an n x K matrix of log-densities, where K = nrow(tune_grid),
#' aligned to .tune.
#'
#' Sampling from the fitted direct density model
#'
#' The runner provides a sample() method that generates draws
#'   A* ~ f_hat(· | W)
#' from the fitted conditional density.
#'
#' Sampling assumes the fit_bundle contains exactly one tuned fit
#' (length(fit_bundle$fits) == 1). This is the intended usage after model
#' selection (for example, after applying select_fit_tune() or fitting the
#' selected tuning index via fit_one()).
#'
#' The sample() method expects newdata in wide format containing only covariates W.
#' It returns an nrow(newdata) x n_samp numeric matrix.
#'
#' For GAMLSS fits, distribution parameters are obtained via
#' predictAll(type = "response") and sampling is performed using the
#' corresponding r<family>() function from gamlss.dist. For fallback normal
#' models, sampling uses rnorm() with mean predicted by the linear model and
#' constant sigma estimated on the training set.
#'
#' Lightweight fit objects
#'
#' When strip_fit = TRUE, fitted gamlss objects are stripped of large components
#' (responses, fitted values, residuals) before storage. This reduces memory usage
#' while preserving the ability to predict distribution parameters and evaluate
#' densities and samples.
#'
#' @param mu_rhs_list RHS specifications for the location parameter mu. May be
#'   one-sided formulas (for example, ~ W1 + W2) or character strings
#'   (for example, "W1 + W2"). Must have length at least 1.
#'
#' @param family_list Character vector of GAMLSS family names (for example, "NO",
#'   "TF", "BCCG"). Each family must have corresponding d<family>() and r<family>()
#'   functions available in gamlss.dist.
#'
#' @param sigma_rhs_list RHS specifications for the scale parameter sigma.
#'   Defaults to "1". May be formulas or character strings.
#'
#' @param nu_rhs_list Optional RHS specifications for the nu parameter. If NULL,
#'   nu is not modeled.
#'
#' @param tau_rhs_list Optional RHS specifications for the tau parameter. If NULL,
#'   tau is not modeled.
#'
#' @param use_weights_col Logical. If TRUE and the training data contain a column
#'   named wts, it is passed to gamlss::gamlss() via weights. Otherwise, fitting
#'   is unweighted.
#'
#' @param strip_fit Logical. If TRUE, strip fitted gamlss or fallback models
#'   before storing them.
#'
#' @param control A gamlss.control() object controlling the fitting procedure.
#'
#' @param eps Small positive constant used to bound log-densities away from
#'   -Inf when fits fail or densities underflow.
#'
#' @param fallback Character string specifying fallback behavior when a GAMLSS fit
#'   fails. One of "normal_lm" or "none".
#'
#' @param ... Additional arguments forwarded to gamlss::gamlss().
#'
#' @return A named list (runner) with elements:
#'   method: Character string "gamlss".
#'   tune_grid: Data frame enumerating all family / RHS combinations, including .tune.
#'   fit: Function fit(train_set, ...) returning a fit bundle containing all fits.
#'   log_density: Function log_density(fit_bundle, newdata, ...) returning an n x K
#'     matrix of log-densities.
#'   density: Function density(fit_bundle, newdata, ...) returning densities on
#'     the original scale.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected tuning index.
#'   select_fit: Function select_fit(fit_bundle, tune) extracting a single tuning configuration.
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing samples from the
#'     fitted conditional density (assumes length(fit_bundle$fits) == 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata in wide format containing:
#'   - a numeric outcome column A,
#'   - covariates referenced in the RHS specifications,
#'   - an optional weight column wts.
#'
#' @examples
#' mu_rhs <- list(~ W1 + W2, ~ W1 + W2 + W3)
#'
#' runner <- make_gamlss_runner(
#'   mu_rhs_list = mu_rhs,
#'   family_list = c("NO", "TF"),
#'   sigma_rhs_list = c("1", "~ W1"),
#'   fallback = "normal_lm"
#' )
#'
#' @export
make_gamlss_runner <- function(
  mu_rhs_list,
  family_list = c("NO"),
  sigma_rhs_list = c("1"),
  nu_rhs_list = NULL,
  tau_rhs_list = NULL,
  use_weights_col = TRUE,
  strip_fit = TRUE,
  control = gamlss::gamlss.control(n.cyc = 50, trace = FALSE),
  eps = 1e-12,
  fallback = c("normal_lm", "none"),
  ...
) {
  fallback <- match.arg(fallback)

  if (!requireNamespace("gamlss", quietly = TRUE)) {
    stop("Package 'gamlss' is required.")
  }
  if (!requireNamespace("gamlss.dist", quietly = TRUE)) {
    stop("Package 'gamlss.dist' is required.")
  }

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

  fam_chr <- as.character(family_list)
  if (length(fam_chr) < 1L) stop("family_list must have length >= 1")

  nu_rhs_chr  <- if (is.null(nu_rhs_list))  NA_character_ else rhs_to_chr(nu_rhs_list)
  tau_rhs_chr <- if (is.null(tau_rhs_list)) NA_character_ else rhs_to_chr(tau_rhs_list)

  tune_grid <- expand.grid(
    family = fam_chr,
    mu_rhs = mu_rhs_chr,
    sigma_rhs = sigma_rhs_chr,
    nu_rhs = nu_rhs_chr,
    tau_rhs = tau_rhs_chr,
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

  strip_lm <- function(fit) {
    fit$model <- NULL
    fit$y <- NULL
    fit$x <- NULL
    fit$qr <- NULL
    fit
  }

  get_dfun <- function(family) {
    fn <- paste0("d", family)
    if (exists(fn, where = asNamespace("gamlss.dist"), inherits = FALSE)) {
      return(get(fn, envir = asNamespace("gamlss.dist")))
    }
    if (exists(fn, mode = "function")) {
      return(get(fn, mode = "function"))
    }
    stop("Could not find density function ", fn, " (expected in gamlss.dist).")
  }

  get_rfun <- function(family) {
    fn <- paste0("r", family)
    if (exists(fn, where = asNamespace("gamlss.dist"), inherits = FALSE)) {
      return(get(fn, envir = asNamespace("gamlss.dist")))
    }
    if (exists(fn, mode = "function")) {
      return(get(fn, mode = "function"))
    }
    stop("Could not find random generator ", fn, " (expected in gamlss.dist).")
  }

  sigma_hat <- function(resid, w) {
    if (is.null(w)) sqrt(mean(resid^2)) else sqrt(sum(w * resid^2) / sum(w))
  }

  fit_fallback_normal <- function(train_set, mu_rhs, wts_vec) {
    f_mu <- stats::as.formula(paste0("A ~ ", mu_rhs))
    fit <- stats::lm(f_mu, data = train_set, weights = wts_vec)
    r <- stats::residuals(fit)
    sig <- pmax(sigma_hat(r, wts_vec), sqrt(eps))
    if (strip_fit) fit <- strip_lm(fit)
    list(mu_fit = fit, sigma = sig)
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

  fit_gamlss_one <- function(train_set, tr, wts_vec, ...) {
    dat <- as.data.frame(train_set)

    mu_f  <- stats::reformulate(tr$mu_rhs, response = "A")
    sig_f <- stats::reformulate(tr$sigma_rhs)
    nu_f  <- if (!is.na(tr$nu_rhs))  stats::reformulate(tr$nu_rhs)  else NULL
    tau_f <- if (!is.na(tr$tau_rhs)) stats::reformulate(tr$tau_rhs) else NULL

    fam_chr <- as.character(tr$family)

    fit <- NULL
    tmp <- utils::capture.output({
      fit <- gamlss::gamlss(
        formula = mu_f,
        sigma.formula = sig_f,
        nu.formula = nu_f,
        tau.formula = tau_f,
        family = fam_chr,
        data = dat,
        weights = wts_vec,
        control = control,
        ...
      )
    }, type = "output")

    fit <- fix_family_call(fit, fam_chr)
    fit <- fix_data_call(fit, dat)
    fit
  }

  safe_fit_gamlss <- function(train_set, tr, wts_vec, ...) {
    tryCatch(
      list(ok = TRUE, fit = fit_gamlss_one(train_set = train_set, tr = tr, wts_vec = wts_vec, ...), err = NULL),
      error = function(e) list(ok = FALSE, fit = NULL, err = e)
    )
  }

  list(
    method = "gamlss",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      fits <- vector("list", nrow(tune_grid))
      errors <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        tr <- tune_grid[k, , drop = FALSE]
        fam <- as.character(tr$family)

        res <- safe_fit_gamlss(train_set = train_set, tr = tr, wts_vec = wts_vec, ...)
        fit_k <- res$fit

        if (!is.null(fit_k) && strip_fit) fit_k <- strip_gamlss(fit_k)

        if (!res$ok) {
          errors[[k]] <- res$err
        }

        if (is.null(fit_k) && fallback == "normal_lm") {
          fb <- fit_fallback_normal(train_set, mu_rhs = tr$mu_rhs, wts_vec = wts_vec)
          fits[[k]] <- list(
            kind = "fallback_normal",
            fallback = fb,
            family = "NO"
          )
        } else if (!is.null(fit_k)) {
          fits[[k]] <- list(
            kind = "gamlss",
            gamlss_fit = fit_k,
            family = fam
          )
        } else {
          fits[[k]] <- list(
            kind = "failed",
            family = fam,
            error = if (!is.null(errors[[k]])) conditionMessage(errors[[k]]) else NA_character_
          )
        }
      }

      out <- list(fits = fits)
      if (fallback == "none") out$errors <- errors
      out
    },

    log_density = function(fit_bundle, newdata, eps = eps, ...) {
      nd <- as.data.frame(newdata)
      if (!("A" %in% names(nd))) stop("newdata must contain column 'A'.")

      K <- length(fit_bundle$fits)
      out <- matrix(log(eps), nrow = nrow(nd), ncol = K)

      for (k in seq_len(K)) {
        obj <- fit_bundle$fits[[k]]

        if (identical(obj$kind, "gamlss")) {
          dfun <- get_dfun(obj$family)

          par <- tryCatch(
            gamlss::predictAll(obj$gamlss_fit, newdata = nd, type = "response"),
            error = function(e) NULL
          )
          if (is.null(par)) next

          args <- list(x = nd$A, log = TRUE)
          if (!is.null(par$mu))    args$mu <- par$mu
          if (!is.null(par$sigma)) args$sigma <- par$sigma
          if (!is.null(par$nu))    args$nu <- par$nu
          if (!is.null(par$tau))   args$tau <- par$tau

          lk <- tryCatch(do.call(dfun, args), error = function(e) rep(log(eps), nrow(nd)))
          out[, k] <- pmax(lk, log(eps))
          next
        }

        if (identical(obj$kind, "fallback_normal")) {
          mu <- stats::predict(obj$fallback$mu_fit, newdata = nd)
          sd <- pmax(obj$fallback$sigma, sqrt(eps))
          out[, k] <- pmax(stats::dnorm(nd$A, mean = mu, sd = sd, log = TRUE), log(eps))
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
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      if (length(tune) != 1L || is.na(tune) || tune < 1L || tune > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..nrow(tune_grid).")
      }
      tr <- tune_grid[tune, , drop = FALSE]
      fam <- as.character(tr$family)

      res <- safe_fit_gamlss(train_set = train_set, tr = tr, wts_vec = wts_vec, ...)
      fit_k <- res$fit
      err_k <- res$err

      if (!is.null(fit_k) && strip_fit) fit_k <- strip_gamlss(fit_k)

      if (is.null(fit_k) && fallback == "normal_lm") {
        fb <- fit_fallback_normal(train_set, mu_rhs = tr$mu_rhs, wts_vec = wts_vec)
        fit_obj <- list(kind = "fallback_normal", fallback = fb, family = "NO")
      } else if (!is.null(fit_k)) {
        fit_obj <- list(kind = "gamlss", gamlss_fit = fit_k, family = fam)
      } else {
        fit_obj <- list(
          kind = "failed",
          family = fam,
          error = if (!is.null(err_k)) conditionMessage(err_k) else NA_character_
        )
      }

      out <- list(fits = list(fit_obj), tune = as.integer(tune))
      if (fallback == "none") out$errors <- list(err_k)
      out
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

      if (identical(obj$kind, "fallback_normal")) {
        mu <- stats::predict(obj$fallback$mu_fit, newdata = nd)
        sd <- pmax(obj$fallback$sigma, sqrt(eps))
        for (s in seq_len(n_samp)) {
          out[, s] <- stats::rnorm(n, mean = mu, sd = sd)
        }
        return(out)
      }

      if (identical(obj$kind, "gamlss")) {
        rfun <- get_rfun(obj$family)

        par <- tryCatch(
          gamlss::predictAll(obj$gamlss_fit, newdata = nd, type = "response"),
          error = function(e) NULL
        )
        if (is.null(par)) {
          warning("gamlss sample(): predictAll() failed; returning NA samples.")
          return(out)
        }

        args_base <- list()
        if (!is.null(par$mu))    args_base$mu <- par$mu
        if (!is.null(par$sigma)) args_base$sigma <- par$sigma
        if (!is.null(par$nu))    args_base$nu <- par$nu
        if (!is.null(par$tau))   args_base$tau <- par$tau

        for (s in seq_len(n_samp)) {
          args <- c(list(n = n), args_base)
          out[, s] <- tryCatch(
            do.call(rfun, args),
            error = function(e) rep(NA_real_, n)
          )
        }

        return(out)
      }

      if (identical(obj$kind, "failed")) {
        warning("gamlss sample(): selected fit is marked as failed; returning NA samples.")
        return(out)
      }

      stop("Unexpected fit kind in gamlss runner: ", as.character(obj$kind))
    }
  )
}

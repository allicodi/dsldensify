#' Create a GAMLSS runner for direct conditional density estimation
#'
#' @description
#' Constructs a **runner** (learner adapter) compatible with the
#' \code{dsldensify()} / \code{summarize_and_select()} workflow for **direct
#' conditional density estimation** of a continuous outcome \code{A | W} using
#' generalized additive models for location, scale, and shape (GAMLSS).
#'
#' The runner fits one or more GAMLSS models via \code{gamlss::gamlss()}, allowing
#' flexible specification of regression formulas for distribution parameters
#' (e.g., mean, scale, skewness, kurtosis) and supports tuning over:
#' \itemize{
#'   \item distribution family,
#'   \item mean model (\code{mu_rhs}),
#'   \item scale model (\code{sigma_rhs}),
#'   \item optional shape models (\code{nu_rhs}, \code{tau_rhs}).
#' }
#'
#' This runner is intended for **direct density learners** (not hazard-based):
#' it returns log-densities \eqn{\log f(A | W)} directly, which are used for
#' cross-validated model selection on the negative log-likelihood scale.
#'
#' @section Robustness and fallback behavior:
#' GAMLSS fits can occasionally fail to converge or error for certain combinations
#' of families and formulas, especially in small samples or during cross-validation.
#' To make large CV routines robust, this runner supports an optional **fallback
#' mechanism**:
#' \itemize{
#'   \item \code{fallback = "normal_lm"}: if a GAMLSS fit fails, fall back to a
#'         homoscedastic normal linear model for the mean with an empirical
#'         residual standard deviation.
#'   \item \code{fallback = "none"}: retain the failure and return a finite but
#'         extremely small log-density (via \code{eps}); error objects are stored
#'         for debugging.
#' }
#'
#' @section Numeric-only requirement:
#' This runner assumes that all covariates referenced in the RHS formulas are
#' **numeric**. Factor handling, contrasts, and categorical smoothing terms are
#' not supported. Inputs are assumed to be preprocessed appropriately.
#'
#' @section Tuning grid and prediction layout:
#' The internal \code{tune_grid} is the Cartesian product of:
#' \itemize{
#'   \item \code{family_list},
#'   \item \code{mu_rhs_list},
#'   \item \code{sigma_rhs_list},
#'   \item \code{nu_rhs_list} (if provided),
#'   \item \code{tau_rhs_list} (if provided).
#' }
#' Each row corresponds to a distinct GAMLSS specification. During
#' cross-validation, \code{log_density()} returns an \code{n x K} matrix of
#' log-densities, where \code{K = nrow(tune_grid)}, aligned to the tuning grid.
#'
#' @section Lightweight fit objects:
#' When \code{strip_fit = TRUE}, fitted \code{gamlss} objects are stripped of
#' large components (responses, fitted values, residuals) before storage.
#' This substantially reduces memory usage when saving fold-specific CV fits,
#' while preserving the ability to:
#' \itemize{
#'   \item predict distribution parameters via \code{predictAll()},
#'   \item evaluate the log-density via the appropriate \code{d<family>()} function.
#' }
#'
#' @param mu_rhs_list A list of RHS specifications for the location (mean)
#' parameter \code{mu}, either as one-sided formulas (e.g., \code{~ W1 + W2}) or
#' character strings (e.g., \code{"W1 + W2"}). Must have length \eqn{\ge 1}.
#'
#' @param family_list Character vector of GAMLSS family names (e.g., \code{"NO"},
#' \code{"TF"}, \code{"BCCG"}). Each family must have a corresponding density
#' function \code{d<family>()} available in \pkg{gamlss.dist}.
#'
#' @param sigma_rhs_list RHS specifications for the scale parameter \code{sigma}.
#' Defaults to \code{"1"} (constant scale). May be formulas or character strings.
#'
#' @param nu_rhs_list Optional RHS specifications for the \code{nu} parameter
#' (e.g., skewness). If \code{NULL}, \code{nu} is not modeled.
#'
#' @param tau_rhs_list Optional RHS specifications for the \code{tau} parameter
#' (e.g., kurtosis). If \code{NULL}, \code{tau} is not modeled.
#'
#' @param use_weights_col Logical. If \code{TRUE} and the training data contain a
#' column named \code{wts}, it is passed to \code{gamlss::gamlss()} via the
#' \code{weights} argument. Otherwise, fitting is unweighted.
#'
#' @param strip_fit Logical. If \code{TRUE} (default), apply a stripping step to
#' fitted GAMLSS or fallback models before storing them.
#'
#' @param control A \code{gamlss.control()} object controlling the GAMLSS fitting
#' procedure (e.g., number of iterations, tracing). Defaults to a conservative,
#' quiet configuration suitable for CV.
#'
#' @param eps Small positive constant used to bound log-densities away from
#' \eqn{-\infty} when fits fail or densities underflow.
#'
#' @param fallback Character string specifying fallback behavior when a GAMLSS
#' fit fails. One of:
#' \describe{
#'   \item{\code{"normal_lm"}}{Fall back to a homoscedastic normal linear model
#'   for \code{mu}.}
#'   \item{\code{"none"}}{Do not fall back; failed fits return log(eps).}
#' }
#'
#' @param ... Additional arguments forwarded to \code{gamlss::gamlss()}.
#'
#' @return A named list (runner) with elements:
#' \describe{
#'   \item{method}{Character string \code{"gamlss"}.}
#'   \item{tune_grid}{Data frame enumerating all family / RHS combinations, with
#'   column \code{.tune}.}
#'   \item{fit}{Function \code{fit(train_set, ...)} returning a fit bundle
#'   containing all fitted (or failed/fallback) models.}
#'   \item{log_density}{Function \code{log_density(fit_bundle, newdata, ...)}
#'   returning an \code{n x K} matrix of log-densities.}
#'   \item{density}{Function \code{density(fit_bundle, newdata, ...)} returning
#'   densities on the original scale.}
#'   \item{fit_one}{Function \code{fit_one(train_set, tune, ...)} fitting only
#'   the selected tuning index.}
#'   \item{select_fit}{Function for extracting a single tuning from a fit bundle.}
#' }
#'
#' @details
#' ## Data requirements
#' The runner expects \code{train_set} and \code{newdata} in **wide format**
#' containing:
#' \itemize{
#'   \item a numeric outcome column \code{A},
#'   \item covariates referenced in the RHS specifications,
#'   \item an optional weight column \code{wts}.
#' }
#'
#' ## Density evaluation
#' For GAMLSS fits, distribution parameters are obtained via
#' \code{predictAll(type = "response")}, and the log-density is evaluated using
#' the corresponding \code{d<family>()} function from \pkg{gamlss.dist}. For
#' fallback normal models, densities are computed using \code{dnorm()}.
#'
#' @examples
#' mu_rhs <- list(~ W1 + W2, ~ splines::ns(W1, df = 3) + W2)
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
    stop("Package 'gamlss.dist' is required (for d<family>() density functions).")
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
    # store a self-contained expression in the call:
    # get("<FAM>", envir = asNamespace("gamlss.dist"))
    fit$call$family <- substitute(
      get(FAM, envir = asNamespace("gamlss.dist")),
      list(FAM = fam_chr)
    )
    fit
  }

  fix_data_call <- function(fit, dat) {
    # Store data where predict.gamlss can find it via 'object'
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

    # CRITICAL: make call self-contained for later prediction
    fit <- fix_family_call(fit, fam_chr)
    fit <- fix_data_call(fit, dat)

    fit
  }


  # ---- helper: safe fit wrapper that preserves errors when fallback="none" ----
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
      errors <- vector("list", nrow(tune_grid)) # store errors when fallback="none"

      for (k in seq_len(nrow(tune_grid))) {
        tr <- tune_grid[k, , drop = FALSE]
        fam <- as.character(tr$family)

        res <- safe_fit_gamlss(train_set = train_set, tr = tr, wts_vec = wts_vec, ...)
        fit_k <- res$fit

        if (!is.null(fit_k) && strip_fit) fit_k <- strip_gamlss(fit_k)

        if (!res$ok) {
          # record the error
          errors[[k]] <- res$err
        }

        # if failed, fallback
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
          # fallback="none": keep failure marker AND error (if any)
          fits[[k]] <- list(
            kind = "failed",
            family = fam,
            error = if (!is.null(errors[[k]])) conditionMessage(errors[[k]]) else NA_character_
          )
        }
      }

      out <- list(fits = fits)
      if (fallback == "none") out$errors <- errors  # keep full condition objects for debugging
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

        # "failed" -> leave log(eps)
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
        # fallback="none": keep the error for inspection (and still allow finite predictions)
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
    }
  )
}

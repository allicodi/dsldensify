#' Create a mixture-of-experts location-scale residual KDE runner for direct conditional density estimation
#'
#' Constructs a runner (learner adapter) compatible with the
#' dsldensify() / run_direct_setting() / summarize_and_select() workflow for
#' direct conditional density estimation of a continuous outcome \eqn{A} given
#' covariates \eqn{W}.
#'
#' This runner models the conditional density as a finite mixture of
#' location-scale residual density components ("experts"):
#' \deqn{f(a \mid W) = \sum_{k=1}^K \pi_k(W)\, f_k(a \mid W),}
#' where mixture weights \eqn{\pi_k(W)} satisfy \eqn{\pi_k(W) \ge 0} and
#' \eqn{\sum_{k=1}^K \pi_k(W) = 1}.
#'
#' Each expert \eqn{f_k(a \mid W)} uses a location-scale decomposition with a
#' nonparametric residual density on the standardized residual scale:
#' \deqn{A = \mu_k(W) + \sigma_k(W)\,\varepsilon_k,}
#' \deqn{f_k(a \mid W) = g_k\!\left(\frac{a - \mu_k(W)}{\sigma_k(W)}\right)\,\frac{1}{\sigma_k(W)},}
#' where \eqn{\mu_k(W)} is a component-specific mean function, \eqn{\sigma_k(W) > 0}
#' is a component-specific (or shared) scale parameter, and \eqn{g_k} is a
#' univariate residual density estimated by kernel density estimation (KDE)
#' on standardized residuals.
#'
#' Mixture weights (gating)
#'
#' The mixture weights \eqn{\pi_k(W)} are controlled by \code{gate_grid}:
#' \itemize{
#' \item \code{"const"}: constant mixing proportions, \eqn{\pi_k(W) = \pi_k}.
#' \item \code{"glm"}: multinomial logistic gating on the design matrix built
#'   from \code{rhs_list}, using a baseline-category parameterization and
#'   softmax normalization.
#' }
#'
#' Estimation by EM
#'
#' For each tuning configuration, the runner fits the mixture using an
#' expectation-maximization (EM) algorithm on the observed-data log-likelihood.
#' Let \eqn{r_{ik}} denote the responsibility of component \eqn{k} for observation
#' \eqn{i}. The EM iterations proceed as follows:
#'
#' E-step:
#' \deqn{r_{ik} \propto \pi_k(W_i)\, g_k\!\left(\frac{A_i - \mu_k(W_i)}{\sigma_k}\right)\,\frac{1}{\sigma_k},}
#' where the proportionality constant is chosen so that \eqn{\sum_{k=1}^K r_{ik} = 1}.
#'
#' M-step:
#' \enumerate{
#' \item Update component mean functions \eqn{\mu_k(W)} by weighted least squares
#'   regression of \eqn{A} on the design matrix induced by \code{rhs_list},
#'   using weights \eqn{r_{ik}}.
#' \item Update component scales \eqn{\sigma_k} according to \code{var_grid}:
#'   either a common shared scale across components (\code{"shared"}) or
#'   component-specific scales (\code{"by_component"}). Predicted scales are
#'   floored by \code{min_sigma} for numerical stability.
#' \item Update component residual KDEs \eqn{g_k} by fitting a univariate KDE on
#'   standardized residuals \eqn{\hat\varepsilon_{ik} = (A_i - \hat\mu_k(W_i))/\hat\sigma_k}
#'   using weights \eqn{r_{ik}}.
#' \item Update mixture weights \eqn{\pi_k(W)} using either constant proportions
#'   (weighted averages of responsibilities) or multinomial logistic gating
#'   on the design matrix built from \code{rhs_list}.
#' }
#'
#' Convergence is assessed using the relative change in the observed-data log-likelihood,
#' stopping when the criterion falls below \code{tol} or after \code{max_iter} iterations.
#'
#' Kernel density estimation on residuals
#'
#' For each component \eqn{k}, the residual density \eqn{g_k} is estimated by
#' \code{stats::density()} on standardized residuals. The KDE bandwidth rule is
#' controlled by \code{kde_bw_method_grid} and multiplicative scaling by
#' \code{kde_adjust_grid}. The estimated density is stored as a linear interpolant
#' via \code{approxfun()} for evaluation, and an inverse-CDF interpolant is stored
#' for sampling. The inverse-CDF interpolant is built from the numerically
#' integrated KDE CDF on the residual grid.
#'
#' Model selection via log_density()
#'
#' Model selection uses likelihood-based scoring via log_density(): for each
#' tuning row and each observation, log_density() evaluates
#' \deqn{\log \hat f(A_i \mid W_i),}
#' where \eqn{\hat f} is the fitted mixture-of-experts KDE density described above.
#' During cross-validation, \code{log_density()} returns an \eqn{n \times K}
#' matrix of log-densities aligned to \code{tune_grid$.tune}, where
#' \eqn{K = nrow(tune_grid)}.
#'
#' Sampling from the fitted model
#'
#' The runner provides a \code{sample()} method that generates draws
#' \deqn{A^\ast \sim \hat f(\cdot \mid W)}
#' by first sampling a component label \eqn{Z \in \{1,\dots,K\}} from
#' \eqn{\hat\pi(W)} and then sampling \eqn{\varepsilon^\ast} from the corresponding
#' residual KDE via inverse-CDF sampling. The final draw is formed as
#' \deqn{A^\ast = \hat\mu_Z(W) + \hat\sigma_Z\,\varepsilon^\ast.}
#' Sampling assumes the fit bundle contains exactly one tuned fit
#' (length(fit_bundle$fits) == 1), which is the intended post-selection usage.
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All variables
#' referenced in \code{rhs_list} must be numeric. Factor handling is not
#' supported and variables are not coerced internally.
#'
#' Tuning grid
#'
#' The tuning grid is the Cartesian product of:
#' \itemize{
#' \item RHS specifications from \code{rhs_list},
#' \item mixture size \code{K_grid},
#' \item gating model choice \code{gate_grid},
#' \item variance structure \code{var_grid},
#' \item KDE bandwidth rule \code{kde_bw_method_grid} and adjustment \code{kde_adjust_grid},
#' \item initialization strategy \code{init_grid}.
#' }
#'
#' Stabilization and numerical safety
#'
#' Densities on the residual scale are bounded below by \code{eps} before taking logs.
#' Mixture weights are bounded below by \code{min_pi} and renormalized to sum to one.
#' Scale parameters are bounded below by \code{min_sigma}. These stabilizations are
#' applied both during EM updates and during prediction.
#'
#' @param rhs_list A list of RHS specifications, either as one-sided formulas
#'   (for example, \code{~ x1 + x2}) or as character strings
#'   (for example, \code{"x1 + x2"}). These RHS are used to build the design
#'   matrix for both expert mean models \eqn{\mu_k(W)} and, when \code{gate_grid}
#'   includes \code{"glm"}, the gating model for \eqn{\pi_k(W)}.
#'
#' @param K_grid Integer vector of candidate mixture sizes \eqn{K}.
#'
#' @param gate_grid Character vector specifying gating model choices for
#'   \eqn{\pi_k(W)}. Supported values are \code{"const"} and \code{"glm"}.
#'
#' @param var_grid Character vector specifying the scale structure.
#'   Supported values are \code{"shared"} (one common \eqn{\sigma} across components)
#'   and \code{"by_component"} (component-specific \eqn{\sigma_k}).
#'
#' @param kde_bw_method_grid Character vector of KDE bandwidth rules. Supported
#'   values are \code{"nrd0"} and \code{"SJ"}. These bandwidth rules are applied
#'   to standardized residuals within each component.
#'
#' @param kde_adjust_grid Numeric vector of multiplicative bandwidth adjustments.
#'
#' @param init_grid Character vector specifying initialization strategies for
#'   responsibilities. Supported values are \code{"kmeansA"} (k-means on \eqn{A})
#'   and \code{"random"} (random component assignment).
#'
#' @param max_iter Integer maximum number of EM iterations for each tuning row.
#'
#' @param tol Nonnegative convergence tolerance for relative change in the
#'   observed-data log-likelihood.
#'
#' @param eps Small positive constant used to bound residual KDE densities away
#'   from zero and log-densities away from \eqn{-\infty}.
#'
#' @param min_sigma Small positive constant used as a floor for the estimated
#'   scale parameter(s) \eqn{\sigma} or \eqn{\sigma_k}.
#'
#' @param min_pi Small positive constant used as a floor for mixture weights
#'   \eqn{\pi_k(W)} prior to renormalization.
#'
#' @param kde_n Integer number of grid points used in \code{stats::density()}
#'   for each component KDE.
#'
#' @param kde_trim Nonnegative scalar controlling the residual-domain padding
#'   used when forming the KDE evaluation grid. The grid endpoints are set using
#'   high and low residual quantiles plus \code{kde_trim} times the residual
#'   standard deviation.
#'
#' @param strip_fit Logical. If TRUE (default), store lightweight
#'   coefficient-based representations sufficient for prediction and sampling.
#'   If FALSE, store the full internal fit objects produced during EM.
#'
#' @param seed Optional integer seed. If provided, each tuning row is fit with
#'   a deterministic seed offset (for example, \code{seed + .tune}) to improve
#'   reproducibility.
#'
#' @return A named list (runner) with elements:
#'   method: Character string \code{"mix_locscale_kde"}.
#'   tune_grid: Data frame describing the tuning grid, including \code{.tune}.
#'   fit: Function \code{fit(train_set, ...)} returning a fit bundle.
#'   log_density: Function \code{log_density(fit_bundle, newdata, ...)} returning
#'     an \eqn{n \times K} matrix of log-densities.
#'   fit_one: Function \code{fit_one(train_set, tune, ...)} fitting only the
#'     selected tuning index.
#'   sample: Function \code{sample(fit_bundle, newdata, n_samp, ...)} drawing
#'     samples (assumes \eqn{K = 1}).
#'
#' Data requirements
#'
#' The runner expects \code{train_set} and \code{newdata} in wide format containing:
#' \itemize{
#' \item a numeric outcome column \code{A} (required for \code{fit()} and \code{log_density()}),
#' \item covariates referenced in \code{rhs_list}.
#' }
#' \code{sample()} expects \code{newdata} to contain only covariates \code{W}
#' (it must not require an \code{A} column).
#'
#' @examples
#' runner <- make_mix_locscale_kde_runner(
#'   rhs_list = list(~ x1 + x2),
#'   K_grid = c(1L, 2L),
#'   gate_grid = c("const", "glm"),
#'   var_grid = c("shared"),
#'   kde_bw_method_grid = c("nrd0"),
#'   kde_adjust_grid = c(1.0),
#'   init_grid = c("kmeansA"),
#'   strip_fit = TRUE,
#'   eps = 1e-10,
#'   seed = 123
#' )
#'
#' @export

make_mix_locscale_kde_runner <- function(
  rhs_list,
  K_grid = c(1L, 2L, 3L),
  gate_grid = c("const", "glm"),
  var_grid  = c("by_component", "shared"),

  kde_bw_method_grid = c("nrd0", "SJ"),
  kde_adjust_grid = c(1.0, 1.3),

  init_grid = c("kmeansA", "random"),

  max_iter = 200L,
  tol = 1e-6,

  eps = 1e-15,
  min_sigma = 1e-6,
  min_pi = 1e-10,

  kde_n = 1024L,
  kde_trim = 5,           # KDE support: [q_lo - trim*sd, q_hi + trim*sd] in residual space

  strip_fit = TRUE,
  seed = NULL
) {
  stopifnot(requireNamespace("stats", quietly = TRUE))

  # ---- rhs parsing into strings for tune grid ----
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  K_grid <- as.integer(K_grid)
  if (anyNA(K_grid) || any(K_grid < 1L)) stop("K_grid must be positive integers")

  gate_grid <- match.arg(gate_grid, several.ok = TRUE)
  var_grid  <- match.arg(var_grid,  several.ok = TRUE)
  init_grid <- match.arg(init_grid, several.ok = TRUE)

  tune_grid <- expand.grid(
    rhs = rhs_chr,
    K = K_grid,
    gate = gate_grid,
    var = var_grid,
    kde_bw_method = kde_bw_method_grid,
    kde_adjust = kde_adjust_grid,
    init = init_grid,
    stringsAsFactors = FALSE
  )
  tune_grid$.tune <- seq_len(nrow(tune_grid))
  tune_grid <- tune_grid[, c(".tune","rhs","K","gate","var","kde_bw_method","kde_adjust","init")]

  # ---- helpers ----
  logsumexp_vec <- function(x) {
    m <- max(x)
    m + log(sum(exp(x - m)))
  }

  softmax_rows <- function(eta_mat) {
    row_max <- apply(eta_mat, 1, max)
    z <- exp(eta_mat - row_max)
    z / rowSums(z)
  }

  build_terms <- function(rhs) stats::as.formula(paste0("~", rhs))

  build_X <- function(W, terms_obj) {
    stats::model.matrix(terms_obj, data = W)
  }

  fit_wls <- function(X, y, w) {
    out <- stats::lm.wfit(x = X, y = y, w = w)
    list(beta = out$coefficients, mu = as.vector(X %*% out$coefficients))
  }

  # multinomial logit gating (baseline class K), or constant
  fit_gate <- function(X, r, gate_type, min_pi) {
    n <- nrow(X); K <- ncol(r)
    if (gate_type == "const" || K == 1L) {
      pi <- colMeans(r)
      pi <- pmax(pi, min_pi); pi <- pi / sum(pi)
      return(list(type = "const", pi = pi, coef = NULL))
    }

    p <- ncol(X)
    B <- matrix(0, nrow = p, ncol = K - 1L)  # IRLS init

    for (iter in seq_len(50L)) {
      eta <- cbind(X %*% B, 0)
      pi_hat <- softmax_rows(eta)

      G <- matrix(0, nrow = p, ncol = K - 1L)
      H <- matrix(0, nrow = p * (K - 1L), ncol = p * (K - 1L))

      for (k in seq_len(K - 1L)) {
        G[, k] <- crossprod(X, r[, k] - pi_hat[, k])
      }

      for (k in seq_len(K - 1L)) {
        for (l in seq_len(K - 1L)) {
          w_kl <- if (k == l) pi_hat[, k] * (1 - pi_hat[, k]) else -pi_hat[, k] * pi_hat[, l]
          Xw <- X * as.vector(w_kl)
          H_block <- crossprod(X, Xw)
          rr <- ((k - 1L) * p + 1L):((k) * p)
          cc <- ((l - 1L) * p + 1L):((l) * p)
          H[rr, cc] <- H_block
        }
      }

      diag(H) <- diag(H) + 1e-8
      step <- tryCatch(solve(H, as.vector(G)), error = function(e) NULL)
      if (is.null(step)) break
      B_new <- B + matrix(step, nrow = p, ncol = K - 1L)
      if (max(abs(B_new - B)) < 1e-6) { B <- B_new; break }
      B <- B_new
    }

    list(type = "glm", pi = NULL, coef = B)
  }

  predict_gate <- function(gate_fit, X, K, min_pi) {
    if (K == 1L) return(matrix(1, nrow = nrow(X), ncol = 1L))
    if (gate_fit$type == "const") {
      pi <- gate_fit$pi
      mat <- matrix(rep(pi, each = nrow(X)), nrow = nrow(X))
      mat <- pmax(mat, min_pi); mat <- mat / rowSums(mat)
      return(mat)
    }
    B <- gate_fit$coef
    eta <- cbind(X %*% B, 0)
    pi_hat <- softmax_rows(eta)
    pi_hat <- pmax(pi_hat, min_pi); pi_hat <- pi_hat / rowSums(pi_hat)
    pi_hat
  }

  # KDE builder for residuals (supports weights if stats::density has weights)
  has_density_weights <- "weights" %in% names(formals(stats::density.default))

  build_kde <- function(eps_hat, w, bw_method, bw_adjust, n = 1024L, trim = 5) {
    eps_hat <- as.numeric(eps_hat)
    w <- as.numeric(w)
    w[!is.finite(w)] <- 0
    if (sum(w) <= 0) w <- rep(1, length(eps_hat))
    w <- w / sum(w)

    s <- stats::sd(eps_hat)
    if (!is.finite(s) || s <= 0) s <- 1
    qlo <- stats::quantile(eps_hat, 0.01, names = FALSE, type = 8)
    qhi <- stats::quantile(eps_hat, 0.99, names = FALSE, type = 8)
    from <- qlo - trim * s
    to   <- qhi + trim * s

    # ---- choose bandwidth (numeric) in a way that respects weights ----
    # base R bw selectors ignore weights; easiest workaround: weighted resample for bw selection
    m <- length(eps_hat)
    idx_bw <- sample.int(m, size = m, replace = TRUE, prob = w)
    eps_bw <- eps_hat[idx_bw]

    bw_num <- bw_method
    if (is.character(bw_method) && length(bw_method) == 1L) {
      if (bw_method == "nrd0") {
        bw_num <- stats::bw.nrd0(eps_bw)
      } else if (bw_method == "SJ") {
        bw_num <- stats::bw.SJ(eps_bw)
      } else {
        stop("Unsupported bw_method: ", bw_method)
      }
    }
    bw_num <- as.numeric(bw_num) * as.numeric(bw_adjust)
    if (!is.finite(bw_num) || bw_num <= 0) bw_num <- stats::bw.nrd0(eps_bw)

    # ---- density with weights, but numeric bw (no warning) ----
    dens_obj <- stats::density(
      eps_hat,
      weights = w,
      bw = bw_num,
      n = n,
      from = from,
      to = to
    )

    x <- dens_obj$x
    y <- pmax(dens_obj$y, 0)

    # normalize (numerical)
    dx <- x[2L] - x[1L]
    mass <- sum(y) * dx
    if (!is.finite(mass) || mass <= 0) {
      y <- rep(1 / (length(y) * dx), length(y))
    } else {
      y <- y / mass
    }

    # build CDF
    cdf <- cumsum(y) * dx
    cdf <- pmin(pmax(cdf, 0), 1)
    cdf[1] <- 0
    cdf[length(cdf)] <- 1

    # density evaluator
    dfun <- stats::approxfun(x, y, rule = 2)

    # ---- inverse CDF: enforce strictly increasing x-grid for approxfun ----
    # keep last x for each unique cdf value to preserve monotonicity
    u <- !duplicated(cdf, fromLast = TRUE)
    cdf_u <- cdf[u]
    x_u <- x[u]

    # handle pathological case where everything collapses
    if (length(cdf_u) < 2L) {
      cdf_u <- c(0, 1)
      x_u <- c(min(x), max(x))
    } else {
      cdf_u[1] <- 0
      cdf_u[length(cdf_u)] <- 1
    }

    qfun <- stats::approxfun(cdf_u, x_u, rule = 2)

    list(x = x, y = y, cdf = cdf, dfun = dfun, qfun = qfun, bw = bw_num)
  }


  init_resp <- function(A, K, init, seed) {
    n <- length(A)
    if (!is.null(seed)) set.seed(seed)
    if (K == 1L) return(matrix(1, n, 1L))
    if (init == "kmeansA") {
      km <- stats::kmeans(A, centers = K, nstart = 5)
      z <- km$cluster
    } else {
      z <- sample.int(K, n, replace = TRUE)
    }
    r <- matrix(0, nrow = n, ncol = K)
    r[cbind(seq_len(n), z)] <- 1
    r
  }

  fit_one_em <- function(A, W, rhs, K, gate, var, bw_method, bw_adjust, init, seed) {
    terms_obj <- build_terms(rhs)
    X <- build_X(W, terms_obj)
    n <- length(A)

    r <- init_resp(A, K, init, seed)

    beta_list <- vector("list", K)
    mu <- matrix(0, nrow = n, ncol = K)
    sigma <- rep(stats::sd(A), K)
    sigma[!is.finite(sigma) | sigma <= 0] <- 1

    kde_list <- vector("list", K)
    gate_fit <- list(type = "const", pi = rep(1 / K, K), coef = NULL)

    ll_prev <- -Inf

    for (iter in seq_len(max_iter)) {
      # ----- M step: means -----
      for (k in seq_len(K)) {
        fk <- fit_wls(X, A, w = r[, k])
        beta_list[[k]] <- fk$beta
        mu[, k] <- fk$mu
      }

      # ----- M step: sigmas -----
      if (K == 1L) {
        s <- stats::sd(A - mu[, 1])
        sigma <- rep(max(s, min_sigma), 1L)
      } else if (var == "shared") {
        num <- 0; den <- 0
        for (k in seq_len(K)) {
          res2 <- (A - mu[, k])^2
          num <- num + sum(r[, k] * res2)
          den <- den + sum(r[, k])
        }
        s2 <- num / max(den, 1e-12)
        sigma <- rep(max(sqrt(s2), min_sigma), K)
      } else {
        for (k in seq_len(K)) {
          res2 <- (A - mu[, k])^2
          s2 <- sum(r[, k] * res2) / max(sum(r[, k]), 1e-12)
          sigma[k] <- max(sqrt(s2), min_sigma)
        }
      }

      # ----- M step: component residual KDEs -----
      for (k in seq_len(K)) {
        ehat <- (A - mu[, k]) / sigma[k]
        wk <- r[, k]
        kde_list[[k]] <- build_kde(ehat, wk, bw_method = bw_method, bw_adjust = bw_adjust, n = kde_n, trim = kde_trim)
      }

      # ----- gating -----
      gate_fit <- fit_gate(X, r, gate_type = gate, min_pi = min_pi)
      pi_hat <- predict_gate(gate_fit, X, K = K, min_pi = min_pi)

      # ----- E step: responsibilities -----
      ll <- 0
      r_new <- matrix(NA_real_, nrow = n, ncol = K)

      for (i in seq_len(n)) {
        lcomp <- numeric(K)
        for (k in seq_len(K)) {
          eik <- (A[i] - mu[i, k]) / sigma[k]
          gk <- pmax(kde_list[[k]]$dfun(eik), eps)  # density in residual space
          lcomp[k] <- log(pmax(pi_hat[i, k], min_pi)) + log(gk) - log(sigma[k])
        }
        lse <- logsumexp_vec(lcomp)
        ll <- ll + lse
        r_new[i, ] <- exp(lcomp - lse)
      }

      r <- r_new

      if (is.finite(ll_prev) && abs(ll - ll_prev) / (abs(ll_prev) + 1e-8) < tol) break
      ll_prev <- ll
    }

    list(
      terms = terms_obj,
      rhs = rhs,
      K = K,
      gate = gate,
      var = var,
      kde_bw_method = bw_method,
      kde_adjust = bw_adjust,
      beta_list = beta_list,
      sigma = sigma,
      gate_fit = gate_fit,
      kde_list = kde_list
    )
  }

  strip_fit_bundle <- function(fit) {
    # keep only what is needed for prediction + sampling
    list(
      terms = fit$terms,
      rhs = fit$rhs,
      K = fit$K,
      gate = fit$gate,
      var = fit$var,
      kde_bw_method = fit$kde_bw_method,
      kde_adjust = fit$kde_adjust,
      beta_list = fit$beta_list,
      sigma = fit$sigma,
      gate_fit = fit$gate_fit,
      kde_list = fit$kde_list
    )
  }

  list(
    method = "mix_locscale_kde",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      if (!is.null(seed)) set.seed(seed)

      A <- train_set[["A"]]
      if (is.null(A)) stop("train_set must contain column 'A'")
      W <- train_set
      W[["A"]] <- NULL

      fits <- vector("list", nrow(tune_grid))
      for (i in seq_len(nrow(tune_grid))) {
        row <- tune_grid[i, ]
        fi <- fit_one_em(
          A = A,
          W = W,
          rhs = row$rhs,
          K = as.integer(row$K),
          gate = row$gate,
          var = row$var,
          bw_method = row$kde_bw_method,
          bw_adjust = as.numeric(row$kde_adjust),
          init = row$init,
          seed = if (is.null(seed)) NULL else seed + i
        )
        fits[[i]] <- if (strip_fit) strip_fit_bundle(fi) else fi
      }

      list(fits = fits, tune_grid = tune_grid, tune = NULL)
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      A <- train_set[["A"]]
      if (is.null(A)) stop("train_set must contain column 'A'")
      W <- train_set
      W[["A"]] <- NULL

      row <- tune_grid[k, ]
      fk <- fit_one_em(
        A = A,
        W = W,
        rhs = row$rhs,
        K = as.integer(row$K),
        gate = row$gate,
        var = row$var,
        bw_method = row$kde_bw_method,
        bw_adjust = as.numeric(row$kde_adjust),
        init = row$init,
        seed = if (is.null(seed)) NULL else seed + k
      )
      fk <- if (strip_fit) strip_fit_bundle(fk) else fk

      list(
        fits = list(fk),
        tune_grid = tune_grid[k, , drop = FALSE],
        tune = k
      )
    },

    log_density = function(fit_bundle, newdata, ...) {
      fits <- fit_bundle$fits
      if (is.null(fits) || length(fits) < 1L) stop("fit_bundle$fits is empty")

      A <- newdata[["A"]]
      if (is.null(A)) stop("newdata must contain column 'A' for log_density()")

      W <- newdata
      W[["A"]] <- NULL

      n <- length(A)
      Ktune <- length(fits)
      out <- matrix(NA_real_, nrow = n, ncol = Ktune)

      for (j in seq_len(Ktune)) {
        fit <- fits[[j]]
        X <- build_X(W, fit$terms)
        Kmix <- fit$K

        # component means
        mu <- matrix(0, nrow = n, ncol = Kmix)
        for (k in seq_len(Kmix)) {
          mu[, k] <- as.vector(X %*% fit$beta_list[[k]])
        }

        # gating
        pi_hat <- predict_gate(fit$gate_fit, X, K = Kmix, min_pi = min_pi)

        # log f via log-sum-exp
        ll <- numeric(n)
        for (i in seq_len(n)) {
          lcomp <- numeric(Kmix)
          for (k in seq_len(Kmix)) {
            eik <- (A[i] - mu[i, k]) / fit$sigma[k]
            gk <- pmax(fit$kde_list[[k]]$dfun(eik), eps)
            lcomp[k] <- log(pmax(pi_hat[i, k], min_pi)) + log(gk) - log(fit$sigma[k])
          }
          ll[i] <- logsumexp_vec(lcomp)
        }
        out[, j] <- ll
      }

      out
    },

    sample = function(fit_bundle, newdata, n_samp = 1L, seed = NULL, ...) {
      fits <- fit_bundle$fits
      if (length(fits) != 1L) stop("sample() assumes post-selection (fit_bundle$fits length 1)")
      fit <- fits[[1]]

      if (!is.null(seed)) set.seed(seed)

      W <- newdata
      if (!is.null(W[["A"]])) W[["A"]] <- NULL

      X <- build_X(W, fit$terms)
      n <- nrow(X)
      Kmix <- fit$K

      # means
      mu <- matrix(0, nrow = n, ncol = Kmix)
      for (k in seq_len(Kmix)) {
        mu[, k] <- as.vector(X %*% fit$beta_list[[k]])
      }

      # gating probs
      pi_hat <- predict_gate(fit$gate_fit, X, K = Kmix, min_pi = min_pi)

      n_samp <- as.integer(n_samp)
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer")

      samp <- matrix(NA_real_, nrow = n, ncol = n_samp)

      for (s in seq_len(n_samp)) {
        z <- integer(n)
        if (Kmix == 1L) {
          z[] <- 1L
        } else {
          for (i in seq_len(n)) {
            z[i] <- sample.int(Kmix, size = 1L, prob = pi_hat[i, ])
          }
        }

        for (i in seq_len(n)) {
          k <- z[i]
          u <- stats::runif(1)
          e <- fit$kde_list[[k]]$qfun(u)
          samp[i, s] <- mu[i, k] + fit$sigma[k] * e
        }
      }

      samp
    }
  )
}

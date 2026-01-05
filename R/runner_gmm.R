#' Create a Gaussian mixture-of-experts runner for direct conditional density estimation
#'
#' Constructs a runner (learner adapter) compatible with the
#' dsldensify() / run_direct_setting() / summarize_and_select() workflow for
#' direct conditional density estimation of a continuous outcome \eqn{A} given
#' covariates \eqn{W}.
#'
#' This runner models the conditional density as a finite Gaussian mixture
#' with component-specific mean functions and either shared or component-specific
#' scales:
#' \deqn{f(a \mid W) = \sum_{k=1}^K \pi_k(W)\,\phi\!\left(a;\mu_k(W),\sigma_k\right),}
#' where \eqn{\phi(\cdot;\mu,\sigma)} denotes the Normal density with mean \eqn{\mu}
#' and standard deviation \eqn{\sigma}, and mixture weights \eqn{\pi_k(W)} satisfy
#' \eqn{\pi_k(W) \ge 0} and \eqn{\sum_{k=1}^K \pi_k(W) = 1}.
#'
#' Component means
#'
#' Each component mean is modeled as a linear regression on the design matrix
#' induced by \code{rhs_list}:
#' \deqn{\mu_k(W) = X(W)\,\beta_k,}
#' where \eqn{X(W)} is the model matrix produced by \code{stats::model.matrix()}
#' from the RHS specification and \eqn{\beta_k} is a component-specific coefficient
#' vector estimated by weighted least squares in the M-step of EM.
#'
#' Mixture weights (gating)
#'
#' The mixture weights \eqn{\pi_k(W)} are controlled by \code{gate_grid}:
#' \itemize{
#' \item \code{"const"}: constant mixing proportions, \eqn{\pi_k(W) = \pi_k}.
#' \item \code{"glm"}: multinomial logistic gating on the design matrix \eqn{X(W)}
#'   with a baseline-category parameterization and softmax normalization:
#'   \deqn{\pi_k(W) = \frac{\exp\{\eta_k(W)\}}{\sum_{\ell=1}^K \exp\{\eta_\ell(W)\}},}
#'   where \eqn{\eta_k(W) = X(W)^\top b_k} for \eqn{k=1,\dots,K-1} and
#'   \eqn{\eta_K(W) = 0}.
#' }
#'
#' Scale structure
#'
#' The component standard deviations are controlled by \code{var_grid}:
#' \itemize{
#' \item \code{"shared"}: a common \eqn{\sigma} is used for all components
#'   (\eqn{\sigma_k \equiv \sigma}).
#' \item \code{"by_component"}: component-specific \eqn{\sigma_k} are used.
#' }
#' In both cases, scales are floored by \code{min_sigma} for numerical stability.
#'
#' Estimation by EM
#'
#' For each tuning configuration, parameters are estimated by an
#' expectation-maximization (EM) algorithm on the observed-data log-likelihood.
#' Let \eqn{r_{ik}} denote the responsibility of component \eqn{k} for observation
#' \eqn{i}. The EM iterations proceed as follows:
#'
#' E-step:
#' \deqn{r_{ik} \propto \pi_k(W_i)\,\phi\!\left(A_i;\mu_k(W_i),\sigma_k\right),}
#' with normalization so that \eqn{\sum_{k=1}^K r_{ik} = 1}.
#'
#' M-step:
#' \enumerate{
#' \item Update regression coefficients \eqn{\beta_k} by weighted least squares of
#'   \eqn{A} on \eqn{X(W)} using weights \eqn{r_{ik}}.
#' \item Update \eqn{\sigma} or \eqn{\sigma_k} from weighted residual sums of squares,
#'   depending on \code{var_grid}.
#' \item Update mixture weights \eqn{\pi_k(W)} using either constant proportions
#'   (weighted averages of responsibilities) or multinomial logistic gating on
#'   \eqn{X(W)} (Newton/IRLS updates on the multinomial log-likelihood).
#' }
#'
#' Convergence is assessed using the relative change in the observed-data log-likelihood,
#' stopping when the criterion falls below \code{tol} or after \code{max_iter} iterations.
#'
#' Model selection via log_density()
#'
#' Model selection uses likelihood-based scoring via log_density(): for each
#' tuning row and each observation, log_density() evaluates
#' \deqn{\log \hat f(A_i \mid W_i),}
#' where \eqn{\hat f} is the fitted Gaussian mixture density.
#' During cross-validation, \code{log_density()} returns an \eqn{n \times K}
#' matrix of log-densities aligned to \code{tune_grid$.tune}, where
#' \eqn{K = nrow(tune_grid)}.
#'
#' Sampling from the fitted model
#'
#' The runner provides a \code{sample()} method that generates draws
#' \deqn{A^\ast \sim \hat f(\cdot \mid W)}
#' by first sampling a component label \eqn{Z \in \{1,\dots,K\}} from
#' \eqn{\hat\pi(W)} and then sampling from the corresponding Normal distribution:
#' \deqn{A^\ast = \hat\mu_Z(W) + \hat\sigma_Z\,\xi,}
#' where \eqn{\xi \sim N(0,1)}. Sampling assumes the fit bundle contains exactly
#' one tuned fit (length(fit_bundle$fits) == 1), which is the intended
#' post-selection usage.
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
#' \item initialization strategy \code{init_grid}.
#' }
#'
#' Stabilization and numerical safety
#'
#' Mixture weights are bounded below by \code{min_pi} and renormalized to sum to one.
#' Scale parameters are bounded below by \code{min_sigma}. These stabilizations are
#' applied both during EM updates and during prediction. The internal softmax
#' computations for gating and the component mixture evaluations use log-sum-exp
#' stabilization to reduce underflow.
#'
#' @param rhs_list A list of RHS specifications, either as one-sided formulas
#'   (for example, \code{~ x1 + x2}) or as character strings
#'   (for example, \code{"x1 + x2"}). These RHS are used to build the design
#'   matrix for the component mean regressions and, when \code{gate_grid} includes
#'   \code{"glm"}, the gating model for \eqn{\pi_k(W)}.
#'
#' @param K_grid Integer vector of candidate mixture sizes \eqn{K}.
#'
#' @param gate_grid Character vector specifying gating model choices for
#'   \eqn{\pi_k(W)}. Supported values are \code{"const"} and \code{"glm"}.
#'
#' @param var_grid Character vector specifying the scale structure.
#'   Supported values are \code{"by_component"} (component-specific \eqn{\sigma_k})
#'   and \code{"shared"} (a common \eqn{\sigma} for all components).
#'
#' @param init_grid Character vector specifying initialization strategies for
#'   responsibilities. Supported values are \code{"kmeansA"} (k-means clustering
#'   on \eqn{A}) and \code{"random"} (random component assignment).
#'
#' @param max_iter Integer maximum number of EM iterations for each tuning row.
#'
#' @param tol Nonnegative convergence tolerance for relative change in the
#'   observed-data log-likelihood.
#'
#' @param min_sigma Small positive constant used as a floor for the estimated
#'   scale parameter(s) \eqn{\sigma} or \eqn{\sigma_k}.
#'
#' @param min_pi Small positive constant used as a floor for mixture weights
#'   \eqn{\pi_k(W)} prior to renormalization.
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
#'   method: Character string \code{"gmm"}.
#'   tune_grid: Data frame describing the tuning grid, including \code{.tune}.
#'   fit: Function \code{fit(train_set, ...)} returning a fit bundle.
#'   fit_one: Function \code{fit_one(train_set, tune, ...)} fitting only the
#'     selected tuning index.
#'   log_density: Function \code{log_density(fit_bundle, newdata, ...)} returning
#'     an \eqn{n \times K} matrix of log-densities.
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
#' runner <- make_gmm_runner(
#'   rhs_list = list(~ x1 + x2),
#'   K_grid = c(1L, 2L),
#'   gate_grid = c("const", "glm"),
#'   var_grid = c("shared"),
#'   init_grid = c("kmeansA"),
#'   max_iter = 150L,
#'   tol = 1e-6,
#'   min_sigma = 1e-4,
#'   strip_fit = TRUE,
#'   seed = 123
#' )
#'
#' @export

make_gmm_runner <- function(
  rhs_list,
  K_grid = c(1L, 2L, 3L),
  gate_grid = c("const", "glm"),     # mixture weights: constant or softmax on RHS
  var_grid  = c("by_component", "shared"),  # sigma_k or common sigma
  init_grid = c("kmeansA", "random"), # init responsibilities
  max_iter = 200L,
  tol = 1e-6,
  min_sigma = 1e-6,
  min_pi = 1e-10,
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
    rhs  = rhs_chr,
    K    = K_grid,
    gate = gate_grid,
    var  = var_grid,
    init = init_grid,
    stringsAsFactors = FALSE
  )
  tune_grid$.tune <- seq_len(nrow(tune_grid))
  tune_grid <- tune_grid[, c(".tune", "rhs", "K", "gate", "var", "init")]

  # ---- helpers ----
  logsumexp_vec <- function(x) {
    # x is numeric vector
    m <- max(x)
    m + log(sum(exp(x - m)))
  }

  softmax_rows <- function(eta_mat) {
    # eta_mat: n x K
    # returns n x K probs
    row_max <- apply(eta_mat, 1, max)
    z <- exp(eta_mat - row_max)
    z / rowSums(z)
  }

  build_terms <- function(rhs) stats::as.formula(paste0("~", rhs))

  build_X <- function(W, terms_obj) {
    # W can be data.frame or data.table; model.matrix will coerce
    stats::model.matrix(terms_obj, data = W)
  }

  # Fit weighted least squares for each component: y ~ X
  fit_wls <- function(X, y, w) {
    # returns beta and fitted mu
    # note: stats::lm.wfit expects X as matrix with intercept if desired.
    out <- stats::lm.wfit(x = X, y = y, w = w)
    list(beta = out$coefficients, mu = as.vector(X %*% out$coefficients))
  }

  # Gating: constant pi OR multinomial logit on X
  # We store coefs for K-1 classes with baseline class K.
  fit_gate <- function(X, r, gate_type, min_pi) {
    n <- nrow(X); K <- ncol(r)
    if (gate_type == "const" || K == 1L) {
      pi <- colMeans(r)
      pi <- pmax(pi, min_pi); pi <- pi / sum(pi)
      return(list(type = "const", pi = pi, coef = NULL))
    }

    # multinomial logit with baseline K: eta_k = X %*% B_k for k=1..K-1, eta_K = 0
    # We'll do IRLS on the multinomial log-likelihood.
    p <- ncol(X)
    B <- matrix(0, nrow = p, ncol = K - 1L)  # initialize
    for (iter in seq_len(50L)) {
      eta <- cbind(X %*% B, 0) # n x K
      pi_hat <- softmax_rows(eta)
      # gradient and Hessian block structure:
      # For k in 1..K-1: grad_k = X^T (r_k - pi_k)
      # Hessian: sum_i pi_k*(delta_kl - pi_l) x_i x_i^T
      G <- matrix(0, nrow = p, ncol = K - 1L)

      # Build Hessian as (p*(K-1)) x (p*(K-1)) block matrix
      H <- matrix(0, nrow = p * (K - 1L), ncol = p * (K - 1L))

      for (k in seq_len(K - 1L)) {
        G[, k] <- crossprod(X, r[, k] - pi_hat[, k])
      }

      # Hessian blocks
      for (k in seq_len(K - 1L)) {
        for (l in seq_len(K - 1L)) {
          w_kl <- if (k == l) pi_hat[, k] * (1 - pi_hat[, k]) else -pi_hat[, k] * pi_hat[, l]
          # crossprod(X * w, X) = X^T diag(w) X
          # do via weighted crossprod
          Xw <- X * as.vector(w_kl)
          H_block <- crossprod(X, Xw)
          # place
          rr <- ((k - 1L) * p + 1L):((k) * p)
          cc <- ((l - 1L) * p + 1L):((l) * p)
          H[rr, cc] <- H_block
        }
      }

      # Newton step with small ridge for stability
      ridge <- 1e-8
      diag(H) <- diag(H) + ridge
      step <- tryCatch(solve(H, as.vector(G)), error = function(e) NULL)
      if (is.null(step)) break
      B_new <- B + matrix(step, nrow = p, ncol = K - 1L)
      if (max(abs(B_new - B)) < 1e-6) { B <- B_new; break }
      B <- B_new
    }

    eta <- cbind(X %*% B, 0)
    pi_hat <- softmax_rows(eta)
    # floors
    pi_hat <- pmax(pi_hat, min_pi)
    pi_hat <- pi_hat / rowSums(pi_hat)
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

  # init responsibilities
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

  # core EM for one tuning row
  fit_gmm_one <- function(A, W, rhs, K, gate, var, init, seed, max_iter, tol, min_sigma, min_pi) {
    terms_obj <- build_terms(rhs)
    X <- build_X(W, terms_obj)
    n <- length(A)

    r <- init_resp(A, K, init, seed)

    # initialize means/vars quickly
    mu <- matrix(0, nrow = n, ncol = K)
    beta_list <- vector("list", K)
    sigma <- rep(stats::sd(A), K)
    if (!is.finite(sigma[1]) || sigma[1] <= 0) sigma <- rep(1, K)
    gate_fit <- list(type = "const", pi = rep(1 / K, K), coef = NULL)

    ll_prev <- -Inf

    for (iter in seq_len(max_iter)) {
      # M-step: component means via WLS
      for (k in seq_len(K)) {
        fk <- fit_wls(X, A, w = r[, k])
        beta_list[[k]] <- fk$beta
        mu[, k] <- fk$mu
      }

      # variances
      if (K == 1L) {
        sigma <- max(stats::sd(A - mu[, 1]), min_sigma)
      } else if (var == "shared") {
        # one sigma for all components
        num <- 0
        den <- 0
        for (k in seq_len(K)) {
          res2 <- (A - mu[, k])^2
          num <- num + sum(r[, k] * res2)
          den <- den + sum(r[, k])
        }
        s2 <- num / max(den, 1e-12)
        sigma <- rep(max(sqrt(s2), min_sigma), K)
      } else { # by_component
        for (k in seq_len(K)) {
          res2 <- (A - mu[, k])^2
          s2 <- sum(r[, k] * res2) / max(sum(r[, k]), 1e-12)
          sigma[k] <- max(sqrt(s2), min_sigma)
        }
      }

      # gating
      gate_fit <- fit_gate(X, r, gate_type = gate, min_pi = min_pi)
      pi_hat <- predict_gate(gate_fit, X, K = K, min_pi = min_pi)

      # E-step: update responsibilities
      loglik_mat <- matrix(NA_real_, nrow = n, ncol = K)
      for (k in seq_len(K)) {
        loglik_mat[, k] <- stats::dnorm(A, mean = mu[, k], sd = sigma[k], log = TRUE) + log(pi_hat[, k])
      }

      # normalize in log space
      r_new <- matrix(NA_real_, nrow = n, ncol = K)
      ll <- 0
      for (i in seq_len(n)) {
        lse <- logsumexp_vec(loglik_mat[i, ])
        ll <- ll + lse
        r_new[i, ] <- exp(loglik_mat[i, ] - lse)
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
      beta_list = beta_list,
      sigma = sigma,
      gate_fit = gate_fit
    )
  }

  # pack stripped fit
  strip_fit_bundle <- function(fit) {
    # store only what we need for prediction
    list(
      terms = fit$terms,
      rhs = fit$rhs,
      K = fit$K,
      gate = fit$gate,
      var = fit$var,
      beta_list = fit$beta_list,
      sigma = fit$sigma,
      gate_fit = fit$gate_fit
    )
  }

  # ---- runner ----
  list(
    method = "gmm",
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
        fit_i <- fit_gmm_one(
          A = A, W = W,
          rhs = row$rhs,
          K = as.integer(row$K),
          gate = row$gate,
          var = row$var,
          init = row$init,
          seed = if (is.null(seed)) NULL else seed + i,
          max_iter = max_iter,
          tol = tol,
          min_sigma = min_sigma,
          min_pi = min_pi
        )
        fits[[i]] <- if (strip_fit) strip_fit_bundle(fit_i) else fit_i
      }

      list(
        fits = fits,
        tune_grid = tune_grid,
        tune = NULL
      )
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
      fit_k <- fit_gmm_one(
        A = A, W = W,
        rhs = row$rhs,
        K = as.integer(row$K),
        gate = row$gate,
        var = row$var,
        init = row$init,
        seed = if (is.null(seed)) NULL else seed + k,
        max_iter = max_iter,
        tol = tol,
        min_sigma = min_sigma,
        min_pi = min_pi
      )
      fit_k <- if (strip_fit) strip_fit_bundle(fit_k) else fit_k

      list(
        fits = list(fit_k),
        tune_grid = tune_grid[k, , drop = FALSE],
        tune = k
      )
    },

    log_density = function(fit_bundle, newdata, eps = 1e-15, ...) {
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

        # gating probs
        pi_hat <- predict_gate(fit$gate_fit, X, K = Kmix, min_pi = min_pi)

        # compute log f via log-sum-exp per row
        ll_vec <- numeric(n)
        for (i in seq_len(n)) {
          lcomp <- numeric(Kmix)
          for (k in seq_len(Kmix)) {
            lcomp[k] <- stats::dnorm(A[i], mean = mu[i, k], sd = fit$sigma[k], log = TRUE) +
              log(pmax(pi_hat[i, k], min_pi))
          }
          ll_vec[i] <- logsumexp_vec(lcomp)
        }
        out[, j] <- ll_vec
      }

      out
    },

    sample = function(fit_bundle, newdata, n_samp = 1L, ...) {
      fits <- fit_bundle$fits
      if (length(fits) != 1L) stop("sample() assumes post-selection K=1 (fit_bundle$fits length 1)")

      fit <- fits[[1]]
      W <- newdata
      if (!is.null(W[["A"]])) W[["A"]] <- NULL

      X <- build_X(W, fit$terms)
      n <- nrow(X)
      Kmix <- fit$K

      # component means
      mu <- matrix(0, nrow = n, ncol = Kmix)
      for (k in seq_len(Kmix)) {
        mu[, k] <- as.vector(X %*% fit$beta_list[[k]])
      }

      # gating probs
      pi_hat <- predict_gate(fit$gate_fit, X, K = Kmix, min_pi = min_pi)

      n_samp <- as.integer(n_samp)
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer")

      # return n x n_samp matrix
      samp <- matrix(NA_real_, nrow = n, ncol = n_samp)

      for (s in seq_len(n_samp)) {
        # sample component per row then sample normal
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
          samp[i, s] <- stats::rnorm(1L, mean = mu[i, k], sd = fit$sigma[k])
        }
      }

      samp
    }
  )
}

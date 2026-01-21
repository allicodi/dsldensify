#' Create a location-scale residual KDE runner for direct conditional density estimation
#'
#' Constructs a runner (learner adapter) compatible with the
#' dsldensify() / run_direct_setting() / summarize_and_select() workflow for
#' direct conditional density estimation of a continuous outcome \eqn{A} given
#' covariates \eqn{W}.
#'
#' The runner represents the conditional distribution using a location-scale
#' decomposition
#' \deqn{A = \mu(W) + \sigma(W)\,\varepsilon,}
#' where \eqn{\mu(W) = E(A \mid W)} is a conditional mean function,
#' \eqn{\sigma(W) > 0} is a conditional scale function, and \eqn{\varepsilon}
#' is a standardized residual with density \eqn{g}.
#'
#' Estimation proceeds in three stages for each tuning configuration:
#' \enumerate{
#' \item Fit a mean model \eqn{\hat\mu(W)}.
#' \item Fit a scale model for \eqn{\log\{\sigma^2(W)\}} using the pseudo-outcome
#'   \eqn{\log\{(A - \hat\mu(W))^2 + c\}}, where \eqn{c > 0} is a stabilizing constant.
#' \item Form standardized residuals \eqn{\hat\varepsilon = (A - \hat\mu(W))/\hat\sigma(W)}
#'   and estimate \eqn{g} by a univariate kernel density estimate on the residual scale.
#' }
#'
#' The implied conditional density estimator is
#' \deqn{\hat f(a \mid W) = \hat g\!\left(\frac{a - \hat\mu(W)}{\hat\sigma(W)}\right)\,\frac{1}{\hat\sigma(W)}.}
#'
#' Model selection uses likelihood-based scoring via log_density(): for each
#' tuning row and each observation, log_density() evaluates \eqn{\log \hat f(A_i \mid W_i)}.
#'
#' Mean and scale learners
#'
#' The runner supports multiple algorithms for the mean and scale models:
#' \itemize{
#' \item \code{"glm"}: least squares via \code{stats::lm.fit()}.
#' \item \code{"glmnet"}: penalized least squares via \code{glmnet::cv.glmnet()}.
#' \item \code{"ranger"}: random forest regression via \code{ranger::ranger()}.
#' \item \code{"xgboost"}: boosted trees via \code{xgboost::xgb.train()}.
#' }
#'
#' For \code{"ranger"} and \code{"xgboost"}, a coarse complexity label
#' (\code{"low"} or \code{"high"}) may be included as part of tuning to provide
#' small, fixed grids without exposing many low-level hyperparameters.
#' For \code{"glmnet"}, tuning uses \code{alpha} and a choice of \code{lambda}
#' (\code{"min"} or \code{"1se"}).
#'
#' Kernel density estimation on residuals
#'
#' The residual density \eqn{g} is estimated using \code{stats::density()} on the
#' training residuals within each tuning configuration. The KDE bandwidth is
#' controlled by \code{kde_bw_method_grid} and \code{kde_adjust_grid}. The KDE is
#' stored as a linear interpolant \eqn{\hat g} via \code{approxfun()}.
#'
#' Sampling from the fitted model
#'
#' The runner provides a \code{sample()} method that generates draws
#' \deqn{A^\ast \sim \hat f(\cdot \mid W)}
#' via
#' \deqn{A^\ast = \hat\mu(W) + \hat\sigma(W)\,\varepsilon^\ast,}
#' where \eqn{\varepsilon^\ast} is obtained by resampling training residuals
#' (optionally weighted) and applying Gaussian jitter with standard deviation
#' equal to the effective KDE bandwidth. Sampling assumes the fit bundle contains
#' exactly one tuned fit (length(fit_bundle$fits) == 1), which is the intended
#' post-selection usage.
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All variables
#' referenced in \code{rhs_list} must be numeric. Factor handling is not
#' supported and variables are not coerced internally.
#'
#' Tuning grid and prediction layout
#'
#' The tuning grid is the Cartesian product of:
#' \itemize{
#' \item RHS specifications from \code{rhs_list},
#' \item mean model specification (method and method-specific tuning),
#' \item scale model specification (method and method-specific tuning),
#' \item KDE bandwidth method and adjustment.
#' }
#'
#' During cross-validation, \code{log_density()} returns an \eqn{n \times K}
#' matrix of log-densities aligned to \code{tune_grid$.tune}, where
#' \eqn{K = nrow(tune_grid)}.
#'
#' Lightweight fit objects
#'
#' When \code{strip_fit = TRUE} (default), \code{"glm"} and \code{"glmnet"} fits
#' are reduced to coefficient-based representations sufficient for prediction:
#' \itemize{
#' \item \code{"glm"}: coefficient vector aligned to the training design columns.
#' \item \code{"glmnet"}: coefficient vector at the selected \code{lambda}
#'   (either \code{"min"} or \code{"1se"}).
#' }
#'
#' When \code{strip_fit = FALSE}, full fitted objects are stored. For
#' \code{"ranger"} and \code{"xgboost"}, full fitted objects are always stored.
#'
#' Stabilization and numerical safety
#'
#' The scale model is fit to \eqn{\log\{(A-\hat\mu(W))^2 + c\}} with
#' \code{resid_c = c}. Predicted scales are floored by
#' \eqn{\max\{\sigma_{\min}, \hat\sigma(W)\}} where
#' \eqn{\sigma_{\min} = \max\{\mathrm{sigma\_floor\_frac} \cdot \mathrm{sd}(A), \mathrm{eps}\}}.
#' Density values are bounded below by \code{eps} before taking logs.
#'
#' @param rhs_list A list of RHS specifications, either as one-sided formulas
#'   (for example, \code{~ x1 + x2}) or as character strings
#'   (for example, \code{"x1 + x2"}). These RHS are used to build the design
#'   matrix for both the mean and scale models.
#'
#' @param mean_methods Character vector specifying candidate algorithms for
#'   the mean model \eqn{\mu(W)}. Supported values are \code{"glm"},
#'   \code{"glmnet"}, \code{"ranger"}, and \code{"xgboost"}.
#'
#' @param mean_levels Character vector of complexity labels used when
#'   \code{mean_methods} includes \code{"ranger"} or \code{"xgboost"}.
#'   Typically \code{c("low","high")}. Ignored for \code{"glm"} and \code{"glmnet"}.
#'
#' @param scale_methods Character vector specifying candidate algorithms for
#'   the scale model for \eqn{\log\{\sigma^2(W)\}}. Supported values are
#'   \code{"glm"}, \code{"glmnet"}, \code{"ranger"}, and \code{"xgboost"}.
#'
#' @param scale_levels Character vector of complexity labels used when
#'   \code{scale_methods} includes \code{"ranger"} or \code{"xgboost"}.
#'
#' @param glmnet_alpha_grid Numeric vector of \code{alpha} values used when
#'   \code{"glmnet"} is included in \code{mean_methods} and/or \code{scale_methods}.
#'   \code{alpha = 0} corresponds to ridge regression and \code{alpha = 1} to lasso.
#'
#' @param glmnet_lambda_choice Character vector of choices for the \code{lambda}
#'   used for prediction in \code{cv.glmnet()}. Supported values are \code{"min"}
#'   and \code{"1se"}.
#'
#' @param xgb_nrounds_low,xgb_nrounds_high Integer numbers of boosting iterations
#'   used for \code{"xgboost"} at complexity levels \code{"low"} and \code{"high"}.
#'
#' @param xgb_max_depth_low,xgb_max_depth_high Integer tree depths used for
#'   \code{"xgboost"} at complexity levels \code{"low"} and \code{"high"}.
#'
#' @param xgb_eta,xgb_subsample,xgb_colsample_bytree Tuning parameters forwarded
#'   to \code{xgboost::xgb.train()} for \code{"xgboost"} fits.
#'
#' @param ranger_num_trees Integer number of trees for \code{ranger::ranger()}.
#'
#' @param ranger_min_node_size_low,ranger_min_node_size_high Integer minimum node
#'   sizes used for \code{"ranger"} at complexity levels \code{"low"} and \code{"high"}.
#'
#' @param ranger_mtry Optional integer specifying \code{mtry} for \code{"ranger"}.
#'   If NULL, a default \eqn{\lfloor \sqrt{p} \rfloor} heuristic is used, where
#'   \eqn{p} is the number of non-intercept predictors in the design matrix.
#'
#' @param kde_bw_method_grid Character vector of KDE bandwidth rules passed to
#'   \code{stats::density(bw = ...)}. Supported values are \code{"nrd0"} and \code{"SJ"}.
#'
#' @param kde_adjust_grid Numeric vector of multiplicative bandwidth adjustments
#'   passed to \code{stats::density(adjust = ...)}.
#'
#' @param resid_c Positive constant \eqn{c} used in the scale pseudo-outcome
#'   \eqn{\log\{(A-\hat\mu(W))^2 + c\}}.
#'
#' @param sigma_floor_frac Nonnegative scalar controlling the floor applied to
#'   predicted scales, as a fraction of \code{sd(A)}.
#'
#' @param eps Small positive constant used to bound densities away from zero and
#'   log-densities away from \eqn{-\infty}.
#'
#' @param use_weights_col Logical. If TRUE and \code{train_set} contains a column
#'   named \code{weights_col}, it is passed as case weights to supported learners
#'   (glm, glmnet, ranger, xgboost) and to KDE when supported by
#'   \code{stats::density()}.
#'
#' @param weights_col Character name of the weight column in \code{train_set}.
#'
#' @param standardize_glmnet Logical. Passed to \code{glmnet::cv.glmnet()}.
#'
#' @param strip_fit Logical. If TRUE (default), store lightweight
#'   coefficient-based representations for \code{"glm"} and \code{"glmnet"} fits.
#'   If FALSE, store full fitted objects for these methods. Full objects are
#'   always stored for \code{"ranger"} and \code{"xgboost"}.
#'
#' @param seed Optional integer seed. If provided, each tuning row is fit with
#'   \code{set.seed(seed + .tune)} for reproducibility.
#'
#' @param ... Additional arguments forwarded to the underlying fitting routines:
#'   \code{stats::lm.fit()}, \code{glmnet::cv.glmnet()}, \code{ranger::ranger()},
#'   and \code{xgboost::xgb.train()}.
#'
#' @return A named list (runner) with elements:
#'   method: Character string \code{"locscale_kde"}.
#'   tune_grid: Data frame describing the tuning grid, including \code{.tune}.
#'   fit: Function \code{fit(train_set, ...)} returning a fit bundle.
#'   log_density: Function \code{log_density(fit_bundle, newdata, ...)} returning
#'     an \eqn{n \times K} matrix of log-densities.
#'   density: Function \code{density(fit_bundle, newdata, ...)} returning densities.
#'   fit_one: Function \code{fit_one(train_set, tune, ...)} fitting only the
#'     selected tuning index.
#'   select_fit: Function \code{select_fit(fit_bundle, tune)} extracting a single
#'     tuning configuration.
#'   sample: Function \code{sample(fit_bundle, newdata, n_samp, ...)} drawing
#'     samples (assumes \eqn{K = 1}).
#'
#' Data requirements
#'
#' The runner expects \code{train_set} and \code{newdata} in wide format containing:
#' \itemize{
#' \item a numeric outcome column \code{A} (required for \code{fit()} and \code{log_density()}),
#' \item covariates referenced in \code{rhs_list},
#' \item an optional weight column named by \code{weights_col}.
#' }
#'
#' \code{sample()} expects \code{newdata} to contain only covariates \code{W}
#' (it must not require an \code{A} column).
#'
#' @examples
#' runner <- make_locscale_kde_runner(
#'   rhs_list = list(~ x1 + x2),
#'   mean_methods = c("glm", "glmnet", "ranger", "xgboost"),
#'   mean_levels = c("low"),
#'   scale_methods = c("glm", "glmnet"),
#'   scale_levels = c("low"),
#'   glmnet_alpha_grid = c(0, 1),
#'   glmnet_lambda_choice = c("1se"),
#'   kde_bw_method_grid = c("nrd0"),
#'   kde_adjust_grid = c(1.0),
#'   strip_fit = TRUE,
#'   eps = 1e-10,
#'   seed = 123
#' )
#'
#' @export

make_locscale_kde_direct_runner <- function(
  rhs_list,

  # mean model candidates
  mean_methods = c("glm", "glmnet", "ranger", "xgboost"),
  mean_levels = c("low", "high"),   # used for ranger/xgboost; ignored for glm/glmnet

  # scale model candidates
  scale_methods = c("glm", "glmnet", "ranger", "xgboost"),
  scale_levels = c("low"),

  # glmnet knobs (small + stable)
  glmnet_alpha_grid = c(0, 1),      # ridge, lasso
  glmnet_lambda_choice = c("1se", "min"),

  # xgboost knobs
  xgb_nrounds_low = 300L,
  xgb_nrounds_high = 800L,
  xgb_max_depth_low = 2L,
  xgb_max_depth_high = 4L,
  xgb_eta = 0.05,
  xgb_subsample = 0.8,
  xgb_colsample_bytree = 0.8,

  # ranger knobs
  ranger_num_trees = 500L,
  ranger_min_node_size_low = 50L,
  ranger_min_node_size_high = 10L,
  ranger_mtry = NULL,

  # KDE knobs
  kde_bw_method_grid = c("nrd0"),
  kde_adjust_grid = c(0.8, 1.0, 1.2),

  # stabilization
  resid_c = 1e-8,
  sigma_floor_frac = 1e-3,
  eps = 1e-12,

  # misc
  use_weights_col = TRUE,
  weights_col = "wts",
  standardize_glmnet = TRUE,
  strip_fit = TRUE,
  seed = NULL,
  ...
) {
  stopifnot(requireNamespace("stats", quietly = TRUE))
  stopifnot(requireNamespace("glmnet", quietly = TRUE))
  stopifnot(requireNamespace("ranger", quietly = TRUE))
  stopifnot(requireNamespace("xgboost", quietly = TRUE))

  ## ---- rhs parsing -------------------------------------------------------
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1.")

  ok_methods <- c("glm", "glmnet", "ranger", "xgboost")
  mean_methods <- unique(as.character(mean_methods))
  scale_methods <- unique(as.character(scale_methods))
  if (!all(mean_methods %in% ok_methods)) stop("mean_methods must be subset of: ", paste(ok_methods, collapse = ", "))
  if (!all(scale_methods %in% ok_methods)) stop("scale_methods must be subset of: ", paste(ok_methods, collapse = ", "))

  mean_levels <- unique(as.character(mean_levels))
  scale_levels <- unique(as.character(scale_levels))

  glmnet_alpha_grid <- as.numeric(glmnet_alpha_grid)
  if (any(!is.finite(glmnet_alpha_grid)) || any(glmnet_alpha_grid < 0) || any(glmnet_alpha_grid > 1)) {
    stop("glmnet_alpha_grid must be in [0, 1].")
  }
  glmnet_lambda_choice <- unique(as.character(glmnet_lambda_choice))
  if (!all(glmnet_lambda_choice %in% c("1se", "min"))) stop("glmnet_lambda_choice must be subset of c('1se','min').")

  kde_bw_method_grid <- unique(as.character(kde_bw_method_grid))
  if (!all(kde_bw_method_grid %in% c("nrd0", "SJ"))) stop("kde_bw_method_grid must be subset of c('nrd0','SJ').")
  kde_adjust_grid <- as.numeric(kde_adjust_grid)
  if (any(!is.finite(kde_adjust_grid)) || any(kde_adjust_grid <= 0)) stop("kde_adjust_grid must be positive numbers.")

  ## ---- tune grid ---------------------------------------------------------
  mean_cfg <- data.frame(
    mean_method = character(0),
    mean_level = character(0),
    mean_alpha = numeric(0),
    mean_lambda_choice = character(0),
    stringsAsFactors = FALSE
  )
  for (m in mean_methods) {
    if (m == "glm") {
      mean_cfg <- rbind(mean_cfg,
        data.frame(mean_method="glm", mean_level=NA_character_,
                   mean_alpha=NA_real_, mean_lambda_choice=NA_character_,
                   stringsAsFactors = FALSE))
    } else if (m == "glmnet") {
      for (a in glmnet_alpha_grid) for (lc in glmnet_lambda_choice) {
        mean_cfg <- rbind(mean_cfg,
          data.frame(mean_method="glmnet", mean_level=NA_character_,
                     mean_alpha=a, mean_lambda_choice=lc,
                     stringsAsFactors = FALSE))
      }
    } else {
      for (lv in mean_levels) {
        mean_cfg <- rbind(mean_cfg,
          data.frame(mean_method=m, mean_level=lv,
                     mean_alpha=NA_real_, mean_lambda_choice=NA_character_,
                     stringsAsFactors = FALSE))
      }
    }
  }

  scale_cfg <- data.frame(
    scale_method = character(0),
    scale_level = character(0),
    scale_alpha = numeric(0),
    scale_lambda_choice = character(0),
    stringsAsFactors = FALSE
  )
  for (m in scale_methods) {
    if (m == "glm") {
      scale_cfg <- rbind(scale_cfg,
        data.frame(scale_method="glm", scale_level=NA_character_,
                   scale_alpha=NA_real_, scale_lambda_choice=NA_character_,
                   stringsAsFactors = FALSE))
    } else if (m == "glmnet") {
      for (a in glmnet_alpha_grid) for (lc in glmnet_lambda_choice) {
        scale_cfg <- rbind(scale_cfg,
          data.frame(scale_method="glmnet", scale_level=NA_character_,
                     scale_alpha=a, scale_lambda_choice=lc,
                     stringsAsFactors = FALSE))
      }
    } else {
      for (lv in scale_levels) {
        scale_cfg <- rbind(scale_cfg,
          data.frame(scale_method=m, scale_level=lv,
                     scale_alpha=NA_real_, scale_lambda_choice=NA_character_,
                     stringsAsFactors = FALSE))
      }
    }
  }

  tune_grid <- merge(
    merge(data.frame(rhs = rhs_chr, stringsAsFactors = FALSE), mean_cfg, by = NULL),
    scale_cfg, by = NULL
  )
  tune_grid <- merge(
    tune_grid,
    expand.grid(
      kde_bw_method = kde_bw_method_grid,
      kde_adjust = kde_adjust_grid,
      stringsAsFactors = FALSE,
      KEEP.OUT.ATTRS = FALSE
    ),
    by = NULL
  )
  rownames(tune_grid) <- NULL
  tune_grid$.tune <- seq_len(nrow(tune_grid))

  ## ---- design helpers (predictor-only terms) -----------------------------
  build_design_train <- function(rhs_raw, train_df) {
    f <- stats::as.formula(paste0("A ~ ", rhs_raw))
    tt <- stats::terms(f, data = train_df)
    tt_x <- stats::delete.response(tt)
    X <- stats::model.matrix(tt_x, data = train_df)
    list(X = X, tt_x = tt_x, x_cols = colnames(X), rhs = rhs_raw)
  }

  build_design_new <- function(design_spec, new_df) {
    Xn <- stats::model.matrix(design_spec$tt_x, data = new_df)

    x_cols <- design_spec$x_cols
    miss <- setdiff(x_cols, colnames(Xn))
    if (length(miss)) {
      Xn <- cbind(Xn, matrix(0, nrow(Xn), length(miss), dimnames = list(NULL, miss)))
    }
    Xn[, x_cols, drop = FALSE]
  }

  ## ---- engine fit/predict ------------------------------------------------
  fit_engine_gaussian <- function(method, level, alpha, lambda_choice, X, y, w, ...) {

    # GLM
    if (method == "glm") {
      fit <- stats::lm.wfit(x = X, y = y, w = w)
      if (isTRUE(strip_fit)) {
        beta <- as.numeric(fit$coefficients)
        names(beta) <- colnames(X)
        beta[is.na(beta)] <- 0
        return(list(method = "glm_stripped", beta = beta))
      }
      return(list(method = "glm", fit = fit))
    }

    # GLMNET
    if (method == "glmnet") {
      cv <- glmnet::cv.glmnet(
        x = X, y = y,
        weights = w,
        alpha = alpha,
        family = "gaussian",
        standardize = isTRUE(standardize_glmnet),
        ...
      )
      lam <- if (lambda_choice == "min") cv$lambda.min else cv$lambda.1se
      if (isTRUE(strip_fit)) {
        b <- glmnet::coef.glmnet(cv$glmnet.fit, s = lam)
        b <- as.matrix(b)
        beta <- as.numeric(b[, 1L])
        names(beta) <- rownames(b)
        return(list(method = "glmnet_stripped", beta = beta))
      }
      return(list(method = "glmnet", cv = cv, lambda = lam))
    }

    # RANGER (keep full fit)
    if (method == "ranger") {
      p <- max(1L, ncol(X) - 1L)
      mtry <- if (is.null(ranger_mtry)) max(1L, floor(sqrt(p))) else as.integer(ranger_mtry)
      min_node <- if (identical(level, "high")) ranger_min_node_size_high else ranger_min_node_size_low

      df <- data.frame(y = y, X)
      fit <- ranger::ranger(
        y ~ ., data = df,
        num.trees = as.integer(ranger_num_trees),
        min.node.size = as.integer(min_node),
        mtry = as.integer(mtry),
        case.weights = w,
        respect.unordered.factors = "order",
        ...
      )
      return(list(method = "ranger", fit = fit))
    }

    # XGBOOST (keep full fit)
    if (method == "xgboost") {
      nrounds <- if (identical(level, "high")) as.integer(xgb_nrounds_high) else as.integer(xgb_nrounds_low)
      max_depth <- if (identical(level, "high")) as.integer(xgb_max_depth_high) else as.integer(xgb_max_depth_low)

      dtrain <- xgboost::xgb.DMatrix(data = X, label = y, weight = w)
      params <- list(
        objective = "reg:squarederror",
        eta = xgb_eta,
        max_depth = max_depth,
        subsample = xgb_subsample,
        colsample_bytree = xgb_colsample_bytree
      )
      fit <- xgboost::xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0, ...)
      return(list(method = "xgboost", fit = fit))
    }

    stop("Unknown method: ", method)
  }

  predict_engine <- function(obj, Xnew) {
    if (obj$method == "glm_stripped") {
      beta <- obj$beta
      beta <- beta[colnames(Xnew)]
      beta[is.na(beta)] <- 0
      return(drop(Xnew %*% beta))
    }
    if (obj$method == "glm") {
      return(drop(Xnew %*% obj$fit$coefficients))
    }
    if (obj$method == "glmnet_stripped") {
      # beta includes "(Intercept)" + feature names; Xnew includes intercept column named "(Intercept)"
      beta <- obj$beta
      b <- beta[colnames(Xnew)]
      b[is.na(b)] <- 0
      return(drop(Xnew %*% b))
    }
    if (obj$method == "glmnet") {
      return(as.numeric(predict(obj$cv$glmnet.fit, newx = Xnew, s = obj$lambda)))
    }
    if (obj$method == "ranger") {
      return(as.numeric(predict(obj$fit, data = data.frame(Xnew))$predictions))
    }
    if (obj$method == "xgboost") {
      return(as.numeric(predict(obj$fit, newdata = Xnew)))
    }
    stop("Unknown fitted object method: ", obj$method)
  }

  ## ---- KDE helpers -------------------------------------------------------
  fit_kde <- function(eps_train, w, bw_method, adjust) {
    args <- list(x = eps_train, bw = bw_method, adjust = adjust, n = 4096L)
    if (!is.null(w) && "weights" %in% names(formals(stats::density))) {
      args$weights <- w / sum(w)
    }
    d <- do.call(stats::density, args)
    f_eps <- stats::approxfun(d$x, pmax(d$y, eps), rule = 2L)
    list(
      f_eps = f_eps,
      bw_eff = as.numeric(d$bw) * as.numeric(adjust),
      eps_train = eps_train,
      w = w
    )
  }

  ## ---- core scoring (no Recall()) ----------------------------------------
  log_density_impl <- function(fit_bundle, newdata, eps = fit_bundle$eps, ...) {
    nd <- as.data.frame(newdata)
    if (!("A" %in% names(nd))) stop("newdata must contain column 'A'.")
    a <- as.numeric(nd$A)

    n <- nrow(nd)
    fits <- fit_bundle$fits
    K <- length(fits)
    out <- matrix(log(eps), nrow = n, ncol = K)

    for (k in seq_len(K)) {
      obj <- fits[[k]]
      Xn <- build_design_new(obj$design_spec, nd)

      mu <- predict_engine(obj$mean_obj, Xn)
      log_sig2 <- predict_engine(obj$scale_obj, Xn)
      sigma <- pmax(sqrt(exp(log_sig2)), obj$sigma_floor)

      e <- (a - mu) / sigma
      e[!is.finite(e)] <- 0

      dens <- obj$kde$f_eps(e) / sigma
      dens[!is.finite(dens)] <- eps
      out[, k] <- log(pmax(dens, eps))
    }

    out
  }

  ## ---- fit one tuning row ------------------------------------------------
  fit_one_row <- function(train_df, tg, ...) {
    if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(tg$.tune))

    y <- as.numeric(train_df$A)
    w <- if (use_weights_col && (weights_col %in% names(train_df))) as.numeric(train_df[[weights_col]]) else rep(1, length(y))
    sigma_floor <- max(sigma_floor_frac * stats::sd(y), eps)

    design <- build_design_train(tg$rhs, train_df)
    X <- design$X
    design_spec <- list(rhs = design$rhs, x_cols = design$x_cols, tt_x = design$tt_x)

    mean_obj <- fit_engine_gaussian(
      method = tg$mean_method,
      level = tg$mean_level,
      alpha = tg$mean_alpha,
      lambda_choice = tg$mean_lambda_choice,
      X = X, y = y, w = w,
      ...
    )
    mu <- predict_engine(mean_obj, X)

    r <- y - mu
    y_scale <- log(r^2 + resid_c)

    scale_obj <- fit_engine_gaussian(
      method = tg$scale_method,
      level = tg$scale_level,
      alpha = tg$scale_alpha,
      lambda_choice = tg$scale_lambda_choice,
      X = X, y = y_scale, w = w,
      ...
    )
    log_sig2 <- predict_engine(scale_obj, X)
    sigma <- pmax(sqrt(exp(log_sig2)), sigma_floor)

    eps_hat <- (y - mu) / sigma
    eps_hat[!is.finite(eps_hat)] <- 0

    kde <- fit_kde(eps_hat, w, tg$kde_bw_method, tg$kde_adjust)

    list(
      design_spec = design_spec,
      mean_obj = mean_obj,
      scale_obj = scale_obj,
      sigma_floor = sigma_floor,
      kde = kde,
      .tune = tg$.tune
    )
  }

  ## ---- runner ------------------------------------------------------------
  list(
    method = "locscale_kde",
    tune_grid = tune_grid,
    positive_support = FALSE,
    
    fit = function(train_set, ...) {
      train_df <- as.data.frame(train_set)
      if (!("A" %in% names(train_df))) stop("train_set must contain column 'A'.")
      if (!is.numeric(train_df$A)) stop("train_set$A must be numeric.")

      fits <- vector("list", nrow(tune_grid))
      for (k in seq_len(nrow(tune_grid))) {
        fits[[k]] <- fit_one_row(train_df, tune_grid[k, , drop = FALSE], ...)
      }
      list(fits = fits, tune_grid = tune_grid, eps = eps)
    },

    log_density = function(fit_bundle, newdata, eps = fit_bundle$eps, ...) {
      log_density_impl(fit_bundle, newdata, eps = eps, ...)
    },

    density = function(fit_bundle, newdata, eps = fit_bundle$eps, ...) {
      exp(log_density_impl(fit_bundle, newdata, eps = eps, ...))
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..nrow(tune_grid).")
      }
      train_df <- as.data.frame(train_set)
      if (!("A" %in% names(train_df))) stop("train_set must contain column 'A'.")
      if (!is.numeric(train_df$A)) stop("train_set$A must be numeric.")

      tg <- tune_grid[k, , drop = FALSE]
      fit_k <- fit_one_row(train_df, tg, ...)

      list(
        fits = list(fit_k),
        tune = k,
        tune_grid = tg,
        eps = eps
      )
    },

    select_fit = function(fit_bundle, tune) {
      k <- as.integer(tune)
      if (!is.null(fit_bundle$fits) && length(fit_bundle$fits) >= k) {
        fit_bundle$fits <- fit_bundle$fits[k]
      }
      fit_bundle
    },

    sample = function(fit_bundle, newdata, n_samp, seed = NULL, ...) {
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      if (length(fits) != 1L) stop("sample() assumes K=1: fit_bundle$fits must have length 1 (selected model).")
      if (!is.null(seed)) set.seed(seed)

      nd <- as.data.frame(newdata)
      obj <- fits[[1L]]
      n <- nrow(nd)
      if (n < 1L) stop("newdata must have at least one row.")

      Xn <- build_design_new(obj$design_spec, nd)

      mu <- predict_engine(obj$mean_obj, Xn)
      log_sig2 <- predict_engine(obj$scale_obj, Xn)
      sigma <- pmax(sqrt(exp(log_sig2)), obj$sigma_floor)

      eps_train <- obj$kde$eps_train
      w <- obj$kde$w
      prob <- if (is.null(w)) NULL else w / sum(w)

      idx <- sample.int(length(eps_train), size = n * n_samp, replace = TRUE, prob = prob)
      eps_star <- eps_train[idx] + stats::rnorm(length(idx), mean = 0, sd = obj$kde$bw_eff)

      eps_star <- matrix(eps_star, nrow = n, ncol = n_samp)
      mu_mat <- matrix(mu, nrow = n, ncol = n_samp)
      sigma_mat <- matrix(sigma, nrow = n, ncol = n_samp)

      mu_mat + sigma_mat * eps_star
    }
  )
}

#' Create a random forest runner for discrete-time hazard modeling
#'
#' Constructs a runner (learner adapter) compatible with the
#' run_grid_setting() / summarize_and_select() workflow used in dsl_densify.
#' The runner fits probabilistic random forest models via ranger::ranger()
#' on long-format discrete-time hazard data with binary outcome in_bin and
#' returns per-bin discrete-time hazard estimates as class probabilities
#' for in_bin = 1.
#'
#' Tuning is performed over a grid defined by:
#'   - multiple RHS feature specifications (rhs_list),
#'   - node-size parameters (min_node_size_grid),
#'   - optional feature-subset sizes (mtry_grid),
#'   - sampling behavior (bootstrap vs subsampling without replacement).
#'
#' The fitted models estimate per-bin discrete-time hazards
#'   P(T in bin_j | T >= bin_j, W)
#' using probability = TRUE, returning predicted probabilities for class "1".
#'
#' RHS specifications (column selection only)
#'
#' The rhs_list argument defines feature sets by extracting variable names:
#'   - RHS may be provided as one-sided formulas (for example, ~ W1 + W2),
#'     or as character strings (for example, "W1 + W2").
#'   - Each RHS is converted to a formula internally and variable names are
#'     extracted via all.vars().
#'   - No transformations, interactions, or spline terms are evaluated; the
#'     referenced columns must already exist in the long hazard data.
#'
#' By default, bin_id_col is enforced as part of each feature set. If it is
#' missing from an RHS specification, it is appended automatically.
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All columns
#' referenced by rhs_list (including bin_id) should already be numeric.
#' Factors, characters, and ordered factors are not supported and should be
#' encoded upstream.
#'
#' Tuning grid and prediction layout
#'
#' Each row of tune_grid corresponds to exactly one fitted forest.
#' The tune_grid is constructed so that RHS varies first, followed by
#' min_node_size, sampling behavior (sampling, replace, sample_fraction), and
#' optionally mtry (if mtry_grid is supplied). During cross-validation, predict()
#' returns an n_long x K matrix of predicted hazards, where K = nrow(tune_grid),
#' with columns aligned to .tune.
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
#' Sampling as a tuning dimension
#'
#' The runner supports a sampling scheme tuning dimension via sampling_grid:
#'   - "bootstrap" uses sampling with replacement (replace = TRUE) and
#'     sample.fraction = 1.0.
#'   - "subsample" uses sampling without replacement (replace = FALSE) with
#'     fractions provided in subsample_fraction_grid.
#'
#' Weights
#'
#' If use_weights_col = TRUE, observation weights are passed to ranger::ranger()
#' via case.weights using weights_col (default "wts"). If use_weights_col = TRUE
#' and weights_col is missing, fitting errors.
#'
#' Deterministic seeding
#'
#' If seed is provided, a tune-specific seed is used for each fit by calling
#' set.seed(seed + .tune). This yields deterministic fitting behavior across
#' runs while keeping different tuning rows distinct.
#'
#' @param rhs_list A list of RHS specifications, either one-sided formulas
#'   (for example, ~ W1 + W2 + bin_id) or character strings
#'   (for example, "W1 + W2 + bin_id"). These RHS are used for column selection
#'   only. If bin_id_col is missing, it is appended automatically.
#'
#' @param mtry_grid Optional integer vector of mtry values to tune over.
#'   If NULL, mtry is set per fit to floor(sqrt(p)) bounded to [1, p], where p
#'   is the number of selected features.
#'
#' @param min_node_size_grid Integer vector of minimum terminal node sizes
#'   passed to ranger::ranger(min.node.size = ...).
#'
#' @param num_trees Integer number of trees in each forest.
#'
#' @param sampling_grid Character vector specifying sampling schemes to include.
#'   Must be a subset of c("bootstrap", "subsample").
#'
#' @param subsample_fraction_grid Numeric vector of sampling fractions used
#'   only when sampling = "subsample".
#'
#' @param outcome_col Name of the binary outcome column in the long hazard data.
#'
#' @param bin_id_col Name of the time-bin column. This column is enforced in
#'   every feature set unless already included.
#'
#' @param use_weights_col Logical. If TRUE, pass weights_col as case.weights to
#'   ranger::ranger().
#'
#' @param weights_col Name of the weights column.
#'
#' @param respect_unordered_factors Passed to
#'   ranger::ranger(respect.unordered.factors = ...).
#'
#' @param importance Passed to ranger::ranger(importance = ...).
#'
#' @param seed Optional integer seed. If provided, a deterministic tune-specific
#'   seed seed + .tune is used during fitting.
#'
#' @param ... Additional arguments forwarded to ranger::ranger().
#'
#' @return A named list (runner) with the following elements:
#'   method: Character string "rf".
#'   tune_grid: Data frame describing the tuning grid, including .tune.
#'   fit: Function fit(train_set, ...) returning a fit bundle.
#'   predict: Function predict(fit_bundle, newdata, ...) returning an
#'     n_long x K matrix of hazard predictions.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected
#'     tuning index.
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing samples
#'     from the implied conditional density (assumes K = 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata as data.table objects in long hazard
#' format, including:
#'   - a binary outcome column outcome_col (default "in_bin"),
#'   - a time-bin column bin_id_col (default "bin_id"),
#'   - covariates referenced in rhs_list,
#'   - an optional weight column weights_col (default "wts").
#'
#' newdata passed to sample() must additionally include:
#'   - obs_id,
#'   - bin_lower,
#'   - bin_upper.
#'
#' @examples
#' rhs_list <- list(
#'   ~ bin_id + W1 + W2,
#'   ~ bin_id + W1 + W2 + W3
#' )
#'
#' runner <- make_rf_runner(
#'   rhs_list = rhs_list,
#'   min_node_size_grid = c(5L, 20L, 50L),
#'   num_trees = 500L,
#'   sampling_grid = c("bootstrap", "subsample"),
#'   subsample_fraction_grid = c(0.6, 0.8),
#'   mtry_grid = c(5L, 10L),
#'   seed = 123
#' )
#'
#' @export
make_rf_runner <- function(
  rhs_list,
  mtry_grid = NULL,
  min_node_size_grid = c(5L, 20L, 50L),
  num_trees = 500L,
  sampling_grid = c("bootstrap", "subsample"),
  subsample_fraction_grid = c(0.6, 0.8),
  outcome_col = "in_bin",
  bin_id_col = "bin_id",
  use_weights_col = TRUE,
  weights_col = "wts",
  respect_unordered_factors = "order",
  importance = "none",
  seed = NULL,
  ...
) {
  stopifnot(requireNamespace("ranger", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))

  # rhs parsing
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  add_bin_id_if_missing <- function(rhs) {
    vars <- all.vars(stats::as.formula(paste0("~", rhs)))
    if (!(bin_id_col %in% vars)) paste(rhs, "+", bin_id_col) else rhs
  }
  rhs_chr <- vapply(rhs_chr, add_bin_id_if_missing, character(1))

  sampling_grid <- unique(as.character(sampling_grid))
  if (!all(sampling_grid %in% c("bootstrap", "subsample"))) {
    stop("sampling_grid must be subset of c('bootstrap','subsample')")
  }

  sampling_tbl <- do.call(
    rbind,
    lapply(sampling_grid, function(s) {
      if (s == "bootstrap") {
        data.frame(
          sampling = "bootstrap",
          replace = TRUE,
          sample_fraction = 1.0,
          stringsAsFactors = FALSE
        )
      } else {
        data.frame(
          sampling = "subsample",
          replace = FALSE,
          sample_fraction = as.numeric(subsample_fraction_grid),
          stringsAsFactors = FALSE
        )
      }
    })
  )

  tune_grid <- merge(
    expand.grid(
      rhs = rhs_chr,
      min_node_size = as.integer(min_node_size_grid),
      stringsAsFactors = FALSE,
      KEEP.OUT.ATTRS = FALSE
    ),
    sampling_tbl,
    by = NULL
  )

  if (!is.null(mtry_grid)) {
    tune_grid <- merge(
      tune_grid,
      data.frame(mtry = as.integer(mtry_grid), stringsAsFactors = FALSE),
      by = NULL
    )
  } else {
    tune_grid$mtry <- NA_integer_
  }

  tune_grid$.tune <- seq_len(nrow(tune_grid))

  rhs_to_cols <- function(rhs) setdiff(all.vars(stats::as.formula(paste0("~", rhs))), outcome_col)
  default_mtry <- function(p) max(1L, min(p, as.integer(floor(sqrt(p)))))

  # ---- hazard conventions (match dsldensify + other hazard runners) -------
  id_col <- "obs_id"
  bin_var <- bin_id_col
  eps <- 1e-15
  clip01 <- function(p) pmin(pmax(p, eps), 1 - eps)

  # Predict hazards for all tuned fits in a fit bundle
  predict_hazards <- function(fits, newdata, ...) {
    if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table.")
    if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")

    n <- nrow(newdata)
    out <- matrix(NA_real_, nrow = n, ncol = length(fits))

    for (k in seq_along(fits)) {
      cols <- fits[[k]]$cols
      df_new <- as.data.frame(newdata[, cols, with = FALSE])

      pr <- ranger::predict(
        fits[[k]]$model,
        data = df_new,
        type = "response",
        ...
      )$predictions

      if (is.matrix(pr) || is.data.frame(pr)) {
        if (!("1" %in% colnames(pr))) stop("ranger probability predictions missing class '1'.")
        pr <- pr[, "1", drop = TRUE]
      }

      out[, k] <- clip01(as.numeric(pr))
    }

    out
  }

  list(
    method = "rf",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      if (!data.table::is.data.table(train_set)) stop("train_set must be a data.table")
      if (!(outcome_col %in% names(train_set))) stop("Missing outcome_col in train_set")
      if (use_weights_col && !(weights_col %in% names(train_set))) stop("Missing weights_col in train_set")

      fits <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        tg <- tune_grid[k, , drop = FALSE]
        cols <- rhs_to_cols(tg$rhs)

        mtry_k <- tg$mtry
        if (is.na(mtry_k)) mtry_k <- default_mtry(length(cols))

        df <- as.data.frame(train_set[, c(outcome_col, cols), with = FALSE])
        y <- df[[outcome_col]]
        if (is.logical(y)) y <- as.integer(y)
        if (is.numeric(y) || is.integer(y)) {
          df[[outcome_col]] <- factor(as.integer(y), levels = c(0L, 1L))
        }

        case_wts <- if (use_weights_col) train_set[[weights_col]] else NULL
        if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(tg$.tune))

        fit_obj <- ranger::ranger(
          formula = stats::as.formula(paste0(outcome_col, " ~ .")),
          data = df,
          num.trees = as.integer(num_trees),
          mtry = as.integer(mtry_k),
          min.node.size = as.integer(tg$min_node_size),
          probability = TRUE,
          classification = TRUE,
          respect.unordered.factors = respect_unordered_factors,
          importance = importance,
          case.weights = case_wts,
          replace = isTRUE(tg$replace),
          sample.fraction = as.numeric(tg$sample_fraction),
          write.forest = TRUE,
          ...
        )

        fits[[k]] <- list(model = fit_obj, cols = cols, .tune = tg$.tune)
      }

      list(method = "rf", fits = fits, tune_grid = tune_grid)
    },

    predict = function(fit_bundle, newdata, ...) {
      if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table")
      predict_hazards(fits = fit_bundle$fits, newdata = newdata, ...)
    },

    fit_one = function(train_set, tune, ...) {
      if (!data.table::is.data.table(train_set)) stop("train_set must be a data.table")
      if (!(outcome_col %in% names(train_set))) stop("Missing outcome_col in train_set")
      if (use_weights_col && !(weights_col %in% names(train_set))) stop("Missing weights_col in train_set")

      tg <- tune_grid[tune_grid$.tune == tune, , drop = FALSE]
      if (nrow(tg) != 1L) stop("Invalid tune index")

      cols <- rhs_to_cols(tg$rhs)
      mtry_k <- tg$mtry
      if (is.na(mtry_k)) mtry_k <- default_mtry(length(cols))

      df <- as.data.frame(train_set[, c(outcome_col, cols), with = FALSE])
      y <- df[[outcome_col]]
      if (is.logical(y)) y <- as.integer(y)
      if (is.numeric(y) || is.integer(y)) {
        df[[outcome_col]] <- factor(as.integer(y), levels = c(0L, 1L))
      }

      case_wts <- if (use_weights_col) train_set[[weights_col]] else NULL
      if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(tg$.tune))

      fit_obj <- ranger::ranger(
        formula = stats::as.formula(paste0(outcome_col, " ~ .")),
        data = df,
        num.trees = as.integer(num_trees),
        mtry = as.integer(mtry_k),
        min.node.size = as.integer(tg$min_node_size),
        probability = TRUE,
        classification = TRUE,
        respect.unordered.factors = respect_unordered_factors,
        importance = importance,
        case.weights = case_wts,
        replace = isTRUE(tg$replace),
        sample.fraction = as.numeric(tg$sample_fraction),
        write.forest = TRUE,
        ...
      )

      list(
        method = "rf",
        fits = list(list(model = fit_obj, cols = cols, .tune = tg$.tune)),
        tune_grid = tg
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

      haz <- predict_hazards(fits = fits, newdata = newdata, ...)
      haz <- as.numeric(haz[, 1L])

      dt <- data.table::as.data.table(newdata)
      dt[, .row_id__ := .I]
      data.table::setorderv(dt, c(id_col, bin_var))
      haz <- haz[dt$.row_id__]

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

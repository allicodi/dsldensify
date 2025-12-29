#' Create a random forest runner for discrete-time hazard modeling
#'
#' @description
#' Constructs a **runner** (learner adapter) compatible with the
#' \code{run_grid_setting()} / \code{summarize_and_select()} workflow used in
#' \code{dsl_densify}. The runner fits probabilistic random forest models via
#' \code{ranger::ranger()} on long-format discrete-time hazard data
#' (binary outcome \code{in_bin}), and supports tuning over:
#' \itemize{
#'   \item multiple RHS feature specifications (\code{rhs_list}),
#'   \item node-size parameters (\code{min_node_size_grid}),
#'   \item optional feature-subset sizes (\code{mtry_grid}),
#'   \item sampling behavior (bootstrap vs subsampling).
#' }
#'
#' The fitted models estimate per-bin discrete-time hazards
#' \eqn{P(T \in \text{bin}_j \mid T \ge \text{bin}_j, W)} using a probabilistic
#' classification forest (\code{probability = TRUE}), returning predicted
#' probabilities for \code{in_bin = 1}.
#'
#' @section RHS specifications (column selection only):
#' The \code{rhs_list} argument is used to define a feature set by extracting
#' variable names from each RHS specification:
#' \itemize{
#'   \item RHS may be provided as one-sided formulas (e.g., \code{~ W1 + W2}),
#'         or as character strings (e.g., \code{"W1 + W2"}).
#'   \item Each RHS is converted to a formula internally and variable names are
#'         extracted via \code{all.vars()}.
#'   \item The runner does not evaluate transformations; it assumes the referenced
#'         columns already exist in the long hazard data.
#' }
#'
#' By default, \code{bin_id_col} is enforced as part of the feature set; if it is
#' missing from an RHS specification, it is appended automatically.
#'
#' @section Tuning grid and prediction layout:
#' The internal \code{tune_grid} is ordered such that:
#' \itemize{
#'   \item RHS varies first,
#'   \item then \code{min_node_size},
#'   \item then sampling behavior (\code{sampling}, \code{replace}, \code{sample_fraction}),
#'   \item and optionally \code{mtry} (if \code{mtry_grid} is supplied).
#' }
#'
#' Each row of \code{tune_grid} corresponds to **exactly one fitted forest**.
#' During cross-validation, \code{predict()} returns an \code{n_long x K} matrix
#' of predicted hazards, where \code{K = nrow(tune_grid)}, with columns aligned
#' to \code{.tune}.
#'
#' @section Sampling as a tuning dimension:
#' The runner supports a sampling scheme tuning dimension via \code{sampling_grid}:
#' \itemize{
#'   \item \code{"bootstrap"} uses sampling with replacement
#'         (\code{replace = TRUE}, \code{sample.fraction = 1.0}).
#'   \item \code{"subsample"} uses sampling without replacement
#'         (\code{replace = FALSE}) with fractions provided in
#'         \code{subsample_fraction_grid}.
#' }
#'
#' This allows direct comparison of bootstrap forests and subsampled forests
#' under the same runner interface.
#'
#' @section Weights:
#' If \code{use_weights_col = TRUE}, observation weights are passed to
#' \code{ranger::ranger()} via \code{case.weights}. The weight column defaults
#' to \code{wts}. If \code{use_weights_col = TRUE} and \code{weights_col} is
#' missing, fitting will error.
#'
#' @section Deterministic seeding:
#' If \code{seed} is provided, a tune-specific seed is used for each fit by
#' setting \code{set.seed(seed + .tune)}. This yields deterministic fitting
#' behavior across runs while keeping different tuning rows distinct.
#'
#' @param rhs_list A list of RHS specifications, either:
#' \itemize{
#'   \item one-sided formulas such as \code{~ W1 + W2 + bin_id}, or
#'   \item character strings such as \code{"W1 + W2 + bin_id"}.
#' }
#' These RHS are used for **column selection** only. If \code{bin_id_col} is
#' missing, it is appended automatically.
#'
#' @param mtry_grid Optional integer vector of \code{mtry} values to tune over.
#' If \code{NULL}, \code{mtry} is set per fit to \code{floor(sqrt(p))} (bounded
#' to \code{[1, p]}), where \code{p} is the number of selected features.
#'
#' @param min_node_size_grid Integer vector of minimum terminal node sizes
#' passed to \code{ranger::ranger(min.node.size = ...)}.
#'
#' @param num_trees Integer number of trees in each forest.
#'
#' @param sampling_grid Character vector specifying sampling schemes to include.
#' Must be a subset of \code{c("bootstrap", "subsample")}.
#'
#' @param subsample_fraction_grid Numeric vector of sampling fractions used
#' only when \code{sampling = "subsample"} (sampling without replacement).
#'
#' @param outcome_col Name of the binary outcome column in the long hazard data.
#' Defaults to \code{"in_bin"}.
#'
#' @param bin_id_col Name of the time-bin column. Defaults to \code{"bin_id"}.
#' This column is enforced in every feature set unless already included.
#'
#' @param use_weights_col Logical. If \code{TRUE}, pass \code{weights_col} as
#' \code{case.weights} to \code{ranger::ranger()}.
#'
#' @param weights_col Name of the weights column. Defaults to \code{"wts"}.
#'
#' @param respect_unordered_factors Passed to
#' \code{ranger::ranger(respect.unordered.factors = ...)}.
#'
#' @param importance Passed to \code{ranger::ranger(importance = ...)}.
#'
#' @param seed Optional integer seed. If provided, a deterministic tune-specific
#' seed \code{seed + .tune} is used during fitting.
#'
#' @return A named list (runner) with elements:
#' \describe{
#'   \item{method}{Character string \code{"rf"}.}
#'   \item{tune_grid}{Data frame describing the tuning grid, including \code{.tune}.}
#'   \item{fit}{Function \code{fit(train_set, ...)} returning a fit bundle.}
#'   \item{predict}{Function \code{predict(fit_bundle, newdata, ...)} returning
#'         an \code{n_long x K} matrix of hazard predictions.}
#'   \item{fit_one}{Function \code{fit_one(train_set, tune, ...)} fitting only
#'         the selected tuning index.}
#' }
#'
#' @details
#' ## Data requirements
#' The runner expects \code{train_set} and \code{newdata} as \code{data.table}s
#' in the **long hazard format** produced by \code{format_long_hazards()},
#' including:
#' \itemize{
#'   \item a binary outcome column \code{outcome_col} (default \code{in_bin}),
#'   \item a time-bin column \code{bin_id_col} (default \code{bin_id}),
#'   \item covariates referenced in \code{rhs_list},
#'   \item an optional weight column \code{weights_col} (default \code{wts}).
#' }
#'
#' Internally, the runner fits a probabilistic classification forest by
#' constructing a data.frame containing \code{outcome_col} and the selected
#' feature columns, and fitting \code{outcome_col ~ .}. If the outcome is numeric
#' or integer, it is coerced to a factor with levels \code{c(0, 1)} to ensure
#' probability predictions are returned in a consistent format.
#'
#' ## Prediction
#' \code{predict()} returns the predicted probability for class \code{"1"} from
#' \code{ranger} (i.e., \code{P(in_bin = 1)}). Predictions are returned as an
#' \code{n_long x K} numeric matrix aligned with the tuning grid. Column names
#' are set to the corresponding \code{.tune} values.
#'
#' ## Model selection
#' Each tuning row corresponds to a single fitted forest, so no \code{select_fit()}
#' method is required.
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

  # NEW: sampling scheme as tune dimension
  sampling_grid = c("bootstrap", "subsample"),
  subsample_fraction_grid = c(0.6, 0.8),   # only used when sampling == "subsample"

  outcome_col = "in_bin",
  bin_id_col = "bin_id",
  use_weights_col = TRUE,
  weights_col = "wts",

  respect_unordered_factors = "order",
  importance = "none",
  seed = NULL
) {
  stopifnot(requireNamespace("ranger", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))

  # rhs parsing
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else rhs_chr <- as.character(rhs_list)
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  add_bin_id_if_missing <- function(rhs) {
    vars <- all.vars(stats::as.formula(paste0("~", rhs)))
    if (!(bin_id_col %in% vars)) paste(rhs, "+", bin_id_col) else rhs
  }
  rhs_chr <- vapply(rhs_chr, add_bin_id_if_missing, character(1))

  # --- NEW: build sampling part of grid ---
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

  # rhs-major tuning grid (rhs first)
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

  # add mtry dimension (rhs-major still preserved because rhs is first in expand.grid)
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
        if (is.numeric(y) || is.integer(y)) df[[outcome_col]] <- factor(as.integer(y), levels = c(0L, 1L))

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

          # --- NEW: per-tune sampling behavior ---
          replace = isTRUE(tg$replace),
          sample.fraction = as.numeric(tg$sample_fraction),

          write.forest = TRUE
        )

        fits[[k]] <- list(model = fit_obj, cols = cols, .tune = tg$.tune)
      }

      list(method = "rf", fits = fits, tune_grid = tune_grid)
    },

    predict = function(fit_bundle, newdata, ...) {
      if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table")
      out <- matrix(NA_real_, nrow(newdata), length(fit_bundle$fits))
      for (k in seq_along(fit_bundle$fits)) {
        cols <- fit_bundle$fits[[k]]$cols
        df_new <- as.data.frame(newdata[, cols, with = FALSE])
        pr <- predict(fit_bundle$fits[[k]]$model, data = df_new, type = "response")$predictions
        if (is.matrix(pr) || is.data.frame(pr)) pr <- pr[, "1", drop = TRUE]
        out[, k] <- as.numeric(pr)
      }
      colnames(out) <- fit_bundle$tune_grid$.tune
      out
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
      if (is.numeric(y) || is.integer(y)) df[[outcome_col]] <- factor(as.integer(y), levels = c(0L, 1L))

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
        write.forest = TRUE
      )

      list(
        method = "rf",
        fits = list(list(model = fit_obj, cols = cols, .tune = tg$.tune)),
        tune_grid = tg
      )
    }
  )
}

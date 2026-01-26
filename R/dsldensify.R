utils::globalVariables(c("in_bin", "bin_id"))

#' Discrete super learner for conditional density estimation
#'
#' Fits and selects among candidate conditional density estimators for a
#' continuous outcome \code{A} given covariates \code{W}, using cross-validated
#' negative log-density risk. Candidate estimators may include hazard-based
#' (discretized) density learners, direct conditional density learners, and
#' (optionally) hurdle model components.
#'
#' Hazard-based learners approximate the conditional density by discretizing the
#' support of \code{A} and modeling discrete-time hazards. Direct learners model
#' the conditional density of \code{A | W} without discretization.
#'
#' When \code{hurdle_learners} are supplied, \code{dsldensify()} fits a hurdle
#' density model with a point mass at \code{hurdle_point} and a positive-part
#' density for \code{A != hurdle_point}. In hurdle mode, the function performs:
#' (i) cross-validated selection of a binary learner for
#' \eqn{\pi(W) = P(A = \mathrm{hurdle\_point} \mid W)}, and (ii) cross-validated
#' selection of a positive-part density learner among \code{hazard_learners} and
#' \code{direct_learners} fit on the subset \code{A != hurdle_point}. The final
#' selected hurdle model combines these two components into a valid conditional
#' density for \code{A}.
#'
#' @param A Numeric vector of length \code{n} containing observed outcomes.
#'
#' @param W Covariates used to condition the density. May be a vector, matrix,
#'   \code{data.frame}, or \code{data.table}.
#'
#' @param hazard_learners Named list of hazard-based runner objects for density
#'   estimation. Each runner must support fitting on long-format hazard data and
#'   evaluation of per-bin hazards. In hurdle mode, these are treated as
#'   positive-part candidates and are fit only on observations with
#'   \code{A != hurdle_point}. All positive-part candidates must set
#'   \code{runner$positive_support = TRUE}.
#'
#' @param direct_learners Named list of direct conditional density runner objects.
#'   Each runner must support fitting on wide data and evaluation of the
#'   conditional log-density. In hurdle mode, these are treated as
#'   positive-part candidates and are fit only on observations with
#'   \code{A != hurdle_point}. All positive-part candidates must set
#'   \code{runner$positive_support = TRUE}.
#'
#' @param hurdle_learners Optional named list of hurdle (point-mass) runner
#'   objects. Supplying a non-empty \code{hurdle_learners} list enables hurdle
#'   mode. Each hurdle runner is a binary regression learner fit on wide data
#'   with outcome \code{in_hurdle = as.integer(A == hurdle_point)} and covariates
#'   \code{W}, and must support evaluation of the log-probability
#'   \eqn{\log \hat\pi(W)} used by \code{summarize_and_select_hurdle()}.
#'
#' @param hurdle_point Numeric scalar giving the location of the point mass in
#'   hurdle mode. Default is \code{0}. In hurdle mode, the positive-part density
#'   is estimated on \code{A != hurdle_point}.
#'
#' @param wts Optional numeric vector of length \code{n} giving observation
#'   weights. Weights are passed through to both hurdle and positive-part
#'   learners when supported by the runners.
#'
#' @param grid_type Character vector specifying binning strategies for hazard
#'   learners. Allowed values are \code{"equal_range"} and \code{"equal_mass"}.
#'   In hurdle mode, this grid applies to the positive-part hazard learners only
#'   (fit on \code{A != hurdle_point}).
#'
#' @param n_bins Integer vector giving numbers of bins to consider for hazard
#'   learners. In hurdle mode, this grid applies to the positive-part hazard
#'   learners only (fit on \code{A != hurdle_point}).
#'
#' @param cv_folds Either a single integer specifying the number of v-folds, or a
#'   fold object returned by \code{origami::make_folds()}.
#'
#' @param return_cv_fits Logical; whether to retain fold-specific fits for the
#'   selected learner(s). In non-hurdle mode, fold-specific fits are retained for
#'   the selected density learner. In hurdle mode, fold-specific fits may be
#'   retained for both the selected positive-part learner and the selected hurdle
#'   learner.
#'
#' @param refit_dsl_full_data Logical; whether to refit the selected learner(s) on
#'   the full dataset after selection. In non-hurdle mode, the selected density
#'   learner is refit on all observations. In hurdle mode, the selected hurdle
#'   learner is refit on all observations (binary outcome \code{in_hurdle}), and
#'   the selected positive-part learner is refit on the subset
#'   \code{A != hurdle_point}.
#'
#' @param ... Additional arguments passed to internal fitting routines and to
#'   learner-specific runner methods.
#'
#' @return An object of class \code{"dsldensify"} containing the selected learner,
#'   tuning parameters, cross-validation summaries, and fitted models. In hurdle
#'   mode, the returned object additionally stores the selected hurdle learner,
#'   its tuning choice, and the fitted hurdle component used by
#'   \code{predict.dsldensify()} and \code{rsample.dsldensify()}.
#'
#' @details
#' Hazard-based density estimation relies on the identity
#' \deqn{
#'   f(a \mid w) = \lambda_j(w)
#'   \prod_{k < j} \left\{1 - \lambda_k(w)\right\} \Big/ \Delta_j,
#' }
#' where \eqn{\lambda_j(w)} is the discrete hazard of \eqn{A} falling in bin
#' \eqn{j} given \eqn{W = w}, and \eqn{\Delta_j} is the bin width. By modeling the
#' hazard with flexible binary regression learners and combining bins via the
#' above factorization, one obtains a valid conditional density estimator.
#'
#' In hurdle mode, the conditional distribution of \code{A} is modeled as a
#' two-part mixture:
#' \deqn{
#'   P(A = a \mid W = w) =
#'   \pi(w)\,\mathbb{I}\{a = a_0\} +
#'   \left\{1 - \pi(w)\right\} f_+(a \mid w)\,\mathbb{I}\{a \ne a_0\},
#' }
#' where \eqn{a_0} is \code{hurdle_point}, \eqn{\pi(w) = P(A = a_0 \mid W = w)} is
#' estimated by the selected \code{hurdle_learners} candidate, and
#' \eqn{f_+(a \mid w)} is a positive-support density estimated by the selected
#' hazard-based or direct density candidate fit on \code{A != hurdle_point}.
#' Cross-validated negative log-likelihood provides a proper scoring rule for
#' comparing candidates and selecting a joint hurdle model.
#'
#' @examples
#' set.seed(1)
#'
#' n <- 100
#' W <- data.frame(
#'   x1 = rnorm(n),
#'   x2 = rnorm(n)
#' )
#' A <- 0.5 * W$x1 - 0.3 * W$x2 + rnorm(n)
#'
#' gaussian_runner <- make_gaussian_homosked_runner(
#'   rhs_list = "~ x1 + x2"
#' )
#'
#' fit <- dsldensify(
#'   A = A,
#'   W = W,
#'   hazard_learners = NULL,
#'   direct_learners = list(gaussian = gaussian_runner),
#'   cv_folds = 3
#' )
#'
#' new_W <- data.frame(x1 = c(0, 1), x2 = c(0, 0))
#' dens <- predict(fit, A = c(0, 0), W = new_W)
#' dens
#'
#' @seealso \code{\link{predict.dsldensify}} \code{\link{rsample.dsldensify}}
#' @import data.table origami
#' @export

dsldensify <- function(
  A, W,
  hazard_learners = NULL,
  direct_learners = NULL,
  hurdle_learners = NULL,
  hurdle_point = 0,
  wts = rep(1, length(A)),
  grid_type = c("equal_range", "equal_mass"),
  n_bins = round(c(0.5, 1, 1.5, 2) * sqrt(length(A))),
  cv_folds = 5L, # either numeric or output of origami::make_folds
  return_cv_fits = FALSE,
  refit_dsl_full_data = TRUE,  
  ...
) {

  n <- length(A)
  if (length(wts) != n) stop("wts must have length(A)")

  hurdle_on <- !is.null(hurdle_learners) && length(hurdle_learners) > 0L
  
  # folds
  if (is.numeric(cv_folds) && length(cv_folds) == 1L) {
    V <- as.integer(cv_folds)
    cv_folds_id <- origami::make_folds(
      n = n, fold_fun = origami::folds_vfold, V = V
    )
  } else {
    cv_folds_id <- cv_folds
    V <- length(cv_folds_id)
  }

  # fold id lookup
  id_fold <- integer(n)
  for (v in seq_len(V)) {
    id_fold[cv_folds_id[[v]]$validation_set] <- v
  }

  if(!hurdle_on){
    # hazard grid
    tune_grid <- expand.grid(
      grid_type = grid_type,
      n_bins = n_bins,
      stringsAsFactors = FALSE
    )
    grid_idx <- seq_len(nrow(tune_grid))

    select_out <- list()

    # hazard settings (vary grid_type/n_bins)
    if (!is.null(hazard_learners) && length(hazard_learners) > 0L) {

      hazard_out <- future.apply::future_lapply(
        grid_idx,
        function(i) {
          run_grid_setting(
            grid_type = tune_grid$grid_type[i],
            n_bins = tune_grid$n_bins[i],
            A = A, W = W, wts = wts,
            cv_folds_id = cv_folds_id,
            id_fold = id_fold,
            learners = hazard_learners,
            return_fits = return_cv_fits,
            ...
          )
        },
        future.seed = TRUE
      )

      select_out <- c(select_out, hazard_out)
    }

    # direct setting (run once; no binning grid)
    if (!is.null(direct_learners) && length(direct_learners) > 0L) {

      direct_out <- run_direct_setting(
        A = A, W = W, wts = wts,
        cv_folds_id = cv_folds_id,
        id_fold = id_fold,
        learners = direct_learners,
        return_fits = return_cv_fits,
        return_density = FALSE,
        ...
      )

      select_out <- c(select_out, list(direct_out))
    }

    if (length(select_out) < 1L) stop("No learners provided: hazard_learners and direct_learners are both empty.")

    select_summary <- summarize_and_select(
      select_out,
      hazard_learners = hazard_learners,
      direct_learners = direct_learners,
      weighted = TRUE
    )

    best_model <- select_summary$best
    best_tune <- best_model$.tune

    gs <- find_grid_setting(
      select_out = select_out,
      grid_type = best_model$grid_type,
      n_bins = if ("n_bins" %in% names(best_model)) best_model$n_bins else NA_integer_
    )

    runner <- if (!is.null(hazard_learners) && best_model$learner %in% names(hazard_learners)) {
      hazard_learners[[best_model$learner]]
    } else if (!is.null(direct_learners) && best_model$learner %in% names(direct_learners)) {
      direct_learners[[best_model$learner]]
    } else {
      stop("Best learner not found in hazard_learners or direct_learners.")
    }

    is_direct_winner <- identical(gs$grid_type, "direct") || is.na(gs$n_bins)

    cv_fit <- NULL
    if (return_cv_fits) {
      cv_fit <- lapply(gs$cv_out, function(fold_obj) {
        L <- fold_obj$learners[[best_model$learner]]
        fit_sel <- select_fit_tune(runner, L$fit, best_tune)

        list(
          fold = fold_obj$fold,
          validation_ids = cv_folds_id[[fold_obj$fold]]$validation_set,
          fit = fit_sel
        )
      })
    }

    # ---- refit on full data ----
    full_fit <- NULL
    if (refit_dsl_full_data) {

      if (is_direct_winner) {
        # direct density learners fit on wide data (A + W [+ wts])
        W_dt <- data.table::as.data.table(W)
        wide_dt <- data.table::data.table(A = A)
        wide_dt <- cbind(wide_dt, W_dt)
        wide_dt[, wts := wts]

        if (!is.null(runner$fit_one) && is.function(runner$fit_one)) {
          full_fit <- runner$fit_one(train_set = wide_dt, tune = best_tune)
        } else {
          full_fit <- select_fit_tune(runner, runner$fit(train_set = wide_dt), best_tune)
        }

      } else {
        # hazard learners refit on long hazards using selected binning
        long_haz <- format_long_hazards(
          A = A, W = W, wts = wts,
          n_bins = gs$n_bins, grid_type = gs$grid_type
        )
        long_data <- data.table::as.data.table(long_haz$data)

        if (!is.null(runner$fit_one) && is.function(runner$fit_one)) {
          full_fit <- runner$fit_one(train_set = long_data, tune = best_tune)
        } else {
          full_fit <- select_fit_tune(runner, runner$fit(train_set = long_data), best_tune)
        }

        # ensure gs contains the bin metadata used by predict.dsldensify
        if (is.null(gs$breaks))     gs$breaks <- long_haz$breaks
        if (is.null(gs$bin_length)) gs$bin_length <- long_haz$bin_length
      }
    }

    out <- list(
      call = match.call(),
      is_hurdle = FALSE, 
      A_range = c(min(A), max(A)),

      # selection summaries + winner
      select_summary = select_summary,
      best_model = best_model,
      best_tune = best_tune,

      # chosen grid setting (only relevant for hazard winner)
      grid_type = gs$grid_type,
      n_bins = gs$n_bins,
      breaks = gs$breaks,
      bin_length = gs$bin_length,

      # chosen learner + tuning row (if available)
      learner = best_model$learner,
      runner = runner,
      tune_row = {
        tg <- runner$tune_grid
        if (is.null(tg) || nrow(tg) == 0L) {
          data.frame(.tune = 1L)
        } else {
          if (!(".tune" %in% names(tg))) tg$.tune <- seq_len(nrow(tg))
          tg[tg$.tune == best_tune, , drop = FALSE]
        }
      },

      # folds / mapping (useful for CV prediction later)
      cv_folds_id = cv_folds_id,
      id_fold = id_fold,

      # fits (optional)
      cv_fit = cv_fit,
      full_fit = full_fit,

      # indicate winner type for prediction dispatch later
      is_direct = is_direct_winner
    )
  }else{
     # positive-part candidates are hazard + direct learners
    pos_hazard <- hazard_learners
    pos_direct <- direct_learners

    if ((is.null(pos_hazard) || length(pos_hazard) < 1L) && (is.null(pos_direct) || length(pos_direct) < 1L)) {
      stop("In hurdle mode, you must supply hazard_learners and/or direct_learners as positive-part candidates.")
    }

    pos_learners <- c(pos_hazard, pos_direct)
    bad_pos <- names(pos_learners)[!vapply(pos_learners, function(r) isTRUE(r$positive_support), logical(1))]
    if (length(bad_pos)) {
      stop("In hurdle mode, all positive-part learners must set runner$positive_support = TRUE. Missing: ",
          paste(bad_pos, collapse = ", "))
    }

    # define masks / subsets
    keep_h <- (A == hurdle_point)
    keep_pos <- !keep_h
    pos_ids_full <- which(keep_pos)

    n_pos <- length(pos_ids_full)
    if (n_pos < 1L) stop("In hurdle mode, no observations are in the positive part (A != hurdle_point).")

    cv_folds_id_pos <- lapply(seq_len(V), function(v) {
      vs_full <- cv_folds_id[[v]]$validation_set         # full-data ids
      # subset indices whose full ids are in this fold's validation set
      vs_pos_idx <- which(pos_ids_full %in% vs_full)     # indices in 1..n_pos
      list(validation_set = vs_pos_idx)
    })

    id_fold_pos <- integer(n_pos)
    for (v in seq_len(V)) {
      id_fold_pos[cv_folds_id_pos[[v]]$validation_set] <- v
    }
    if (any(id_fold_pos == 0L)) stop("Some positive rows were not assigned to a fold.")

    tune_grid <- expand.grid(
      grid_type = grid_type,
      n_bins = n_bins,
      stringsAsFactors = FALSE
    )
    grid_idx <- seq_len(nrow(tune_grid))

    pos_select_out <- list()

    if (!is.null(pos_hazard) && length(pos_hazard) > 0L) {
      hazard_out_pos <- future.apply::future_lapply(
        grid_idx,
        function(i) {
          run_grid_setting(
            grid_type = tune_grid$grid_type[i],
            n_bins = tune_grid$n_bins[i],
            A = A[keep_pos], W = W[keep_pos, , drop = FALSE], wts = wts[keep_pos],
            cv_folds_id = cv_folds_id_pos,
            id_fold = id_fold_pos,
            ids_full = pos_ids_full,
            learners = pos_hazard,
            return_fits = return_cv_fits,
            ...
          )
        },
        future.seed = TRUE
      )
      pos_select_out <- c(pos_select_out, hazard_out_pos)
    }

    if (!is.null(pos_direct) && length(pos_direct) > 0L) {
      direct_out_pos <- run_direct_setting(
        A = A[keep_pos], W = W[keep_pos, , drop = FALSE], wts = wts[keep_pos],
        cv_folds_id = cv_folds_id_pos,
        id_fold = id_fold_pos,
        ids_full = pos_ids_full,
        learners = pos_direct,
        return_fits = return_cv_fits,
        return_density = FALSE,
        ...
      )
      pos_select_out <- c(pos_select_out, list(direct_out_pos))
    }

    if (length(pos_select_out) < 1L) stop("In hurdle mode, positive-part selection produced no settings. Check learners.")

    hurdle_out <- run_hurdle_setting(
      A = A, W = W, wts = wts,
      hurdle_point = hurdle_point,
      cv_folds_id = cv_folds_id,
      id_fold = id_fold,
      learners = hurdle_learners,
      return_fits = return_cv_fits,
      ...
    )

    select_summary <- summarize_and_select_hurdle(
      pos_select_out = pos_select_out,
      hurdle_out = hurdle_out,
      hazard_learners = pos_hazard,
      direct_learners = pos_direct,
      hurdle_learners = hurdle_learners
    )

    best_model <- select_summary$best
    best_hurdle_tune <- best_model$hurdle_tune
    best_pos_tune    <- best_model$pos_tune

    gs_pos <- find_grid_setting(
      select_out = pos_select_out,
      grid_type  = best_model$grid_type,
      n_bins     = best_model$n_bins
    )

    pos_runner <- if (!is.null(hazard_learners) && best_model$pos_learner %in% names(hazard_learners)) {
      hazard_learners[[best_model$pos_learner]]
    } else if (!is.null(direct_learners) && best_model$pos_learner %in% names(direct_learners)) {
      direct_learners[[best_model$pos_learner]]
    } else {
      stop("Best positive learner not found in hazard_learners or direct_learners.")
    }
    is_direct_pos_winner <- identical(gs_pos$grid_type, "direct")

    hurdle_runner <- if (best_model$hurdle_learner %in% names(hurdle_learners)) {
      hurdle_learners[[best_model$hurdle_learner]]
    } else {
      stop("Best hurdle learner not found in hurdle_learners.")
    }

    cv_fit_pos <- NULL
    cv_fit_hurdle <- NULL
    if (return_cv_fits) {
      cv_fit_pos <- lapply(gs_pos$cv_out, function(fold_obj) {
        L <- fold_obj$learners[[best_model$pos_learner]]
        fit_sel <- select_fit_tune(pos_runner, L$fit, best_pos_tune)

        list(
          fold = fold_obj$fold,
          validation_ids = fold_obj$validation_ids_full,  # IMPORTANT: full-data ids
          fit = fit_sel
        )
      })

      cv_fit_hurdle <- lapply(hurdle_out$cv_out, function(fold_obj) {
        L <- fold_obj$learners[[best_model$hurdle_learner]]
        fit_sel <- select_fit_tune(hurdle_runner, L$fit, best_hurdle_tune)

        list(
          fold = fold_obj$fold,
          validation_ids = fold_obj$validation_ids_full,  # full-fold validation ids
          fit = fit_sel
        )
      })
    }

    # ---- refit on full data ----
    hurdle_full_fit <- NULL
    pos_full_fit <- NULL

    if (refit_dsl_full_data) {

      # ---------- hurdle refit on full data ----------
      W_dt <- data.table::as.data.table(W)
      wide_dt <- data.table::data.table(in_hurdle = as.integer(A == hurdle_point))
      wide_dt <- cbind(wide_dt, W_dt)
      wide_dt[, wts := wts]

      if (!is.null(hurdle_runner$fit_one) && is.function(hurdle_runner$fit_one)) {
        hurdle_full_fit <- hurdle_runner$fit_one(train_set = wide_dt, tune = best_hurdle_tune)
      } else {
        hurdle_full_fit <- select_fit_tune(
          hurdle_runner,
          hurdle_runner$fit(train_set = wide_dt),
          best_hurdle_tune
        )
      }

      # ---------- (2) positive refit on positive-only data ----------
      keep_pos <- (A != hurdle_point)
      A_pos   <- A[keep_pos]
      W_pos   <- W[keep_pos, , drop = FALSE]
      wts_pos <- wts[keep_pos]

      if (is_direct_pos_winner) {
        # direct positive learners fit on wide positive-only data
        Wp_dt <- data.table::as.data.table(W_pos)
        wide_p <- data.table::data.table(A = A_pos)
        wide_p <- cbind(wide_p, Wp_dt)
        wide_p[, wts := wts_pos]

        if (!is.null(pos_runner$fit_one) && is.function(pos_runner$fit_one)) {
          pos_full_fit <- pos_runner$fit_one(train_set = wide_p, tune = best_pos_tune)
        } else {
          pos_full_fit <- select_fit_tune(
            pos_runner,
            pos_runner$fit(train_set = wide_p),
            best_pos_tune
          )
        }

      } else {
        # hazard positive learners refit on long hazards constructed from positive-only data
        long_haz_pos <- format_long_hazards(
          A = A_pos, W = W_pos, wts = wts_pos,
          n_bins = gs_pos$n_bins, grid_type = gs_pos$grid_type
        )
        long_data_pos <- data.table::as.data.table(long_haz_pos$data)

        if (!is.null(pos_runner$fit_one) && is.function(pos_runner$fit_one)) {
          pos_full_fit <- pos_runner$fit_one(train_set = long_data_pos, tune = best_pos_tune)
        } else {
          pos_full_fit <- select_fit_tune(
            pos_runner,
            pos_runner$fit(train_set = long_data_pos),
            best_pos_tune
          )
        }

        # ensure gs_pos contains the bin metadata used by predict() for the POSITIVE component
        if (is.null(gs_pos$breaks))     gs_pos$breaks <- long_haz_pos$breaks
        if (is.null(gs_pos$bin_length)) gs_pos$bin_length <- long_haz_pos$bin_length
      }
    }

    out <- list(
      call = match.call(),
      is_hurdle = TRUE,
      hurdle_point = hurdle_point,
      A_range = c(min(A[A != hurdle_point]), max(A[A != hurdle_point])),

      # selection summaries + winner (JOINT best row)
      select_summary = select_summary,
      best_model = best_model,

      # best POSITIVE tune (kept in best_tune for compatibility with predict()/sample())
      best_tune = best_pos_tune,

      # chosen POSITIVE grid setting 
      grid_type  = gs_pos$grid_type,
      n_bins     = gs_pos$n_bins,
      breaks     = gs_pos$breaks,
      bin_length = gs_pos$bin_length,

      # chosen POSITIVE learner + tuning row (if available)
      learner = best_model$pos_learner,
      runner  = pos_runner,
      tune_row = {
        tg <- pos_runner$tune_grid
        if (is.null(tg) || nrow(tg) == 0L) {
          data.frame(.tune = 1L)
        } else {
          if (!(".tune" %in% names(tg))) tg$.tune <- seq_len(nrow(tg))
          tg[tg$.tune == best_pos_tune, , drop = FALSE]
        }
      },

      # full-data folds / mapping
      cv_folds_id = cv_folds_id,
      id_fold     = id_fold,

      # POSITIVE fits 
      cv_fit   = cv_fit_pos,
      full_fit = pos_full_fit,

      # POSITIVE 
      is_direct = is_direct_pos_winner,

      # hurdle-specific components
      hurdle_learner = as.character(best_model[["hurdle_learner"]][1L]),
      hurdle_runner = hurdle_runner,
      hurdle_best_tune = best_hurdle_tune,
      hurdle_tune_row = {
        tg <- hurdle_runner$tune_grid
        if (is.null(tg) || nrow(tg) == 0L) {
          data.frame(.tune = 1L)
        } else {
          if (!(".tune" %in% names(tg))) tg$.tune <- seq_len(nrow(tg))
          tg[tg$.tune == best_hurdle_tune, , drop = FALSE]
        }
      },
      hurdle_cv_fit = cv_fit_hurdle,
      hurdle_full_fit = hurdle_full_fit
    )
  }

  class(out) <- "dsldensify"
  return(out)
}

#' Predict conditional densities from a fitted dsldensify object
#'
#' Computes conditional density estimates \eqn{f(A \mid W)} from an object
#' returned by \code{dsldensify()}. Predictions can be produced either from a
#' full-data refit (\code{type = "full"}) or using fold-specific fits for
#' cross-validated prediction (\code{type = "cv"}).
#'
#' If the selected model is a direct density learner, densities are obtained by
#' evaluating the learner's conditional log-density and exponentiating. If the
#' selected model is a hazard-based learner, new data are expanded to long-format
#' hazards using the stored bin definitions; predicted hazards are converted to
#' bin probabilities and then to a continuous density by dividing by bin widths.
#'
#' If the fitted object was obtained in hurdle mode
#' (\code{object$is_hurdle == TRUE}), this method returns the density from a
#' two-part model with a point mass at \code{object$hurdle_point} and a
#' positive-part conditional density for \code{A != hurdle_point}. In hurdle mode,
#' the hurdle probability is obtained from the selected hurdle learner and the
#' positive-part density is obtained from the selected hazard-based or direct
#' density learner fit on \code{A != hurdle_point}.
#'
#' @param object Fitted \code{"dsldensify"} object returned by
#'   \code{dsldensify()}.
#'
#' @param A Numeric vector of values at which to evaluate the conditional density.
#'   Must have the same length as \code{W}.
#'
#' @param W Covariate values at which to condition the density. May be a vector,
#'   matrix, \code{data.frame}, or \code{data.table}. Each row of \code{W} is
#'   paired with the corresponding element of \code{A}.
#'
#' @param type Character string indicating which fitted object to use.
#'   \code{"full"} uses the full-data refit stored in \code{object$full_fit};
#'   \code{"cv"} uses fold-specific fits stored in \code{object$cv_fit}.
#'
#' @param fold_id Optional integer vector of fold assignments used when
#'   \code{type = "cv"}. Must have length \code{length(A)} and take values in
#'   \code{1, \dots, V}, where \eqn{V} is the number of folds. Each entry indicates
#'   which fold-specific fit should be used for the corresponding observation.
#'
#' @param trim_predict Logical. When the selected model is hazard-based and
#'   \code{TRUE}, replaces predicted densities for \code{A} values outside the
#'   training support with a small positive value. This is intended to avoid
#'   returning exactly zero density due to extrapolation beyond the bin range.
#'
#' @param eps Small positive constant used to bound probabilities and densities
#'   away from zero during computation. For direct learners, it is passed to the
#'   runner's \code{log_density()} method when supported. In hurdle mode, it is
#'   also used to stabilize the hurdle probability estimates.
#'
#' @param .ignore_hurdle Logical. Internal flag used to bypass hurdle composition
#'   logic when recursively evaluating the positive-part density. Users should
#'   not set this argument directly.
#'
#' @param ... Additional arguments passed to the selected learner runner's
#'   prediction methods (\code{predict()} for hazard learners;
#'   \code{log_density()} for direct learners; and \code{logpi()} for hurdle
#'   learners).
#'
#' @return A numeric vector of length \code{length(A)} containing estimated
#'   conditional densities evaluated at each paired \code{(A_i, W_i)}.
#'
#' @details
#' For hazard-based models, let \eqn{\lambda_j(w)} denote the discrete hazard of
#' falling in bin \eqn{j} given \eqn{W = w}, and let \eqn{\Delta_j} denote the
#' bin width. The implied bin probability mass is
#' \deqn{
#'   p_j(w) = \lambda_j(w) \prod_{k < j} \left\{1 - \lambda_k(w)\right\},
#' }
#' and the corresponding density estimate for \eqn{A} in bin \eqn{j} is
#' \deqn{
#'   \hat f(a \mid w) = p_j(w) / \Delta_j.
#' }
#' This function constructs the necessary long-format data at prediction time
#' using \code{object$breaks} and \code{object$bin_length}, obtains hazard
#' predictions from the selected runner, converts them to bin mass, and then
#' scales by the bin widths.
#'
#' In hurdle mode, the conditional distribution of \code{A} is modeled as a
#' two-part mixture with a point mass at \eqn{a_0 = \code{hurdle_point}}:
#' \deqn{
#'   f(a \mid w) =
#'   \pi(w)\,\mathbb{I}\{a = a_0\} +
#'   \left\{1 - \pi(w)\right\} f_+(a \mid w)\,\mathbb{I}\{a \ne a_0\},
#' }
#' where \eqn{\pi(w) = P(A = a_0 \mid W = w)} is estimated by the selected hurdle
#' learner (via its \code{logpi()} method), and \eqn{f_+(a \mid w)} is the
#' selected positive-part density fit on observations with \code{A != a_0}. This
#' method evaluates \eqn{\pi(w)} on the requested covariates, then evaluates the
#' positive-part density by recursively calling \code{predict.dsldensify()} with
#' \code{.ignore_hurdle = TRUE}, and finally composes the two components into the
#' overall density.
#'
#' When \code{type = "cv"}, predictions are computed using the fold-specific fit
#' indexed by \code{fold_id}. This supports cross-fitted workflows, where each
#' observation must be predicted using a model that did not train on that
#' observation.
#'
#' @examples
#' set.seed(1)
#'
#' n <- 80
#' W <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
#' A <- 0.6 * W$x1 - 0.2 * W$x2 + rnorm(n)
#'
#' gaussian_runner <- make_gaussian_homosked_runner(rhs_list = "~ x1 + x2")
#'
#' fit <- dsldensify(
#'   A = A,
#'   W = W,
#'   hazard_learners = NULL,
#'   direct_learners = list(gaussian = gaussian_runner),
#'   cv_folds = 3,
#'   return_cv_fits = TRUE,
#'   refit_dsl_full_data = TRUE
#' )
#'
#' A_new <- c(0, 0)
#' W_new <- data.frame(x1 = c(0, 1), x2 = c(0, 0))
#' predict(fit, A = A_new, W = W_new, type = "full")
#'
#' fold_id <- fit$id_fold
#' predict(fit, A = A, W = W, type = "cv", fold_id = fold_id)
#'
#' @seealso \code{\link{dsldensify}} \code{\link{rsample.dsldensify}}
#' @method predict dsldensify
#' @export


predict.dsldensify <- function(
  object,
  A, W,
  type = c("full", "cv"),
  fold_id = NULL,
  trim_predict = TRUE,
  eps = 1e-12,
  .ignore_hurdle = FALSE,
  ...
) {

  type <- match.arg(type)
  n <- length(A)
  if (is.null(W)) stop("`W` must be provided.")

  # ---------------------------------------------------------
  # 1) HURDLE WRAPPER (thin): compose pi_hat and f_plus
  #    Uses object$hurdle_runner$logpi() and recurses into this
  #    same method with .ignore_hurdle=TRUE (no object copying).
  # ---------------------------------------------------------
  if (isTRUE(object$is_hurdle) && !isTRUE(.ignore_hurdle)) {

    hp <- object$hurdle_point
    W_dt <- data.table::as.data.table(W)
    if (nrow(W_dt) != n) stop("A and W row mismatch.")

    if (is.null(object$hurdle_runner$logpi) || !is.function(object$hurdle_runner$logpi)) {
      stop("Hurdle runner must define logpi(fit, newdata, ...).")
    }

    clamp01 <- function(p) pmin(pmax(p, eps), 1 - eps)

    # compute pi_hat(W)
    if (type == "full") {

      if (is.null(object$hurdle_full_fit)) stop("No hurdle_full_fit stored in object.")
      logpi <- as.numeric(object$hurdle_runner$logpi(
        object$hurdle_full_fit,
        newdata = W_dt,
        eps = eps,
        ...
      ))
      if (length(logpi) != n) stop("hurdle_full_fit logpi length mismatch.")
      pi_hat <- clamp01(exp(logpi))

    } else {

      if (is.null(fold_id)) stop("`fold_id` must be provided for type='cv'.")
      if (length(fold_id) != n) stop("`fold_id` must have length equal to length(A).")
      if (anyNA(fold_id)) stop("`fold_id` contains NA; every observation must be assigned a fold.")
      if (is.null(object$hurdle_cv_fit) || length(object$hurdle_cv_fit) == 0L) {
        stop("No hurdle_cv_fit stored in object. Fit with return_cv_fits=TRUE for hurdle component or use type='full'.")
      }

      V <- length(object$hurdle_cv_fit)
      if (any(fold_id < 1L | fold_id > V)) stop("`fold_id` must be integers in 1..", V)

      fit_by_fold <- vector("list", V)
      for (j in seq_along(object$hurdle_cv_fit)) {
        v <- object$hurdle_cv_fit[[j]]$fold
        fit_by_fold[[v]] <- object$hurdle_cv_fit[[j]]$fit
      }
      if (any(vapply(fit_by_fold, is.null, logical(1)))) {
        stop("hurdle_cv_fit is missing one or more folds; cannot do type='cv' prediction.")
      }

      logpi <- numeric(n)
      for (v in seq_len(V)) {
        ids_v <- which(fold_id == v)
        if (!length(ids_v)) next
        logpi[ids_v] <- as.numeric(object$hurdle_runner$logpi(
          fit_by_fold[[v]],
          newdata = W_dt[ids_v],
          eps = eps,
          ...
        ))
      }
      pi_hat <- clamp01(exp(logpi))
    }

    is_h <- (A == hp)
    dens <- numeric(n)

    if (any(!is_h)) {
      dens_pos <- predict.dsldensify(
        object,
        A = A[!is_h],
        W = W_dt[!is_h],
        type = type,
        fold_id = if (type == "cv") fold_id[!is_h] else NULL,
        trim_predict = trim_predict,
        eps = eps,
        .ignore_hurdle = TRUE,
        ...
      )
      dens[!is_h] <- (1 - pi_hat[!is_h]) * dens_pos
    }

    dens[is_h] <- pi_hat[is_h]

    if (isTRUE(trim_predict)) dens <- pmax(dens, eps)
    return(dens)
  }

  # ---------------------------------------------------------
  # 2) POSITIVE (non-hurdle) LOGIC: your existing code
  # ---------------------------------------------------------

  train_n <- length(object$id_fold)

  # helper: trim outside training support (hazard only)
  trim_outside <- function(dens, A, a_min, a_max, train_n) {
    dens[A < a_min | A > a_max] <- 5 / (sqrt(train_n) * log(train_n))
    dens
  }

  # ---- DIRECT density winner ----
  if (object$is_direct) {

    # wide newdata (A + W)
    W_dt <- data.table::as.data.table(W)
    wide_dt <- data.table::data.table(A = A)
    wide_dt <- cbind(wide_dt, W_dt)

    if (type == "full") {
      if (is.null(object$full_fit)) {
        stop("No full_fit stored in `object`. Fit with refit_dsl_full_data=TRUE or use type='cv'.")
      }

      logf_mat <- object$runner$log_density(object$full_fit, wide_dt, eps = eps, ...)
      if (is.null(dim(logf_mat))) logf_mat <- matrix(logf_mat, ncol = 1L)
      if (ncol(logf_mat) > 1L) logf_mat <- logf_mat[, object$best_tune, drop = FALSE]

      return(as.numeric(exp(logf_mat[, 1L])))
    }

    # type == "cv"
    if (is.null(object$cv_fit) || length(object$cv_fit) == 0L) {
      stop("No cv_fit stored in `object`. Fit with return_cv_fits=TRUE or use type='full'.")
    }
    if (is.null(fold_id)) stop("`fold_id` must be provided for type='cv'.")
    if (length(fold_id) != n) stop("`fold_id` must have length equal to length(A).")
    if (anyNA(fold_id)) stop("`fold_id` contains NA; every observation must be assigned a fold.")

    V <- length(object$cv_fit)
    if (any(fold_id < 1L | fold_id > V)) stop("`fold_id` must be integers in 1..", V)

    fit_by_fold <- vector("list", V)
    for (j in seq_along(object$cv_fit)) {
      v <- object$cv_fit[[j]]$fold
      fit_by_fold[[v]] <- object$cv_fit[[j]]$fit
    }
    if (any(vapply(fit_by_fold, is.null, logical(1)))) {
      stop("cv_fit is missing one or more folds; cannot do type='cv' prediction.")
    }

    dens_out <- numeric(n)

    for (v in seq_len(V)) {
      ids_v <- which(fold_id == v)
      if (!length(ids_v)) next

      nd_v <- wide_dt[ids_v]
      logf_mat_v <- object$runner$log_density(fit_by_fold[[v]], nd_v, eps = eps, ...)
      if (is.null(dim(logf_mat_v))) logf_mat_v <- matrix(logf_mat_v, ncol = 1L)
      if (ncol(logf_mat_v) > 1L) logf_mat_v <- logf_mat_v[, 1L, drop = FALSE]

      dens_out[ids_v] <- as.numeric(exp(logf_mat_v[, 1L]))
    }

    return(dens_out)
  }

  # ---- HAZARD density winner (existing behavior) ----
  if (is.null(object$breaks) || is.null(object$bin_length)) {
    stop("Hazard-based model is missing `breaks` / `bin_length` in object.")
  }

  breaks_full <- c(object$breaks, tail(object$breaks, 1) + tail(object$bin_length, 1))

  # build long data using selected grid specification
  long_haz <- format_long_hazards(
    A = A, W = W, breaks = breaks_full
  )
  long_data <- data.table::as.data.table(long_haz$data)

  if (type == "full") {
    if (is.null(object$full_fit)) {
      stop("No full_fit stored in `object`. Fit with refit_dsl_full_data=TRUE or use type='cv'.")
    }

    grp_long <- data.table::rleid(long_data$obs_id)
    is_last_long <- c(grp_long[-1L] != grp_long[-length(grp_long)], TRUE)

    pred_mat <- object$runner$predict(object$full_fit, long_data, ...)

    if (is.null(dim(pred_mat))) pred_mat <- matrix(pred_mat, ncol = 1L)
    if (ncol(pred_mat) > 1L) pred_mat <- pred_mat[, object$best_tune, drop = FALSE]

    mass_mat <- hazards_to_mass_by_obs(
      preds   = pred_mat,
      grp     = grp_long,
      is_last = is_last_long,
      eps     = eps
    )

    bin_last <- long_data$bin_id[is_last_long]
    dens <- as.numeric(mass_mat[, 1L]) / object$bin_length[bin_last]

    if (trim_predict) {
      a_min <- breaks_full[1L]
      a_max <- breaks_full[length(breaks_full)]
      dens <- trim_outside(dens, A, a_min, a_max, train_n)
    }
    return(dens)
  }

  # type == "cv"
  if (is.null(object$cv_fit) || length(object$cv_fit) == 0L) {
    stop("No cv_fit stored in `object`. Fit with return_cv_fits=TRUE or use type='full'.")
  }
  if (is.null(fold_id)) stop("`fold_id` must be provided for type='cv'.")
  if (length(fold_id) != n) stop("`fold_id` must have length equal to length(A).")
  if (anyNA(fold_id)) stop("`fold_id` contains NA; every observation must be assigned a fold.")

  V <- length(object$cv_fit)
  if (any(fold_id < 1L | fold_id > V)) stop("`fold_id` must be integers in 1..", V)

  fit_by_fold <- vector("list", V)
  for (j in seq_along(object$cv_fit)) {
    v <- object$cv_fit[[j]]$fold
    fit_by_fold[[v]] <- object$cv_fit[[j]]$fit
  }
  if (any(vapply(fit_by_fold, is.null, logical(1)))) {
    stop("cv_fit is missing one or more folds; cannot do type='cv' prediction.")
  }

  dens_out <- numeric(n)

  for (v in seq_len(V)) {
    ids_v <- which(fold_id == v)
    if (!length(ids_v)) next

    ld_v <- long_data[obs_id %in% ids_v]
    if (nrow(ld_v) == 0L) next

    grp_v <- data.table::rleid(ld_v$obs_id)
    is_last_v <- c(grp_v[-1L] != grp_v[-length(grp_v)], TRUE)

    pred_mat_v <- object$runner$predict(fit_by_fold[[v]], ld_v, ...)

    if (is.null(dim(pred_mat_v))) pred_mat_v <- matrix(pred_mat_v, ncol = 1L)
    if (ncol(pred_mat_v) > 1L) pred_mat_v <- pred_mat_v[, 1L, drop = FALSE]

    mass_mat_v <- hazards_to_mass_by_obs(
      preds   = pred_mat_v,
      grp     = grp_v,
      is_last = is_last_v,
      eps     = eps
    )

    bin_last_v <- ld_v$bin_id[is_last_v]
    dens_vec_v <- as.numeric(mass_mat_v[, 1L]) / object$bin_length[bin_last_v]

    obs_unique <- unique(ld_v$obs_id)
    dens_out[obs_unique] <- dens_vec_v
  }

  if (trim_predict) {
    a_min <- breaks_full[1L]
    a_max <- breaks_full[length(breaks_full)]
    dens_out <- trim_outside(dens_out, A, a_min, a_max, train_n)
  }

  return(dens_out)
}


#' @export
rsample <- function(object, ...) UseMethod("rsample")

#' Sample from a fitted dsldensify conditional density
#'
#' @description
#' Draws samples \eqn{A^* \sim \hat f(\cdot \mid W)} from a fitted
#' \code{dsldensify} object.
#'
#' If the selected model is hazard-based, this method constructs the required
#' long-format hazard grid internally (by repeating each row of \code{W} over all
#' bins) using the bin definitions stored in \code{object}. It then delegates to
#' the hazard runner's \code{sample()} method, which assumes a selected fit
#' (\eqn{K = 1}) and samples via bin mass and within-bin uniform draws.
#'
#' If the selected model is direct, this method delegates directly to the
#' runner's \code{sample()} method on wide \code{W}.
#'
#' If the fitted object was obtained in hurdle mode
#' (\code{object$is_hurdle == TRUE}), this method draws samples from a two-part
#' model with a point mass at \code{object$hurdle_point} and a positive-part
#' conditional density for \code{A != hurdle_point}. In hurdle mode, the hurdle
#' probabilities are obtained from the selected hurdle learner and the
#' positive-part samples are obtained from the selected hazard-based or direct
#' density learner fit on \code{A != hurdle_point}.
#'
#' @param object A fitted \code{dsldensify} object.
#'
#' @param W A \code{data.frame}, \code{data.table}, matrix, or vector of
#'   covariates at which to sample. Each row corresponds to one conditioning
#'   covariate value \eqn{W_i}.
#'
#' @param n_samp Integer number of draws per row of \code{W}.
#'
#' @param type Character; \code{"full"} uses \code{object$full_fit}.
#'   \code{"cv"} uses fold-specific fits stored in \code{object$cv_fit}.
#'
#' @param fold_id Optional integer vector of fold assignments used when
#'   \code{type = "cv"}. Must have length \code{nrow(W)} and take values in
#'   \code{1, \dots, V}, where \eqn{V} is the number of folds. Each entry
#'   indicates which fold-specific fit should be used for the corresponding row.
#'   In hurdle mode, the same \code{fold_id} is used for both the hurdle
#'   component and the positive-part component.
#'
#' @param seed Optional integer seed passed to \code{set.seed()} before sampling.
#'
#' @param .ignore_hurdle Logical. Internal flag used to bypass hurdle composition
#'   logic when recursively sampling from the positive-part component. Users
#'   should not set this argument directly.
#'
#' @param ... Passed through to the underlying runner \code{sample()} method and,
#'   in hurdle mode, to the hurdle runner's \code{logpi()} method.
#'
#' @return A numeric matrix of dimension \code{nrow(W) x n_samp}.
#'
#' @details
#' For hazard-based winners, the method constructs a long-format prediction grid
#' with one row per observation--bin pair. The grid includes \code{obs_id},
#' \code{bin_id}, and bin endpoints \code{bin_lower} and \code{bin_upper}, as
#' well as the covariates \code{W}. The hazard runner's \code{sample()} method is
#' then called on this long data and returns draws for each \code{obs_id}.
#'
#' In hurdle mode, samples are drawn from the two-part mixture distribution.
#'
#' @export

rsample.dsldensify <- function(
  object,
  W,
  n_samp = 1L,
  type = c("full", "cv"),
  fold_id = NULL,
  seed = NULL,
  .ignore_hurdle = FALSE,
  ...
) {
  type <- match.arg(type)
  if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
  n_samp <- as.integer(n_samp)

  if (!is.null(seed)) set.seed(seed)

  # --- coerce W early (needed by hurdle wrapper + positive sampler) ---
  W0 <- if (requireNamespace("data.table", quietly = TRUE)) {
    data.table::as.data.table(W)
  } else {
    as.data.frame(W)
  }
  n <- nrow(W0)
  if (n < 1L) stop("W must have at least one row.")

  # ---------------------------------------------------------
  # 1) HURDLE WRAPPER (thin): draw H ~ Bern(pi_hat), sample positives only
  #    for rows that ever need positives, then overwrite hurdle draws.
  #    Uses hurdle_runner$logpi() only. No object copying.
  # ---------------------------------------------------------
  if (isTRUE(object$is_hurdle) && !isTRUE(.ignore_hurdle)) {

    hp <- object$hurdle_point
    if (is.null(object$hurdle_runner$logpi) || !is.function(object$hurdle_runner$logpi)) {
      stop("Hurdle runner must define logpi(fit, newdata, ...).")
    }

    clamp01 <- function(p) pmin(pmax(p, 1e-12), 1 - 1e-12)

    # ---- compute pi_hat(W) ----
    if (type == "full") {
      if (is.null(object$hurdle_full_fit)) stop("object$hurdle_full_fit is NULL; cannot sample hurdle type='full'.")
      logpi <- as.numeric(object$hurdle_runner$logpi(
        object$hurdle_full_fit,
        newdata = W0,
        ...
      ))
      if (length(logpi) != n) stop("hurdle_full_fit logpi length mismatch.")
      pi_hat <- clamp01(exp(logpi))
    } else {
      # type == "cv"
      if (is.null(object$hurdle_cv_fit) || !length(object$hurdle_cv_fit)) {
        stop("object$hurdle_cv_fit is NULL/empty; cannot sample hurdle type='cv'.")
      }
      Vh <- length(object$hurdle_cv_fit)
      if (is.null(fold_id)) stop("fold_id must be provided when type='cv'.")
      if (length(fold_id) != n) stop("fold_id must have length nrow(W).")
      if (any(is.na(fold_id)) || any(fold_id < 1L) || any(fold_id > Vh)) {
        stop("fold_id entries must be in 1..length(object$hurdle_cv_fit).")
      }

      # map hurdle fits by fold index (mirror your cv pattern)
      fit_by_fold_h <- vector("list", Vh)
      for (j in seq_along(object$hurdle_cv_fit)) {
        v <- object$hurdle_cv_fit[[j]]$fold
        fit_by_fold_h[[v]] <- object$hurdle_cv_fit[[j]]$fit
      }
      if (any(vapply(fit_by_fold_h, is.null, logical(1)))) {
        stop("hurdle_cv_fit is missing one or more folds; cannot do type='cv' sampling.")
      }

      logpi <- numeric(n)
      for (v in seq_len(Vh)) {
        idx <- which(fold_id == v)
        if (!length(idx)) next
        logpi[idx] <- as.numeric(object$hurdle_runner$logpi(
          fit_by_fold_h[[v]],
          newdata = W0[idx, , drop = FALSE],
          ...
        ))
      }
      pi_hat <- clamp01(exp(logpi))
    }

    # ---- draw hurdle indicators H (TRUE => hurdle draw) ----
    # Shape: n x n_samp logical
    H <- matrix(stats::runif(n * n_samp), nrow = n, ncol = n_samp) < pi_hat

    # rows that ever need a positive draw
    need_pos <- rowSums(!H) > 0L

    # initialize output (we'll fill from positive sampler where needed)
    out <- matrix(hp, nrow = n, ncol = n_samp)

    if (any(need_pos)) {
      # sample positives only for needed rows, using existing sampler logic
      samp_pos <- rsample.dsldensify(
        object,
        W = W0[need_pos, , drop = FALSE],
        n_samp = n_samp,
        type = type,
        fold_id = if (type == "cv") fold_id[need_pos] else NULL,
        seed = NULL,              # avoid resetting RNG inside recursion
        .ignore_hurdle = TRUE,    # key: reuse positive logic without copying object
        ...
      )

      # normalize shape to matrix n_need x n_samp
      if (!is.matrix(samp_pos)) {
        if (n_samp == 1L) {
          samp_pos <- matrix(as.numeric(samp_pos), ncol = 1L)
        } else {
          stop("Positive sampler returned a vector but n_samp > 1.")
        }
      }
      if (nrow(samp_pos) != sum(need_pos) || ncol(samp_pos) != n_samp) {
        stop("Unexpected positive sample dimensions: expected ", sum(need_pos), " x ", n_samp, ".")
      }

      # fill positives, then overwrite hurdle positions back to hp
      out[need_pos, ] <- samp_pos
      out[H] <- hp
    } else {
      # all hurdle draws; out already set to hp
      out[,] <- hp
    }

    return(out)
  }

  # ---------------------------------------------------------
  # 2) POSITIVE (non-hurdle) SAMPLING
  # ---------------------------------------------------------

  runner <- object$runner
  if (is.null(runner) || !is.list(runner) || !is.function(runner$sample)) {
    stop("This dsldensify object does not have a runner with a sample() method.")
  }

  # ---- direct winner: call runner$sample on wide W ----
  if (identical(object$grid_type, "direct")) {
    if (type == "full") {
      if (is.null(object$full_fit)) stop("object$full_fit is NULL; cannot sample type='full'.")
      return(runner$sample(object$full_fit, newdata = W0, n_samp = n_samp, seed = NULL, ...))
    }

    # type == "cv"
    if (is.null(object$cv_fit) || !length(object$cv_fit)) stop("object$cv_fit is NULL/empty; cannot sample type='cv'.")
    V <- length(object$cv_fit)
    if (is.null(fold_id)) stop("fold_id must be provided when type='cv'.")
    if (length(fold_id) != n) stop("fold_id must have length nrow(W).")
    if (any(is.na(fold_id)) || any(fold_id < 1L) || any(fold_id > V)) {
      stop("fold_id entries must be in 1..length(object$cv_fit).")
    }

    out <- matrix(NA_real_, nrow = n, ncol = n_samp)
    for (v in seq_len(V)) {
      idx <- which(fold_id == v)
      if (!length(idx)) next
      fit_v <- object$cv_fit[[v]]$fit
      if (is.null(fit_v)) stop("cv_fit[[", v, "]]$fit is NULL.")
      out[idx, ] <- runner$sample(fit_v, newdata = W0[idx, , drop = FALSE], n_samp = n_samp, seed = NULL, ...)
    }
    return(out)
  }

  # ---- hazard winner: build long hazard grid from object + W, then delegate ----
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required for hazard sampling.")
  }

  if (is.null(object$n_bins) || is.na(object$n_bins) || object$n_bins < 1L) {
    stop("Hazard sampling requires object$n_bins (positive integer).")
  }
  if (is.null(object$breaks) || is.null(object$bin_length)) {
    stop("Hazard sampling requires object$breaks and object$bin_length.")
  }

  n_bins <- as.integer(object$n_bins)
  breaks <- as.numeric(object$breaks)
  bin_length <- as.numeric(object$bin_length)

  if (length(breaks) < 1L) stop("object$breaks must have length >= 1.")
  if (length(bin_length) < 1L) stop("object$bin_length must have length >= 1.")

  # same construction used in predict.dsldensify
  breaks_full <- c(breaks, tail(breaks, 1) + tail(bin_length, 1))

  make_long_grid <- function(W_dt, breaks_full) {
    W_dt <- data.table::as.data.table(W_dt)
    nW <- nrow(W_dt)

    # stable per-row id (runner expects obs_id/bin_id names)
    W_dt[, obs_id := seq_len(nW)]

    # bin table
    bin_dt <- data.table::data.table(
      bin_id = seq_len(length(breaks_full) - 1L),
      bin_lower = breaks_full[-length(breaks_full)],
      bin_upper = breaks_full[-1L]
    )

    # cartesian product obs_id x bin_id (repeat W across bins) via CJ()
    long_dt <- data.table::CJ(
      obs_id = W_dt$obs_id,
      bin_id = bin_dt$bin_id,
      unique = TRUE
    )

    # attach bin endpoints + W covariates
    long_dt <- merge(long_dt, bin_dt, by = "bin_id", all.x = TRUE, sort = FALSE)
    long_dt <- merge(long_dt, W_dt, by = "obs_id", all.x = TRUE, sort = FALSE)

    data.table::setorderv(long_dt, c("obs_id", "bin_id"))
    long_dt
  }

  if (type == "full") {
    if (is.null(object$full_fit)) stop("object$full_fit is NULL; cannot sample type='full'.")

    long_newdata <- make_long_grid(W_dt = W0, breaks_full = breaks_full)
    samp <- runner$sample(object$full_fit, newdata = long_newdata, n_samp = n_samp, seed = NULL, ...)

    # runner returns one row per obs_id (typically as rownames); map back to 1..n
    out <- matrix(NA_real_, nrow = n, ncol = n_samp)
    if (!is.null(rownames(samp))) {
      ridx <- suppressWarnings(as.integer(rownames(samp)))
      if (anyNA(ridx)) stop("Unexpected rownames in runner sample output; expected obs_id integers.")
      out[ridx, ] <- samp
    } else {
      if (nrow(samp) != n) stop("Unexpected sample() output rows: expected nrow(W).")
      out[,] <- samp
    }
    return(out)
  }

  # type == "cv"
  if (is.null(object$cv_fit) || !length(object$cv_fit)) stop("object$cv_fit is NULL/empty; cannot sample type='cv'.")
  V <- length(object$cv_fit)
  if (is.null(fold_id)) stop("fold_id must be provided when type='cv'.")
  if (length(fold_id) != n) stop("fold_id must have length nrow(W).")
  if (any(is.na(fold_id)) || any(fold_id < 1L) || any(fold_id > V)) {
    stop("fold_id entries must be in 1..length(object$cv_fit).")
  }

  out <- matrix(NA_real_, nrow = n, ncol = n_samp)

  # fold-wise sampling to ensure each row uses its fold-specific fit
  for (v in seq_len(V)) {
    idx <- which(fold_id == v)
    if (!length(idx)) next

    fit_v <- object$cv_fit[[v]]$fit
    if (is.null(fit_v)) stop("cv_fit[[", v, "]]$fit is NULL.")

    long_newdata_v <- make_long_grid(W_dt = W0[idx, , drop = FALSE], breaks_full = breaks_full)
    samp_v <- runner$sample(fit_v, newdata = long_newdata_v, n_samp = n_samp, seed = NULL, ...)

    # within-fold, obs_id is 1..length(idx)
    if (!is.null(rownames(samp_v))) {
      local_id <- suppressWarnings(as.integer(rownames(samp_v)))
      if (anyNA(local_id)) stop("Unexpected rownames in fold sample output; expected obs_id integers.")
      out[idx[local_id], ] <- samp_v
    } else {
      if (nrow(samp_v) != length(idx)) stop("Unexpected sample() output rows within fold ", v, ".")
      out[idx, ] <- samp_v
    }
  }

  out
}

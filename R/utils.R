#' Plot conditional or implied marginal density estimates from a dsldensify fit
#'
#' Plots density estimates implied by an object returned by \code{dsldensify()}.
#' The function evaluates \eqn{f(A \mid W)} over a grid of \code{A} values for one
#' or more covariate profiles supplied in \code{W_grid}. Two plotting modes are
#' supported:
#' \enumerate{
#'   \item \code{"conditional"}: plot \eqn{f(A \mid W = w_j)} for each row
#'     \eqn{w_j} of \code{W_grid}.
#'   \item \code{"marginal"}: compute \eqn{f(A \mid W = w_j)} for each row and
#'     plot their average over rows of \code{W_grid}, yielding an implied marginal
#'     density over \code{A} for that empirical distribution of \code{W}.
#' }
#'
#' Predictions are obtained by calling \code{predict.dsldensify()}, either using
#' the stored full-data fit (\code{type = "full"}) or using a single specified
#' cross-validation fold fit (\code{type = "cv"} with \code{cv_fold}).
#'
#' @method plot dsldensify
#' @export
#'
#' @param x Fitted \code{"dsldensify"} object returned by \code{dsldensify()}.
#'
#' @param W_grid A \code{data.frame} or \code{data.table} giving covariate
#'   profiles at which to evaluate the density. Each row is treated as a distinct
#'   covariate profile \eqn{w_j}. Must contain columns matching those used to fit
#'   the selected learner.
#'
#' @param n_A Integer giving the number of grid points used to evaluate \code{A}.
#'
#' @param A_range Optional numeric vector of length 2 specifying the range of
#'   \code{A} values over which to plot. If \code{NULL} and the selected model is
#'   hazard-based, the training bin support is used. Otherwise defaults to
#'   \code{c(-3, 3)}.
#'
#' @param type Character string indicating which fit to use for prediction:
#'   \code{"full"} uses \code{x$full_fit}; \code{"cv"} uses fold-specific fits and
#'   requires \code{cv_fold}.
#'
#' @param cv_fold Integer specifying which cross-validation fold fit to use when
#'   \code{type = "cv"}. For plotting, a single fold is used for all evaluations.
#'
#' @param mode Character string specifying the plotting mode. \code{"conditional"}
#'   plots a curve for each row of \code{W_grid}. \code{"marginal"} plots the
#'   average curve across rows of \code{W_grid}.
#'
#' @param trim_predict Logical passed to \code{predict.dsldensify()} controlling
#'   whether to trim hazard-based predictions outside the training support.
#'
#' @param eps Small positive constant passed to \code{predict.dsldensify()} and,
#'   for direct learners, to the underlying runner when supported.
#'
#' @param xlab Character string giving the x-axis label.
#'
#' @param ylab Character string giving the y-axis label.
#'
#' @param main Optional plot title. If \code{NULL} and \code{add = FALSE}, a
#'   default title is constructed from \code{type} and \code{mode}.
#'
#' @param lty Optional vector of line types used for conditional curves. Recycled
#'   to match \code{nrow(W_grid)}.
#'
#' @param legend Logical indicating whether to draw a legend in conditional mode
#'   when \code{nrow(W_grid) > 1}.
#'
#' @param legend_pos Legend position passed to \code{graphics::legend()}.
#'
#' @param add Logical. If \code{FALSE}, a new plot is created. If \code{TRUE},
#'   curves are added to the current plot using \code{graphics::matlines()} (for
#'   conditional mode) or \code{graphics::lines()} (for marginal mode).
#'
#' @param predict_args Optional named list of additional arguments passed to
#'   \code{predict.dsldensify()}.
#'
#' @param plot_args Optional named list of additional arguments passed to the
#'   underlying plotting function. When \code{mode = "conditional"} and
#'   \code{add = FALSE}, arguments are passed to \code{graphics::matplot()}.
#'   When \code{mode = "marginal"} and \code{add = FALSE}, arguments are passed
#'   to \code{graphics::plot()}.
#'
#' @return Invisibly returns a list with components:
#' \describe{
#'   \item{A}{Numeric vector of grid values at which the density was evaluated.}
#'   \item{dens}{In conditional mode, a matrix of densities with
#'     \code{length(A)} rows and \code{nrow(W_grid)} columns. In marginal mode, a
#'     numeric vector of length \code{length(A)} giving the averaged density.}
#'   \item{dens_by_W}{In marginal mode, the full matrix of per-row conditional
#'     densities used to form the average.}
#' }
#'
#' @examples
#' set.seed(1)
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
#' W_grid <- data.frame(
#'   x1 = c(0, 1),
#'   x2 = c(0, 0)
#' )
#'
#' plot(fit, W_grid, mode = "conditional", type = "full")
#' plot(fit, W_grid, mode = "marginal", type = "full")
#'
#' @seealso \code{\link{dsldensify}}, \code{\link{predict.dsldensify}}

plot.dsldensify <- function(
  x,
  W_grid,
  n_A = 200L,
  A_range = NULL,
  type = c("full", "cv"),
  cv_fold = NULL,
  mode = c("conditional", "marginal"),
  trim_predict = TRUE,
  eps = 1e-12,
  xlab = "A",
  ylab = "Density",
  main = NULL,
  lty = NULL,
  legend = TRUE,
  legend_pos = "topright",
  add = FALSE,
  predict_args = list(),
  plot_args = list()
) {
  stopifnot(requireNamespace("data.table", quietly = TRUE))

  type <- match.arg(type)
  mode <- match.arg(mode)

  # coerce W_grid to data.table
  Wdt <- data.table::as.data.table(W_grid)
  if (nrow(Wdt) < 1L) stop("W_grid must have at least one row.")

  # choose A grid
  if (is.null(A_range)) {
    if (!isTRUE(x$is_direct) && !is.null(x$breaks) && !is.null(x$bin_length)) {
      breaks_full <- c(x$breaks, tail(x$breaks, 1) + tail(x$bin_length, 1))
      A_min <- breaks_full[1L]
      A_max <- breaks_full[length(breaks_full)]
    } else {
      A_min <- -3
      A_max <-  3
    }
  } else {
    if (length(A_range) != 2L || anyNA(A_range)) stop("A_range must be length-2 numeric.")
    A_min <- min(A_range)
    A_max <- max(A_range)
  }

  A_grid <- seq(A_min, A_max, length.out = as.integer(n_A))

  # line types
  if (is.null(lty)) lty <- seq_len(max(1L, nrow(Wdt)))
  lty <- rep(lty, length.out = nrow(Wdt))

  if (is.null(main) && !add) {
    main <- if (mode == "conditional") {
      if (type == "full") "Conditional density estimates"
      else paste0("Conditional density estimates (CV fold ", cv_fold, ")")
    } else {
      if (type == "full") "Implied marginal density (average over W_grid)"
      else paste0("Implied marginal density (CV fold ", cv_fold, ")")
    }
  }
  if (is.null(main)) main <- ""

  # helper: predict density for one W row across A_grid
  pred_for_Wrow <- function(Wrow_dt) {
    Wrep <- Wrow_dt[rep(1L, length(A_grid))]

    if (type == "full") {
      args <- c(list(
        object = x,
        A = A_grid,
        W = Wrep,
        type = "full",
        trim_predict = trim_predict,
        eps = eps
      ), predict_args)

      return(as.numeric(do.call(predict, args)))
    }

    # type == "cv"
    if (is.null(cv_fold)) stop("cv_fold must be provided when type = 'cv'.")
    fold_id <- rep.int(as.integer(cv_fold), length(A_grid))

    args <- c(list(
      object = x,
      A = A_grid,
      W = Wrep,
      type = "cv",
      fold_id = fold_id,
      trim_predict = trim_predict,
      eps = eps
    ), predict_args)

    as.numeric(do.call(predict, args))
  }

  # compute density matrix: length(A_grid) x nrow(Wdt)
  dens_mat <- vapply(seq_len(nrow(Wdt)), function(j) pred_for_Wrow(Wdt[j]),
                     numeric(length(A_grid)))

  if (mode == "conditional") {

    if (!add) {
      matplot_args <- c(list(
        x = A_grid,
        y = dens_mat,
        type = "l",
        lty = lty,
        xlab = xlab,
        ylab = ylab,
        main = main
      ), plot_args)

      do.call(graphics::matplot, matplot_args)
    } else {
      # overlay (do not pass plot_args here; they may contain non-lines args)
      graphics::matlines(x = A_grid, y = dens_mat, lty = lty)
    }

    if (!add && legend && nrow(Wdt) > 1L) {
      leg <- paste0("W[", seq_len(nrow(Wdt)), "]")
      graphics::legend(legend_pos, legend = leg, lty = lty, bty = "n")
    }

    return(invisible(list(A = A_grid, dens = dens_mat)))
  }

  # marginal mode: average over rows of W_grid
  dens_marg <- rowMeans(dens_mat)

  if (!add) {
    plot_args2 <- c(list(
      x = A_grid,
      y = dens_marg,
      type = "l",
      lty = 1,
      xlab = xlab,
      ylab = ylab,
      main = main
    ), plot_args)

    do.call(graphics::plot, plot_args2)
  } else {
    graphics::lines(x = A_grid, y = dens_marg, lty = 1)
  }

  invisible(list(A = A_grid, dens = dens_marg, dens_by_W = dens_mat))
}



#' Print a summary of a dsldensify fit
#'
#' Displays a concise, human-readable summary of a fitted
#' \code{"dsldensify"} object. The printed output includes the selected learner,
#' whether the winning model is hazard-based or direct, the chosen binning
#' configuration (if applicable), the cross-validated risk of the selected
#' model, and whether cross-validated and full-data fits are available for
#' prediction.
#'
#' This method is intended for interactive inspection and debugging rather
#' than programmatic access. All printed information is also available
#' directly as components of the \code{"dsldensify"} object.
#'
#' @method print dsldensify
#' @export
#'
#' @param x An object of class \code{"dsldensify"} returned by
#'   \code{\link{dsldensify}}.
#'
#' @param ... Unused. Included for compatibility with the generic
#'   \code{print()} function.
#'
#' @return Invisibly returns \code{x}.
#'
#' @examples
#' set.seed(1)
#' n <- 50
#' W <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
#' A <- 0.5 * W$x1 - 0.2 * W$x2 + rnorm(n)
#'
#' gaussian_runner <- make_gaussian_homosked_runner(rhs_list = "~ x1 + x2")
#'
#' fit <- dsldensify(
#'   A = A,
#'   W = W,
#'   hazard_learners = NULL,
#'   direct_learners = list(gaussian = gaussian_runner),
#'   cv_folds = 3,
#'   refit_dsl_full_data = TRUE
#' )
#'
#' print(fit)
#'
#' @seealso \code{\link{dsldensify}}, \code{\link{predict.dsldensify}},
#'   \code{\link{plot.dsldensify}}

print.dsldensify <- function(x, ...) {

  cat("\n")
  cat("dsldensify fit\n")
  cat(rep("-", 40), sep = "", "\n")

  # ---- winner summary ----
  cat("Selected learner:\n")
  cat("  • learner   :", x$learner, "\n")

  if (isTRUE(x$is_direct)) {
    cat("  • type      : direct density\n")
  } else {
    cat("  • type      : hazard-based density\n")
    cat("  • grid_type :", x$grid_type, "\n")
    cat("  • n_bins    :", x$n_bins, "\n")
  }

  if (!is.null(x$select_summary$best$cv_risk)) {
    cat("  • CV risk   :", formatC(x$select_summary$best$cv_risk, digits = 4), "\n")
  }

  # ---- tuning details (winner only) ----
  if (!is.null(x$tune_row) && nrow(x$tune_row) == 1L) {
    cat("\nTuning parameters (selected):\n")
    tr <- x$tune_row
    tr <- tr[, setdiff(names(tr), c(".tune")), drop = FALSE]

    for (nm in names(tr)) {
      cat("  •", nm, ":", tr[[nm]], "\n")
    }
  }

  # ---- CV / refit status ----
  cat("\nFitting details:\n")
  cat("  • CV folds stored :", !is.null(x$cv_fit), "\n")
  cat("  • Full refit      :", !is.null(x$full_fit), "\n")

  # ---- model class info ----
  cat("\nPrediction:\n")
  cat("  • predict(type = \"full\") available :", !is.null(x$full_fit), "\n")
  cat("  • predict(type = \"cv\")   available :", !is.null(x$cv_fit), "\n")

  cat(rep("-", 40), sep = "", "\n")
  invisible(x)
}



#' Create long-format discrete-time hazard data for density estimation
#'
#' @description
#' Converts observation-level data \code{(A_i, W_i, wt_i)} into a long-format
#' representation suitable for fitting a discrete-time hazard model.
#' For each observation \code{i}, the function creates one row for each bin
#' \code{j = 1, ..., bin_id[i]}, where \code{bin_id[i]} is the bin into which 
#' \code{A_i} falls. The function defines an indicator \code{in_bin} that equals
#' 1 on the terminal bin and 0 on all prior bins. This supports a
#' discrete-time hazard regression for \code{P(A in bin_j | A >= left_j, W)} and
#' subsequent mapping of hazards to bin masses (and, with bin-width scaling, to
#' a continuous density approximation).
#'
#' @details
#' ## Grid/bins
#' The binning grid may be provided directly via \code{breaks} or constructed
#' from \code{A} using \code{n_bins} and \code{grid_type}:
#'
#' * If \code{breaks} is provided, it must be a numeric vector specifying the full bin 
#'   endpoints. Observation bin membership is computed via \code{findInterval(A, 
#'   breaks, rightmost.closed = TRUE, all.inside = TRUE)}. Values of \code{A} outside 
#'   the range of \code{breaks} are clipped to the boundary bins due to \code{all.inside = TRUE}.
#'
#' * If \code{n_bins} is provided, the function constructs \code{J = n_bins}
#'   bins over the observed range of \code{A}. For \code{grid_type = "equal_range"}, bins are
#'   equally spaced over \code{[min(A), max(A)]}. For \code{grid_type = "equal_mass"}, bin
#'   endpoints are empirical quantiles of \code{A} using \code{quantile_type}.
#'   When ties in \code{A} cause duplicated quantile breakpoints, duplicates are
#'   dropped and the effective number of bins may be smaller than requested.
#'
#' If using \code{n_bins}, the function returns:
#' \itemize{
#'   \item \code{breaks}: the left endpoints
#'   \item \code{bin_length}: the bin widths
#' }
#'
#' ## Long-format construction
#' Let \code{bin_id[i]} be the terminal bin index for observation \code{i}. The
#' output data has \code{sum_i bin_id[i]} rows, with:
#' \itemize{
#'   \item \code{obs_id}: original observation index \code{i}
#'   \item \code{bin_id}: bin index \code{j} running from 1 to \code{bin_id[i]}
#'   \item \code{in_bin}: indicator equal to 1 if \code{j == bin_id[i]}, else 0
#'   \item Baseline covariates \code{W} repeated across the long rows for \code{i}
#'   \item \code{wts}: observation-level weights repeated across the long rows
#' }
#'
#' ## Relationship to densities and bin widths
#' Downstream, the fitted hazard model can be mapped to a bin mass
#' \code{P(A in bin_j | W)} for each observation. To interpret this as an
#' approximation to a continuous density at the observed \code{A_i}, one divides
#' the terminal-bin mass by the terminal-bin width.
#'
#' @param A Numeric vector representing outcome/exposure being discretized into bins.
#' @param W Baseline covariates. Either a vector of length \code{n} or a
#'   2D object (matrix/data.frame). If \code{W} has no column
#'   names, columns are named \code{W_1, W_2, ...}.
#' @param wts Numeric vector of length \code{n} giving non-negative observation-level weights.
#' @param grid_type Character. One of \code{"equal_range"} or \code{"equal_mass"}.
#'   Only used when \code{n_bins} is provided.
#' @param n_bins Integer number of bins (used when \code{breaks} is \code{NULL}).
#' @param breaks Optional numeric vector of full bin endpoints of length. If provided, \code{n_bins} 
#'   is ignored and bins are defined by these endpoints.
#' @param quantile_type Integer passed to \code{stats::quantile(type = ...)} when
#'   \code{grid_type = "equal_mass"}.
#'
#' @return A list with components:
#' \describe{
#'   \item{data}{A \code{data.table} in long format with columns \code{obs_id},
#'     \code{in_bin}, \code{bin_id}, covariates from \code{W}, and \code{wts}.}
#'   \item{breaks}{Left endpoints of bins (numeric vector length \code{J}) when
#'     \code{n_bins} is used; \code{NULL} when \code{breaks} is supplied.}
#'   \item{bin_length}{Bin widths (numeric vector length \code{J}) when
#'     \code{n_bins} is used; \code{NULL} when \code{breaks} is supplied.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 10
#' A <- rexp(n)
#' W <- data.frame(w1 = rnorm(n), w2 = rbinom(n, 1, 0.5))
#'
#' # Equal-width bins
#' out1 <- format_long_hazards(A, W, n_bins = 10, grid_type = "equal_range")
#'
#' # Equal-mass (quantile) bins
#' out2 <- format_long_hazards(A, W, n_bins = 10, grid_type = "equal_mass")
#'
#' # Recreate the same endpoints using breaks
#' breaks_full <- c(out1$breaks, tail(out1$breaks, 1) + tail(out1$bin_length, 1))
#' out3 <- format_long_hazards(A, W, breaks = breaks_full)
#' }
#'
#' @export

format_long_hazards <- function(
  A, W, wts = rep(1, length(A)),
  grid_type = c("equal_range", "equal_mass"),
  n_bins = NULL,
  breaks = NULL,
  quantile_type = 8L
) {
  grid_type <- match.arg(grid_type)
  n <- length(A)
  if (length(wts) != n) stop("wts must have length(A).")

  if (is.null(dim(W))) {
    Wdt <- data.table::data.table(W = W)
  } else {
    Wdt <- data.table::as.data.table(W)
    if (is.null(colnames(Wdt))) {
      data.table::setnames(Wdt, paste0("W_", seq_len(ncol(Wdt))))
    }
  }

  breaks_left <- NULL
  bin_length  <- NULL

  if (!is.null(breaks)) {
    breaks_left  <- breaks[-length(breaks)]
    breaks_right <- breaks[-1L]
    bin_length   <- breaks_right - breaks_left
    bin_id <- findInterval(A, breaks, rightmost.closed = TRUE, all.inside = TRUE)
    J <- length(breaks) - 1L
  } else if (!is.null(n_bins)) {
    a_min <- min(A, na.rm = TRUE)
    a_max <- max(A, na.rm = TRUE)

    if (grid_type == "equal_range") {
      brks <- seq(a_min, a_max, length.out = n_bins + 1L)
    } else {
      probs <- seq(0, 1, length.out = n_bins + 1L)
      brks <- as.numeric(
        stats::quantile(A, probs = probs, type = quantile_type, na.rm = TRUE)
      )
      # ensure nondecreasing and handle potential duplicates
      brks <- cummax(brks)
      if (anyDuplicated(brks)) {
        brks <- unique(brks)
        if (length(brks) < 2L) stop("Quantile breaks collapsed; A has too few unique values.")
        n_bins <- length(brks) - 1L
      }
    }

    breaks_left <- brks[-length(brks)]
    breaks_right <- brks[-1L]
    bin_length <- breaks_right - breaks_left

    bin_id <- findInterval(A, brks, rightmost.closed = TRUE, all.inside = TRUE)
    J <- n_bins
  } else {
    stop("Combination of arguments `breaks`, `n_bins` incorrectly specified.")
  }

  counts <- bin_id  # number of long rows per obs
  obs_id_long <- rep.int(seq_len(n), counts)
  bin_id_long <- sequence(counts)

  # indicator for terminal bin (failure bin)
  bin_id_rep <- rep.int(bin_id, counts)
  in_bin <- as.integer(bin_id_long == bin_id_rep)

  long_dt <- data.table::data.table(
    obs_id = obs_id_long,
    in_bin = in_bin,
    bin_id = bin_id_long
  )

  # attach W and weights by row-number lookup
  long_dt <- cbind(long_dt, Wdt[obs_id_long])
  long_dt[, wts := wts[obs_id_long]]

  # key for fast downstream ops
  data.table::setkey(long_dt, obs_id, bin_id)

  list(
    data = long_dt,
    breaks = breaks_left,
    bin_length = bin_length
  )
}

#' Select a single tuning value from a fitted learner bundle
#'
#' @description
#' Extracts the fit corresponding to a chosen tuning index from a learner-specific
#' fit bundle. This function provides a generic mechanism for reducing a
#' multi-tuning fit object (e.g., multiple formulas, lambdas, or hyperparameter
#' settings) to the single fit (e.g., selected by cross-validation).
#'
#' The behavior is learner-dependent:
#' \itemize{
#'   \item If the learner runner defines a \code{select_fit()} method, that method
#'   is used to extract the selected tuning in a learner-specific way.
#'   \item Otherwise, a default rule is applied assuming \code{fit_bundle$fits} is
#'   a list of fits indexed by tuning parameter. In this case, only the element
#'   corresponding to \code{tune} is retained.
#'   \item If neither condition applies, the fit bundle is returned unchanged,
#'   which is appropriate when the learner has no tuning parameter or when tuning
#'   selection is irrelevant.
#' }
#'
#' @details
#' This helper is intended to be used after model selection, for example when:
#' \itemize{
#'   \item Storing only the selected fit from each cross-validation fold
#'   \item Refitting the selected model on the full data and discarding unused
#'     tuning configurations
#' }
#'
#' By delegating to \code{runner$select_fit()} when available, this function allows
#' different learners (e.g., \code{glm}, \code{glmnet}, tree-based models) to store
#' and reduce their fitted objects in a way that is natural for the underlying
#' fitting procedure, while keeping the higher-level pipeline generic.
#'
#' @param runner A learner runner object. If it defines a function
#'   \code{runner$select_fit}, that function is used to perform the tuning
#'   selection.
#' @param fit_bundle An object returned by \code{runner$fit()}, potentially
#'   containing fits for multiple tuning values.
#' @param tune Integer index of the selected tuning configuration (1-based).
#'
#' @return
#' A fit bundle corresponding to the selected tuning configuration. The returned
#' object is suitable for passing directly to \code{runner$predict()}.
#'
#' @examples
#' \dontrun{
#' # Default behavior with a list-of-fits bundle
#' fit_bundle <- list(fits = list(fit1, fit2, fit3))
#' fit_sel <- select_fit_tune(runner, fit_bundle, tune = 2)
#' # fit_sel$fits contains only fit2
#'
#' # Learner-specific selection
#' runner$select_fit <- function(fit_bundle, tune) {
#'   fit_bundle$lambda <- fit_bundle$lambda[tune]
#'   fit_bundle
#' }
#' fit_sel <- select_fit_tune(runner, fit_bundle, tune = 5)
#' }
#'
#' @export
select_fit_tune <- function(runner, fit_bundle, tune) {
  if (!is.null(runner$select_fit) && is.function(runner$select_fit)) {
    return(runner$select_fit(fit_bundle, tune))
  }
  # default for "fits is a list" pattern
  if (!is.null(fit_bundle$fits) && length(fit_bundle$fits) >= tune) {
    fit_bundle$fits <- fit_bundle$fits[tune]
    return(fit_bundle)
  }
  # otherwise, just return as-is (works if tune is irrelevant)
  fit_bundle
}




summarize_and_select <- function(
  select_out,
  hazard_learners = NULL,
  direct_learners = NULL,
  weighted = TRUE
) {
  stopifnot(requireNamespace("data.table", quietly = TRUE))
  DT <- data.table::data.table

  hazard_names <- if (!is.null(hazard_learners)) names(hazard_learners) else character(0)
  direct_names <- if (!is.null(direct_learners)) names(direct_learners) else character(0)

  learners <- c(
    if (!is.null(hazard_learners)) hazard_learners else list(),
    if (!is.null(direct_learners)) direct_learners else list()
  )

  is_direct_learner <- function(nm) nm %in% direct_names

  # ---- fold-level long table (NO grid_idx) ----
  fold_dt <- data.table::rbindlist(lapply(seq_along(select_out), function(g) {
    gs <- select_out[[g]]

    data.table::rbindlist(lapply(gs$cv_out, function(fold_obj) {
      fw <- fold_obj$fold_weight
      v  <- fold_obj$fold

      data.table::rbindlist(lapply(fold_obj$learners, function(L) {
        loss <- L$loss
        loss_vec <- if (is.matrix(loss)) colMeans(loss) else as.numeric(loss)
        K <- length(loss_vec)

        DT(
          grid_type   = gs$grid_type,   # hazard: "equal_*"; direct: "direct"
          n_bins      = gs$n_bins,      # hazard meaningful; direct likely NA
          fold        = v,
          fold_weight = fw,
          learner     = L$learner,
          .tune       = seq_len(K),
          loss        = loss_vec
        )
      }))
    }))
  }))

  # ---- aggregate to CV risk ----
  key_cols <- c("grid_type", "n_bins", "learner", ".tune")

  cv_dt <- if (weighted) {
    fold_dt[, .(
      cv_risk = sum(fold_weight * loss) / sum(fold_weight),
      V = .N
    ), by = key_cols]
  } else {
    fold_dt[, .(
      cv_risk = mean(loss),
      V = .N
    ), by = key_cols]
  }

  # summary_all: minimal across everyone
  summary_all <- cv_dt[, .(grid_type, n_bins, learner, .tune, cv_risk, V)]
  data.table::setorder(summary_all, cv_risk)

  # ---- tune grid lookup per learner ----
  get_tune_dt <- function(nm) {
    runner <- learners[[nm]]
    tg <- runner$tune_grid
    if (is.null(tg) || nrow(tg) < 1L) return(DT(learner = nm, .tune = 1L))
    tg <- data.table::as.data.table(tg)
    if (!(".tune" %in% names(tg))) tg[, .tune := seq_len(.N)]
    tg[, learner := nm]
    tg
  }

  # ---- summary_by_learner: merge in tuning cols; direct drops n_bins ----
  summary_by_learner <- lapply(split(cv_dt, by = "learner", keep.by = TRUE), function(dtL) {
    nm <- unique(dtL$learner)
    out <- merge(dtL, get_tune_dt(nm), by = c("learner", ".tune"), all.x = TRUE, sort = FALSE)

    if (is_direct_learner(nm)) {
      out[, grid_type := "direct"]
      out[, n_bins := NULL]
    }

    data.table::setorder(out, cv_risk)
    out
  })

  # ---- best: pick from summary_all, then pull from learner-specific table ----
  best_key <- summary_all[1, .(learner, .tune, grid_type, n_bins)]
  best_tbl <- summary_by_learner[[best_key$learner]]

  if (is.null(best_tbl)) stop("Internal error: best learner missing from summary_by_learner.")

  if ("n_bins" %in% names(best_tbl)) {
    best <- best_tbl[
      .tune == best_key$.tune & grid_type == best_key$grid_type & n_bins == best_key$n_bins
    ][1]
  } else {
    best <- best_tbl[
      .tune == best_key$.tune & grid_type == "direct"
    ][1]
  }

  if (nrow(best) != 1L) {
    best <- best_tbl[.tune == best_key$.tune & grid_type == best_key$grid_type][1]
  }

  list(
    summary_all = summary_all,
    summary_by_learner = summary_by_learner,
    best = best
  )
}




#' Weighted negative log loss for discretized mass estimates
#'
#' @description
#' Computes a weighted negative log loss for each column of a mass matrix,
#' with a bin-width adjustment. Intended for evaluating discretized density 
#' or probability mass estimates.
#'
#' @param mass_mat Numeric matrix of nonnegative mass or density values.
#'   Each column is assumed to be mass evaluated at a different set of tuning parameters.
#' @param bw Numeric vector of bin widths. Recycled across columns of \code{mass_mat} as needed.
#' @param w Numeric vector of nonnegative observation weights, with
#'   \code{length(w) == nrow(mass_mat)}.
#' @param eps Small positive constant used to floor \code{mass_mat} values before
#'   taking logs, to avoid \code{-Inf}.
#'
#' @return Numeric vector of length \code{ncol(mass_mat)} giving the weighted
#'   negative log loss for each column.
#'
#' @details
#' The loss is computed as
#' \deqn{
#' \frac{1}{\sum_i w_i} \sum_i w_i \left\{ -\log(m_{ij}) + \log(bw_i) \right\},
#' }
#' where `m_{ij}` is the mass in row `i`, column `j`. Values of `mass_mat`
#' smaller than `eps` are truncated to `eps`.
#'
#' @export
neg_logloss <- function(mass_mat, bw, w, eps = 1e-15) {
  denom <- sum(w)
  log_mass <- log(pmax(mass_mat, eps))
  log_bw <- log(bw) # bw recycled across columns as needed
  colSums((-log_mass + log_bw) * w) / denom
}

#' Convert hazard predictions to bin mass
#'
#' @description
#' Aggregates predicted hazards across discrete intervals to produce
#' per-bin probability mass values. 
#'
#' @param preds Numeric vector or matrix of predicted hazards. Rows correspond
#'   to discrete intervals; columns correspond to different prediction sets.
#'   A vector is treated as a single-column matrix.
#' @param grp Integer or factor vector defining observation group membership
#'   for each row of \code{preds}. Rows with the same group are aggregated together.
#' @param is_last Logical vector indicating whether a row corresponds to the
#'   terminal interval for its group. Must have length \code{nrow(preds)}.
#' @param eps Small positive constant used to floor probabilities before
#'   taking logs, to avoid \code{-Inf}.
#'
#' @return Numeric matrix with one row per unique value of \code{grp} and one column
#'   per column of \code{preds}, giving the probability mass for each
#'   observation.
#'
#' @export

hazards_to_mass_by_obs <- function(preds, grp, is_last, eps = 1e-15) {
  if (is.null(dim(preds))) preds <- matrix(preds, ncol = 1L)

  log_terms <- matrix(NA_real_, nrow(preds), ncol(preds))

  log_terms[is_last, ]  <- log(pmax(preds[is_last, , drop = FALSE], eps))
  log_terms[!is_last, ] <- log(pmax(1 - preds[!is_last, , drop = FALSE], eps))

  log_mass <- rowsum(log_terms, grp, reorder = FALSE)
  exp(log_mass)
}

#' Run cross-validation routine for a single discretization (grid) setting
#'
#' @description
#' Fits and evaluates a set of hazard-based learners under a fixed discretization
#' scheme (defined by \code{grid_type} and \code{n_bins}) using cross-validation. For each
#' fold, learners are trained on long-format hazard data, predicted on validation
#' data, converted to per-observation mass, and scored via negative log loss.
#'
#' @param grid_type Character string specifying the discretization strategy
#'   (e.g., \code{"equal_range"} or \code{"equal_mass"}).
#' @param n_bins Integer number of bins used for discretization.
#' @param A Numeric vector of exposure or time variable.
#' @param W Matrix or data frame of covariates.
#' @param wts Numeric vector of observation weights.
#' @param cv_folds_id List of cross-validation fold objects. Each element must
#'   contain a `validation_set` of observation indices.
#' @param id_fold Integer vector mapping observation IDs to fold numbers.
#' @param learners Named list of learner runner objects. Each runner must implement
#'   \code{fit(train_set)} and \code{predict(fit, newdata)} methods, and may include a
#'   \code{tune_grid}.
#' @param return_fits Logical; if \code{TRUE}, fitted learner objects are returned.
#' @param return_density Logical; if \code{TRUE}, estimated densities (mass / bin width)
#'   are returned for validation observations.
#' @param ... Additional arguments (currently unused).
#'
#' @return A list with components:
#'   \describe{
#'     \item{cv_out}{A list of length \code{V} (number of folds). Each element contains
#'       fold-level results including \code{fold_weight} and per-learner loss vectors.}
#'     \item{breaks}{Vector of bin breakpoints used for discretization.}
#'     \item{bin_length}{Vector of bin widths corresponding to \code{breaks}.}
#'     \item{grid_type}{The discretization type used.}
#'     \item{n_bins}{Number of bins used.}
#'   }
#'
#' @details
#' Internally, the function:
#' \enumerate{
#'   \item Converts \code{(A, W)} into long-format hazard data via \code{format_long_hazards()}.
#'   \item Splits the long data by fold while preserving observation-level grouping.
#'   \item Fits each learner on training hazards and predicts hazards on validation data.
#'   \item Aggregates hazards to per-observation mass using
#'   \code{hazards_to_mass_by_obs()}.
#'   \item Computes weighted negative log loss using \code{neg_logloss()}.
#' }
#'
#' @export
run_grid_setting <- function(grid_type, n_bins,
                             A, W, wts,
                             cv_folds_id, id_fold,
                             learners,   # named list of learner runners
                             return_fits = TRUE,
                             return_density = FALSE,
                             ...) {

  long_hazards_data  <- format_long_hazards(A = A, W = W, wts = wts, n_bins = n_bins, grid_type = grid_type)
  long_data <- long_hazards_data$data

  fold_id_long <- id_fold[ long_data$obs_id ]  # depends on n_bins only through obs_id repetition
  grp_long <- data.table::rleid(long_data$obs_id)
  is_last_long <- c(grp_long[-1L] != grp_long[-length(grp_long)], TRUE)

  bin_last_all <- long_data$bin_id[is_last_long]
  bw_obs_all   <- long_hazards_data$bin_length[bin_last_all]  

  V <- length(cv_folds_id)

  grid_setting_cv_out <- lapply(seq_len(V), function(v) {
    train_set <- long_data[fold_id_long != v]
    valid_set <- long_data[fold_id_long == v]

    # fold-specific grouping data for density mapping
    grp_valid <- grp_long[fold_id_long == v]
    is_last_valid <- is_last_long[fold_id_long == v]

    valid_obs_id <- valid_set$obs_id
    valid_ids <- cv_folds_id[[v]]$validation_set
    wts_valid_v <- wts[valid_ids]
    fold_weight <- sum(wts_valid_v)
    
    bw_valid_v <- bw_obs_all[valid_ids]

    fold_learners_out <- setNames(lapply(names(learners), function(learner_name) {
      runner <- learners[[learner_name]]

      fit_bundle <- runner$fit(train_set = train_set)

      pred_mat <- runner$predict(fit_bundle, newdata = valid_set)

      mass_mat <- hazards_to_mass_by_obs(
        preds   = pred_mat,
        grp     = grp_valid,
        is_last = is_last_valid
      )
      loss_vec <- neg_logloss(mass_mat, bw = bw_valid_v, w = wts_valid_v)

      learner_out <- list(
        learner = learner_name,
        fold = v,
        loss = loss_vec
      )

      if (return_density) learner_out$dens <- mass_mat / bw_valid_v
      if (return_fits) learner_out$fit <- fit_bundle
      return(learner_out)
    }),
    names(learners))

    fold_out <- list(
      fold = v,
      fold_weight = fold_weight,
      learners = fold_learners_out
    )
    return(fold_out)
  })

  grid_setting_out <- list(
    cv_out = grid_setting_cv_out,
    breaks = long_hazards_data$breaks,
    bin_length = long_hazards_data$bin_length,
    grid_type = grid_type,
    n_bins = n_bins
  )
  
  return(grid_setting_out)
}

#' Evaluate direct density learners for a single cross-validation setting
#'
#' Runs cross-validation for a collection of direct conditional density learners
#' (runners) on wide-format data. Each learner is trained on the training split
#' for each fold and evaluated on the validation split using negative
#' log-density loss, \eqn{-\log f(A \mid W)}.
#'
#' This function is used internally by \code{\link{dsldensify}} to evaluate
#' direct density learners once (no binning grid), producing fold-level outputs
#' in a standardized format compatible with \code{summarize_and_select()}.
#'
#' @param A Numeric vector of length \code{n} containing observed outcomes.
#'
#' @param W Covariates used to condition the density. May be a vector, matrix,
#'   \code{data.frame}, or \code{data.table}.
#'
#' @param wts Numeric vector of length \code{n} giving observation weights.
#'
#' @param cv_folds_id Fold object as returned by \code{origami::make_folds()}.
#'   Must have length \code{V}, with each element containing a
#'   \code{validation_set} index vector.
#'
#' @param id_fold Integer vector of length \code{n} giving the fold assignment
#'   for each observation, with values in \code{1, ..., V}. Typically constructed
#'   from \code{cv_folds_id}.
#'
#' @param learners Named list of direct density runners. Each runner must define
#'   a \code{fit(train_set, ...)} method and a
#'   \code{log_density(fit_bundle, newdata, ...)} method. The \code{fit} method
#'   is called on the wide training data, and \code{log_density} is called on the
#'   wide validation data.
#'
#' @param return_fits Logical; whether to store each fitted learner object for
#'   each fold in the returned structure. Storing fits can be memory-intensive.
#'
#' @param return_density Logical; whether to additionally store the validation
#'   density values \eqn{f(A \mid W)} (obtained by exponentiating the log-density)
#'   for each fold and learner.
#'
#' @param eps Small positive constant passed to \code{runner$log_density()} to
#'   bound log-density evaluations away from \code{-Inf} when supported.
#'
#' @param ... Additional arguments passed to \code{runner$fit()} and
#'   \code{runner$log_density()}.
#'
#' @return A named list with components:
#' \describe{
#'   \item{cv_out}{List of length \code{V}. Each element contains the fold index,
#'     a fold weight (sum of validation weights), and a named list of learner
#'     results. Each learner result contains a loss matrix of dimension
#'     \code{n_valid} by \code{K}, where \code{K} is the number of tuning
#'     configurations for that learner (or 1 if untuned).}
#'   \item{breaks}{\code{NULL}. Included for compatibility with hazard-based
#'     outputs.}
#'   \item{bin_length}{\code{NULL}. Included for compatibility with hazard-based
#'     outputs.}
#'   \item{grid_type}{Character string \code{"direct"}.}
#'   \item{n_bins}{\code{NA_integer_}. Included for compatibility with hazard-based
#'     outputs.}
#' }

run_direct_setting <- function(
  A, W, wts,
  cv_folds_id, id_fold,
  learners,
  return_fits = TRUE,
  return_density = FALSE,
  eps = 1e-15,
  ...
) {
  V <- length(cv_folds_id)

  W_dt <- data.table::as.data.table(W)
  wide_dt <- data.table::data.table(A = A)
  wide_dt <- cbind(wide_dt, W_dt)
  wide_dt[, wts := wts]

  cv_out <- lapply(seq_len(V), function(v) {

    train_wide <- wide_dt[id_fold != v]
    valid_wide <- wide_dt[id_fold == v]

    valid_ids <- cv_folds_id[[v]]$validation_set
    wts_valid_v <- wts[valid_ids]
    fold_weight <- sum(wts_valid_v)

    fold_learners_out <- setNames(lapply(names(learners), function(learner_name) {
      runner <- learners[[learner_name]]
      stopifnot(is.function(runner$fit), is.function(runner$log_density))

      fit_bundle <- runner$fit(train_set = train_wide, ...)

      # n_valid x K (K tunes). We'll compute per-obs loss for each tune, and store as matrix.
      logf_mat <- runner$log_density(fit_bundle, newdata = valid_wide, eps = eps, ...)
      if (is.null(dim(logf_mat))) logf_mat <- matrix(logf_mat, ncol = 1L)

      # loss per obs per tune: -log f(A|W)
      loss_mat <- -logf_mat

      learner_out <- list(
        learner = learner_name,
        fold = v,
        loss = loss_mat
      )
      if (return_density) learner_out$dens <- exp(logf_mat)
      if (return_fits) learner_out$fit <- fit_bundle
      learner_out
    }), names(learners))

    list(
      fold = v,
      fold_weight = fold_weight,
      learners = fold_learners_out
    )
  })

  list(
    cv_out = cv_out,
    breaks = NULL,
    bin_length = NULL,
    grid_type = "direct",
    n_bins = NA_integer_
  )
}

strip_glm <- function(model) {
  model$env <- NULL
  model$y = c()
  model$model = c()

  model$residuals = c()
  model$fitted.values = c()
  model$effects = c()
  model$linear.predictors = c()
  model$weights = c()
  model$prior.weights = c()
  model$data = c()

  model$family$variance = c()
  model$family$dev.resids = c()
  model$family$aic = c()
  model$family$validmu = c()
  model$family$simulate = c()

  attr(model$terms,".Environment") = c()
  attr(model$formula,".Environment") = c()
  model$qr = c()

  class(model) <- "strip_glm"
  return(model)
}

#' @export 
predict.strip_glm <- function(object, newdata, ...){
  coefs <- object$coefficients
  family <- object$family
  linkinv <- family$linkinv
  
  terms_obj <- delete.response(terms(object))
  model_vars <- all.vars(terms_obj)

  if (inherits(newdata, "data.table")) {
    newdata_filtered <- newdata[, intersect(names(newdata), model_vars), with = FALSE]
  } else {
    newdata_filtered <- newdata[, intersect(names(newdata), model_vars), drop = FALSE]
  }
  
  mf <- model.frame(terms_obj, data = newdata_filtered, xlev = object$xlevels)
  X <- model.matrix(terms_obj, data = mf)
  
  # Ensure compatibility between model coefficients and new design matrix
  missing_coefs <- setdiff(names(coefs), colnames(X))
  if (length(missing_coefs) > 0) {
    stop("Missing variables in newdata: ", paste(missing_coefs, collapse = ", "))
  }
  
  eta <- X %*% coefs
  predictions <- if (!is.null(linkinv)) linkinv(eta) else eta  # Identity link fallback
  
  return(as.numeric(predictions))
}


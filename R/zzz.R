.onAttach <- function(...) {
  packageStartupMessage(paste0(
    "dsldensify v", utils::packageDescription("dsldensify")$Version,
    ": ", utils::packageDescription("dsldensify")$Title
  ))
}

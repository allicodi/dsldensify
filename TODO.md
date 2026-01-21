# `dsldensify` development notes

- [ ] Ensure all eps defaults to same value
- [ ] Develop unit tests for functions
- [ ] Add warnings if selected hazard model is at edge of n_bins range?
- [ ] Further validate quantile regression methods via code review
- [ ] Further validate mixture model wrappers via code review
- [ ] Add quantile to output summary description (currently will list it as direct learner)

- [ ] Fix plot_args for the plotting functions
- [ ] Consider adding an option for plotting CDF



## For hurdle model

- [ ] Add more positive only runners
- [ ] Run through comments and edit/remove
- [X] Add docs for new utils.R functions
- [X] Update docs for predict and rsample methods
- [X] Update docs for dsldensify
- [X] Test predict and sample code with hurdles
- [X] Add hurdle runners using glmnet, rf, xgboost
- [X] Update plot methods for hurdles
- [X] Add positive_support flag to runners
- [X] Test hurdle runners using glmnet, rf, xgboost
- [X] Double check that tuning grids and model fits line up in select_X methods







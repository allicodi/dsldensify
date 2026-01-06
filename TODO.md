# `dsldensify` development notes

- [X] Test xgboost runner
- [X] Test random forest runner
- [X] Test run with all hazard-based runners together
- [X] Reformat fit_all$select_summary to not add NAs to tuning parameters that are not there
- [ ] Ensure all eps defaults to same value
- [X] Test gaussian homosket runner
- [X] Test hazard and non-hazard based runners together
- [X] Test gamlss runner
- [ ] See about modifying gamlss and other runner to have truncated range?
- [ ] Develop unit tests for functions
- [X] Plot methods for main function
- [X] Print methods for main function
- [ ] Add warnings if selected hazard model is at edge of n_bins range?
- [ ] Further validate quantile regression methods via code review
- [ ] Further validate mixture model wrappers via code review
- [ ] Add quantile to output summary description (currently will list it as direct learner)
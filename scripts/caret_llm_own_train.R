train_df <- churn_train_list[[df_name]]
test_df <- churn_test_list[[df_name]]
set.seed(1)
folds <- createMultiFolds(train_df$churn, k = 2, times = 2)
tune_grid <- expand.grid(
  threshold_pruning = c(.01, .15),
  nbr_obs_leaf = (nrow(train_df) * c(.25, .5)) %>% round()
)
folds_results <- data.frame()
for (params_set in tune_grid %>% split(1:nrow(.)) ){
  params_set_aucs <- numeric()
  for(fold in names(folds)){
    .train_indx <- folds[[fold]]
    .train_df <- train_df[.train_indx, ]
    .test_df <- train_df[-.train_indx, ]
    set.seed(1)
    .llm_fit <- LLM::llm(
      X = .train_df %>% select(-churn), 
      Y = .train_df$churn, 
      threshold_pruning = params_set$threshold_pruning, 
      nbr_obs_leaf = params_set$nbr_obs_leaf
    )
    
    .true_response <- (.test_df$churn == "Churn") * 1
    .predicted <- LLM::predict.llm(.llm_fit, .test_df)$probability
    params_set_aucs <- c(
      params_set_aucs,
      pROC::roc(predictor = .predicted, response = .true_response)$auc
    )
  }
  mean_auc <- mean(params_set_aucs)
  folds_results <- rbind(
    folds_results,
    data.frame(
      threshold_pruning = params_set$threshold_pruning,
      nbr_obs_leaf = params_set$nbr_obs_leaf,
      auc = mean_auc
    )
  )
  print(folds_results)
}
best_params <- 
  folds_results %>% 
  filter(auc == max(auc)) %>% 
  select(-auc) %>% 
  summarise_all(max)

#training model with final parameters
set.seed(1)
model_obj <- LLM::llm(
  X = train_df %>% select(-churn), 
  Y = train_df$churn, 
  threshold_pruning = best_params$threshold_pruning, 
  nbr_obs_leaf = best_params$nbr_obs_leaf
)
true_response <- (test_df$churn == "Churn") * 1
predicted <- LLM::predict.llm(model_obj, test_df)$probability
auc <- pROC::roc(predictor = predicted, response = true_response)$auc
td_lift <- lift::TopDecileLift(predicted, true_response)
mp_result <- EMP::empChurn(
  predicted, 
  true_response, 
  clv = CLV,
  d = ret_offer_cost,
  f = contact_cost
)
mp <- mp_result$MP
mp_frac <- mp_result$MPfrac

results_llm$model[[df_name]] <- model_obj
results_llm$auc[[df_name]] <- auc
results_llm$td_lift[[df_name]] <- td_lift
results_llm$mp[[df_name]] <- mp
results_llm$mp_frac[[df_name]] <- mp_frac 



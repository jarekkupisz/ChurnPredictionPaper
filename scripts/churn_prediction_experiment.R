# necessary to soure first as it loads all datasets and required packages
source("churn_prediction_data_preparation.R")
require(pROC)
require(LLM)
require(lift)
require(EMP)


# parameters for Maximum Profir Criterion
# alpha and beta stays as the industry standard
CLV <- 7000
ret_offer_cost <- 600
contact_cost <- 30

# logistic regression 
results_lr <- list()
for(df_name in names(churn_train_list)) {
  train_control <- trainControl(classProbs = T)
  train_df <- churn_train_list[[df_name]]
  test_df <- churn_test_list[[df_name]]
  
  set.seed(1)
  model_obj <- train(
    churn ~ .,  
    data = train_df, 
    method = "glm", 
    family = "binomial",
    trControl = train_control
  )
  
  predicted <- predict(model_obj, newdata = test_df, type = "prob")$Churn
  true_response <- (test_df$churn == "Churn") * 1
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
  
  results_lr$model[[df_name]] <- model_obj
  results_lr$auc[[df_name]] <- auc
  results_lr$td_lift[[df_name]] <- td_lift
  results_lr$mp[[df_name]] <- mp
  results_lr$mp_frac[[df_name]] <- mp_frac 
}
results_lr

# J48 decision tree
results_dt <- list()
for(df_name in names(churn_train_list)) {
  train_control <- trainControl(
    classProbs = T, 
    method = "repeatedcv",
    number = 2, 
    repeats = 5,
    summaryFunction = twoClassSummary
  )
  train_df <- churn_train_list[[df_name]]
  test_df <- churn_test_list[[df_name]]
  
  set.seed(1)
  model_cv <- train(
    churn ~ .,  
    data = train_df, 
    method = "J48", 
    trControl = train_control,
    metric = "ROC",
    tuneGrid = expand.grid(
      C = c(.01, seq(.15, .3, .01)),
      M = (nrow(train_df) * c(.01, .025, .05, .1, .25, .5)) %>% round()
    )
  )
  
  #training model with final parameters
  set.seed(1)
  model_obj <- train(
    churn ~ .,  
    data = train_df, 
    method = "J48", 
    trControl = trainControl(classProbs = T, summaryFunction = twoClassSummary),
    metric = "ROC",
    tuneGrid = data.frame(C = model_cv$bestTune$C, M = model_cv$bestTune$M)
  )
  
  true_response <- (test_df$churn == "Churn") * 1
  predicted <- predict(model_obj, newdata = test_df, type = "prob")$Churn
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
  
  results_dt$model[[df_name]] <- model_obj
  results_dt$auc[[df_name]] <- auc
  results_dt$td_lift[[df_name]] <- td_lift
  results_dt$mp[[df_name]] <- mp
  results_dt$mp_frac[[df_name]] <- mp_frac 
}
results_dt

# random forest
results_rf <- list()
for(df_name in names(churn_train_list)) {
  train_control <- trainControl(
    classProbs = T, 
    method = "repeatedcv",
    number = 2, 
    repeats = 5,
    summaryFunction = twoClassSummary
  )
  train_df <- churn_train_list[[df_name]]
  test_df <- churn_test_list[[df_name]]
  
  set.seed(1)
  model_cv <- train(
    churn ~ .,  
    data = train_df, 
    method = "rf",
    ntree = 1100,
    trControl = train_control,
    metric = "ROC",
    tuneGrid = expand.grid(
      mtry = (ncol(train_df) - 1) %>% 
        {. * c(.1, .25, .5, 1, 2, 4)} %>%
        sqrt() %>% round() %>% unique()
    )
  )
  print(model_cv)
  #training model with final parameters
  set.seed(1)
  model_obj <- train(
    churn ~ .,  
    data = train_df, 
    method = "rf", 
    ntree = 1100,
    trControl = trainControl(classProbs = T, summaryFunction = twoClassSummary),
    metric = "ROC",
    tuneGrid = data.frame(mtry = model_cv$bestTune$mtry)
  )
  
  true_response <- (test_df$churn == "Churn") * 1
  predicted <- predict(model_obj, newdata = test_df, type = "prob")$Churn
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
  
  results_rf$model[[df_name]] <- model_obj
  results_rf$auc[[df_name]] <- auc
  results_rf$td_lift[[df_name]] <- td_lift
  results_rf$mp[[df_name]] <- mp
  results_rf$mp_frac[[df_name]] <- mp_frac 
}
results_rf

# Radial SVM
results_svm <- list()
for(df_name in names(churn_train_list)) {
  train_control <- trainControl(
    classProbs = T, 
    method = "repeatedcv",
    number = 2, 
    repeats = 5,
    summaryFunction = twoClassSummary
  )
  train_df <- churn_train_list[[df_name]]
  test_df <- churn_test_list[[df_name]]
  
  set.seed(1)
  model_cv <- train(
    churn ~ .,  
    data = train_df, 
    method = "svmRadial",
    trControl = train_control,
    metric = "ROC",
    tuneGrid = expand.grid(
      C = 2 ^ c(-5, -3, -1, 1, 3, 5, 7, 9, 13),
      sigma = 2 ^ c(3, 1, -1, -3, -5, -7, -9, -11, -13, -15)
    )
  )
  
  #training model with final parameters
  set.seed(1)
  model_obj <- train(
    churn ~ .,  
    data = train_df, 
    method = "svmRadial", 
    trControl = trainControl(classProbs = T, summaryFunction = twoClassSummary),
    metric = "ROC",
    tuneGrid = data.frame(
      C = model_cv$bestTune$C, 
      sigma = model_cv$bestTune$sigma
    )
  )
  
  true_response <- (test_df$churn == "Churn") * 1
  predicted <- predict(model_obj, newdata = test_df, type = "prob")$Churn
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
  
  results_svm$model[[df_name]] <- model_obj
  results_svm$auc[[df_name]] <- auc
  results_svm$td_lift[[df_name]] <- td_lift
  results_svm$mp[[df_name]] <- mp
  results_svm$mp_frac[[df_name]] <- mp_frac 
}
results_svm

#LLM
#source("caret_llm.R") implementation in caret failed
# ROC was not calculated and many tuning parameters were not takein into account
# Hence, own CV implementation was used
results_llm <- list()
for(df_name in names(churn_train_list)) {
  train_df <- churn_train_list[[df_name]]
  test_df <- churn_test_list[[df_name]]
  set.seed(1)
  folds <- createMultiFolds(train_df$churn, k = 2, times = 5)
  tune_grid <- expand.grid(
    threshold_pruning = c(.01, seq(.15, .3, .01)),
    nbr_obs_leaf = 
      (nrow(train_df) * c(.01, .025, .05, .1, .25, .5)) %>% round()
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
  results_llm$best_params[[df_name]] <- best_params
  results_llm$auc[[df_name]] <- auc
  results_llm$td_lift[[df_name]] <- td_lift
  results_llm$mp[[df_name]] <- mp
  results_llm$mp_frac[[df_name]] <- mp_frac 
}
results_llm

# the best model does not have segments...
# therefore a model with segments is earch on


results_list <- list(
  results_lr = results_lr,
  results_dt = results_dt,
  results_rf = results_rf,
  results_svm = results_svm,
  results_llm = results_llm
)
# results grids
mp_hypo <- data.frame()
for(metric in names(results_lr) %>% {.[. != "model"]}){
  print(metric)
  .results <- sapply(results_list, function(x) x[[metric]]) %>% t()
  .results %>% round(2) %>% print()
  mp_hypo <- rbind(mp_hypo, data.frame(.results, metric = metric))
}

require(tidyr)

mp_hypo <- mp_hypo %>% 
  {mutate(., algo = rownames(.))} %>% 
  mutate(algo = gsub('[0-9]+', '', algo)) %>% 
  mutate(algo = gsub('results_', '', algo)) %>% 
  gather(dpt, value, normal_dpt, special_dpt, special_dpt_fisher) %>% 
  mutate(name = paste(algo, dpt, sep = "_")) %>% 
  mutate(
    metric = factor(metric, levels = c("mp", "td_lift", "auc", "mp_frac")))
mp_order <- mp_hypo %>% 
  filter(metric == "mp") %>% 
  arrange(value) %>% 
  pull(name)
mp_hypo <- mutate(mp_hypo, name = factor(name, levels = mp_order))

ggplot(mp_hypo, aes(x = name, y = value)) + 
  geom_bar(stat="identity", fill = "light blue") +
  facet_grid(. ~ metric, scales = "free") +
  coord_flip() +
  theme_minimal()

rank_chart_df <- mp_hypo %>% 
  filter(metric != "mp_frac") %>% 
  group_by(dpt, metric) %>% 
  mutate(rank = dense_rank(desc(value))) %>% 
  group_by(algo, metric) %>% 
  summarize(avg_rank = mean(rank)) %>% 
  ungroup() %>% 
  mutate(algo = factor(algo, levels = c("dt", "llm", "svm", "rf", "lr"))) 
# manual order
  
ggplot(rank_chart_df, aes(x = algo, y = avg_rank)) + 
  geom_bar(stat="identity", fill = "light blue") +
  facet_grid(. ~ metric, scales = "free") +
  coord_flip() +
  theme_minimal()

save.image()

rank_chart_df <- mp_hypo %>% 
  filter(metric != "mp_frac") %>% 
  group_by(dpt, metric) %>% 
  mutate(rank = dense_rank(desc(value))) %>% 
  group_by(algo, dpt) %>% 
  summarize(avg_rank = mean(rank)) %>% 
  ungroup() %>% 
  mutate(dpt = 
      factor(dpt, 
        levels = c("special_dpt", "special_dpt_fisher", "normal_dpt"))) %>% 
  mutate(algo = factor(algo, levels = c("dt", "rf", "llm", "svm", "lr"))) 
# manual order

ggplot(rank_chart_df, aes(x = algo, y = avg_rank)) + 
  geom_bar(stat="identity", fill = "light blue") +
  facet_grid(. ~ dpt, scales = "free") +
  coord_flip() +
  theme_minimal()




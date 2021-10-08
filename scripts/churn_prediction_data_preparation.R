#devtools::install_github("awstringer/modellingTools")
require(modellingTools)
require(CHAID)
require(PredPsych)
require(recipes)
require(caret)
require(dplyr)

churn_raw <- read.csv("churn_main_dataset.csv", stringsAsFactors = F)
outbound_sfid <- read.csv("outbound_sfid.csv", stringsAsFactors = F)
churn_raw$client_scource <- ifelse(
  churn_raw$Salesforce_id %in% outbound_sfid$salesforce_id,
  "Outbound",
  "Inbound"
) 
columns_to_drop <- c(
  "Salesforce_id",
  "contract_start",
  "contract_end",
  "Newest_Contract_End_Date",
  "ref_date",
  "extended.",
  "month_joined",
  "date_diff",
  "X"
)
churn_raw <- 
  churn_raw %>% 
  select(-one_of(columns_to_drop)) %>% 
  rename(churn = churned.,
         total_revenue_as_perc_of_first_contract = 
           total_revenue_as_._of_first_contract) %>% 
  mutate(churn = factor(churn, levels = c(1, 0)), 
         contract_no = case_when(
           contract_no <= 1 ~ 1,
           between(contract_no, 2, 4) ~ 4,
           between(contract_no, 5, 12) ~ 12,
           contract_no > 12 ~ 16),
         contract_length = case_when(
           contract_length <= 1 ~ 1,
           between(contract_length, 2, 4) ~ 4,
           between(contract_length, 5, 11) ~ 12,
           contract_length >= 12 ~ 16),
         months_as_client = case_when(
           months_as_client <= 3 ~ 3,
           between(months_as_client, 4, 6) ~ 6,
           between(months_as_client, 7, 12) ~ 12,
           months_as_client > 12 ~ 16)) %>% 
  mutate(
    contract_no = as.character(contract_no),
    contract_length = as.character(contract_length),
    months_as_client = as.character(months_as_client),
    upsell_downsell = as.character(upsell_downsell))
  

set.seed(1)
train_indx <- createDataPartition(churn_raw$churn, p = 0.8, list = F)
churn_train <- churn_raw[train_indx,]
churn_train_y <- churn_train$churn
churn_test <- churn_raw[-train_indx,]
churn_test_y <- churn_test$churn

{
#for tree discretization and establish whether needed for logistic regression
# contract_no:
# CHAID::chaid(churn ~ as.factor(contract_no), data = churn_train) %>% plot()
#0-1, 2-4, 5-12, 12+

# CHAID::chaid(churn ~ as.factor(contract_length), data = churn_train) %>% plot()
# 0-1, 2-4, 12, cała reszta

# moonths_as_client
# CHAID::chaid(churn ~ as.factor(moonths_as_client), data = churn_train) %>% plot()
# 0-3, 4-6, 7-12, 12+
}

# standard DPT
# normalization + dummy variables
numeric_vars <- sapply(churn_train, is.numeric) %>% {names(.)[. == T]}
scaled_churn_train <- scale(churn_train[, numeric_vars])
churn_train_normal_dpt <- cbind(
  churn_train %>% select(-numeric_vars),
  as.data.frame(scaled_churn_train)
)
scaling_factors <- data.frame(
  variable = attributes(scaled_churn_train)$`scaled:scale` %>% names(),
  mean = attributes(scaled_churn_train)$`scaled:center`,
  sd = attributes(scaled_churn_train)$`scaled:scale`,
  stringsAsFactors = F
)
normal_dpt_dummy_coder <- dummyVars(churn ~ ., data = churn_train_normal_dpt)
churn_train_normal_dpt <- 
  predict(normal_dpt_dummy_coder, newdata = churn_train_normal_dpt) %>% 
  data.frame() %>% 
  mutate(churn = churn_train_y)

# Test set
scale_test <- function(numeric_var_name,
                       data = churn_test, 
                       factors = scaling_factors) {
  x <- data[, numeric_var_name]
  .mean <- factors %>% {.$mean[.$variable == numeric_var_name]}
  .sd <- factors %>% {.$sd[.$variable == numeric_var_name]}
  (x - .mean) / .sd
}
churn_test_normal_dpt <- 
  cbind(churn_test %>% select(-numeric_vars),
        sapply(numeric_vars, scale_test) %>% data.frame()) %>% 
  {predict(normal_dpt_dummy_coder, newdata = .)} %>% 
  data.frame() %>% 
  mutate(churn = churn_test_y)

#Coussement optimal DPT
# decision tree based remapping for categorical - done at the beggining!
# equal frequency binning for continous variables
# WOE converesion

bin_predictors <- sapply(
  numeric_vars, 
  function(x){
    recipes::discretize(churn_train[, x], 
      cuts = 10, 
      keep_na = F, 
      min_unique = 1)},
  simplify = FALSE
)

churn_train_special_dpt <- cbind(
  select(churn_train, -numeric_vars),
  sapply(
    numeric_vars,
    function(x) 
      bin_predictors[[x]] %>% 
      predict(newdata = churn_train[, x]) %>% 
      as.data.frame()
  ) 
)
calculate_woe <- function(x, y){
  .df <- cbind(x, y) %>% {data.frame(x = .[, 1], y = .[, 2])}
  .df %>% 
    group_by(x) %>% 
    mutate(group_count = n(),
           churn_prop = sum((y == 1) * 1),
           non_churn_prop = sum((y == 2) * 1),
           woe = log(
             (churn_prop / group_count) / (non_churn_prop / group_count))) %>% 
    ungroup() %>% 
    pull(woe)
}
bins_to_woe_conv <- sapply(
  churn_train_special_dpt %>% select(-churn),
  function(x)
    data.frame(bins = x, woe = calculate_woe(x, churn_train_y)) %>% 
    distinct(),
  simplify = F
)
churn_train_special_dpt <-
  churn_train_special_dpt %>%
  select(-churn) %>%
  mutate_all(calculate_woe, churn_train_y) %>%
  cbind(churn = churn_train_y)

# Test set
churn_test_special_dpt <- cbind(
  select(churn_test, -numeric_vars),
  sapply(
    numeric_vars,
    function(x) 
      bin_predictors[[x]] %>% 
      predict(newdata = churn_test[, x]) %>% 
      as.data.frame()
  ) 
)

for(col in names(churn_test_special_dpt) %>% {.[. != "churn"]}) {
  churn_test_special_dpt[, col] <- sapply(
    churn_test_special_dpt[, col],
    function(x) bins_to_woe_conv[[col]] %>% {.$woe[.$bins == x]}
  )
}
  
# creating feature selection set with Fisher
# churn_train_normal_dpt %>% {names(.)[nearZeroVar(.)]}
# negative_conversation_count_change out
fisher_vars_normal_dpt <- 
  PredPsych::fscore(
    Data = churn_train_normal_dpt,
    classCol = which(names(churn_train_normal_dpt) == "churn"),
    featureCol = which(names(churn_train_normal_dpt) != "churn")) %>% 
  {.[order(., decreasing = T)]} %>% 
  {c("churn", names(.)[1:10])}
churn_train_normal_dpt_fisher <- select(
  churn_train_normal_dpt,
  fisher_vars_normal_dpt
)
churn_test_normal_dpt_fisher <- select(
  churn_test_normal_dpt,
  fisher_vars_normal_dpt
)

# special dpt
fisher_vars_special_dpt <- 
  PredPsych::fscore(
    Data = churn_train_special_dpt,
    classCol = which(names(churn_train_special_dpt) == "churn"),
    featureCol = which(names(churn_train_special_dpt) != "churn")) %>% 
  {.[order(., decreasing = T)]} %>% 
  {c("churn", names(.)[1:10])}
churn_train_special_dpt_fisher <- select(
  churn_train_special_dpt,
  fisher_vars_special_dpt
)
churn_test_special_dpt_fisher <- select(
  churn_test_special_dpt,
  fisher_vars_special_dpt
)

# to avoid problems in caret prediction methods levels are named differently
churn_train_list <- list(
  normal_dpt = churn_train_normal_dpt,
  special_dpt = churn_train_special_dpt,
  #normal_dpt_fisher = churn_train_normal_dpt_fisher,
  special_dpt_fisher = churn_train_special_dpt_fisher
)
for(df_name in names(churn_train_list)) {
  df <- churn_train_list[[df_name]]
  df$churn <- `levels<-`(df$churn, c("Churn", "Extension")) %>% 
    relevel("Extension")
  churn_train_list[[df_name]] <- df
}

churn_test_list <- list(
  normal_dpt = churn_test_normal_dpt,
  special_dpt = churn_test_special_dpt,
  #normal_dpt_fisher = churn_test_normal_dpt_fisher,
  special_dpt_fisher = churn_test_special_dpt_fisher
)
for(df_name in names(churn_test_list)) {
  df <- churn_test_list[[df_name]]
  df$churn <- `levels<-`(df$churn, c("Churn", "Extension")) %>% 
    relevel("Extension")
  churn_test_list[[df_name]] <- df
}





  
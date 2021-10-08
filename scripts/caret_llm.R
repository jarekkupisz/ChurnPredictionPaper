require(LLM)
# Using LLM model in caret
# source: http://topepo.github.io/caret/using-your-own-model-in-train.html
caret_llm <- list(type = "Classification", library = "LLM", loop = NULL)
caret_llm$parameters <- data.frame(
  parameter = c("threshold_pruning", "nbr_obs_leaf"),
  class = rep("numeric", 2),
  label = c("Confidence Threshold Pruning", "No. of Observations in a Leaf")
)
caret_llm$fit <- function(x, y, 
                          wts, 
                          param, 
                          lev, 
                          last, 
                          weights, 
                          classProbs, 
                          ...) {
  LLM::llm(
    X = as.data.frame(x), 
    Y = factor(y), 
    threshold_pruning = param$threshold_pruning, 
    nbr_obs_leaf = param$nbr_obs_leaf,
    ...
  ) 
}
caret_llm$predict <- function(modelFit, 
                              newdata, 
                              preProc = NULL, 
                              submodels = NULL) {
  LLM::predict.llm(modelFit, as.data.frame(newdata))$probability > 0.5 %>%
    ifelse("Churn", "Extension") %>%
    factor(levels = c("Extension", "Churn"))
}
caret_llm$prob <- function(modelFit, 
  newdata, 
  preProc = NULL, 
  submodels = NULL) {
  set.seed(1)
  LLM::predict.llm(modelFit, as.data.frame(newdata))$probability %>% 
    {case_when(. == 1 ~ runif(1, 0.95, 0.99), 
               . == 0 ~ runif(1, 0.01, 0.05), 
               T ~ .)} %>% #this algo can give 1s
    {data.frame(Extension = 1 - ., Churn = .)} 
}
caret_llm$sort <- function(x) x[order(x[,1]), ]
caret_llm$levels <- function(x) 
  factor(c("Extension", "Churn"), levels = c("Extension", "Churn"))
caret_llm$grid <- function(x, y, len = NULL, search = "grid") {
  if(seatch == "random"){
    out <- data.frame(threshold_pruning = 0.25, nbr_obs_leaf = 100)
  } else {
    out <- data.frame(threshold_pruning = 0.25, nbr_obs_leaf = 100)
  }
  out
}
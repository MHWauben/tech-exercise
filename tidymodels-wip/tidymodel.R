library(tidymodels)
library(tidyverse)
library(workflows)
library(tune)
library(randomForest)

# Prepare data
collisions_t <- collisions %>%
  select(casualty_severity, casualty_age, 
         casualty_class, casualty_sex, 
         day, time, road_type, speed_limit, 
         special_conditions, 
         easting, northing) %>%
  mutate_if(is.character, tolower) %>%
  mutate_if(is.character, as.factor) 

# Split data
collisions_split <- rsample::initial_split(collisions_t, prop = 0.75)
c_train <- training(collisions_split)
c_test <- testing(collisions_split)

c_cv <- vfold_cv(c_train)

# Prepare pre-processing
collisions_recipe <- recipe(casualty_severity ~ casualty_age + casualty_class + 
                              casualty_sex + day + time + road_type + speed_limit + 
                              special_conditions + easting + northing,
                            data = collisions_t) %>%
  step_normalize(all_numeric()) %>%
  step_knnimpute(all_predictors()) %>%
  step_dummy(all_predictors()) %>%
  # Data wildly unbalanced, vast majority slight
  step_downsample(casualty_severity)

c_preprocessed <- collisions_recipe %>%
  prep(c_train) %>%
  juice()

# Set up model and workflow
rf_model <- rand_forest() %>%
  set_args(mtry = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(collisions_recipe) %>%
  add_model(rf_model)

# Tune model to get final parameters
rf_tune_results <- rf_workflow %>%
  tune_grid(resamples = c_cv,
            grid =  expand.grid(mtry = c(3, 4, 5)),
            metrics = metric_set(accuracy, roc_auc))
param_final <- rf_tune_results %>%
  select_best(metric = "accuracy")
rf_workflow <- rf_workflow %>%
  finalize_workflow(param_final)

param_final <- rf_tune_results %>%
  select_best(metric = "accuracy")
rf_workflow <- rf_workflow %>%
  finalize_workflow(param_final)

# Fit on the training set and combine with test set
rf_fit <- rf_workflow %>%
  last_fit(collisions_split)
rf_fit %>% collect_metrics()
c_test$rf_predictions <- collect_predictions(rf_fit)$.pred_class

# Visualise performance
c_test %>%
  dplyr::group_by(casualty_severity, rf_predictions) %>%
  summarise(Freq = n()) %>%
  ungroup() %>%
  dplyr::mutate(prop = Freq / sum(Freq)) %>%
  ggplot(aes(x = casualty_severity, y = rf_predictions, fill = Freq))+
  geom_tile()+ 
  guides(colour = guide_legend(reverse=T))


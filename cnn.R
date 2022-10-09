# libraries ---------------------------------------------------------------

library(tidyverse)
library(here)
library(ggthemes)
library(keras)
library(tensorflow)
library(tfdatasets)
library(tidymodels)

# Test whether GPU is accessible ------------------------------------------

tf$config$list_physical_devices() %>%
  map(~ str_detect(., "GPU")) %>%
  unlist() %>%
  sum()

# memory optimisation -----------------------------------------------------

physical_devices <- tf$config$list_physical_devices("GPU")
tf$config$experimental$set_memory_growth(physical_devices[[1]], TRUE)
tf$keras$backend$set_floatx("float32")

# read data ---------------------------------------------------------------

# Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
classes <- tibble(X188 = 0:4,
                  Class = c("N", "S", "V", "F", "Q") %>%
                    fct_reorder(X188))

training_tbl <- here("data", "mitbih_train.csv") %>%
  read_csv(col_names = F) %>%
  left_join(classes) %>%
  select(-X188)

test_tbl <- here("data", "mitbih_test.csv") %>%
  read_csv(col_names = F) %>%
  left_join(classes)  %>%
  select(-X188)

overview <- training_tbl %>% 
  select(Class) %>% 
  mutate(row = row_number()) %>% 
  group_by(Class) %>% 
  summarise(across(everything(), list(start = min, end = max)), 
            n = n()) %>%
  mutate(prop = n / sum(n),
         suggested = floor(prop * floor(sum(n) / 2)) %>% as.integer())

overview

train_ind <- c(
  sample(overview$row_start[1]:overview$row_end[1], overview$suggested[1]),
  sample(overview$row_start[2]:overview$row_end[2], overview$suggested[2]),
  sample(overview$row_start[3]:overview$row_end[3], overview$suggested[3]),
  sample(overview$row_start[4]:overview$row_end[4], overview$suggested[4]),
  sample(overview$row_start[5]:overview$row_end[5], overview$suggested[5])
  )

val_ind <- seq_len(nrow(training_tbl))[-train_ind]

training_data <- training_tbl %>%
  slice(train_ind)

val_data <- training_tbl %>%
  slice(val_ind)

# split into training and validation --------------------------------------

#shuffle training data
shuffle_df <- function(tbl) {
  tbl_len <- nrow(tbl)
  new_order <- sample(1:tbl_len, tbl_len)
  tbl %>% slice(new_order)
}

training_data <- shuffle_df(training_data)

training_data %>% 
  select(Class) %>% 
  mutate(row = row_number()) %>% 
  group_by(Class) %>% 
  summarise(across(everything(), list(min, max)))

# training ----------------------------------------------------------------

# mlp ---------------------------------------------------------------------

mlp_recipe <- recipe(Class ~ ., data = training_data) %>%
  step_normalize(all_predictors()) %>%
  prep(training = training_data, retain = TRUE)

val_normalized <- bake(mlp_recipe, new_data = val_data, all_predictors())

# to test later on
test_normalized <- bake(mlp_recipe, new_data = test_tbl, all_predictors())

set.seed(73)

mlp_fit <-
  mlp(epochs = 20L, hidden_units = 60L, dropout = 0.2) %>%
  set_mode("classification") %>% 
  set_engine("keras") %>%
  fit(Class ~ ., data = bake(mlp_recipe, new_data = NULL))  


# validate ----------------------------------------------------------------
# is a bit risky because the algorithm might learn the person!

results <- val_data %>%
  select(Class) %>%
  bind_cols(
    predict(mlp_fit, new_data = val_normalized),
    predict(mlp_fit, new_data = val_normalized, type = "prob")
  )

results

results %>% 
  accuracy(truth = Class, .pred_class)

results %>% 
  conf_mat(truth = Class, .pred_class)

# test data ---------------------------------------------------------------
# for generalization

test_results <- test_tbl %>%
  select(Class) %>%
  bind_cols(
    predict(mlp_fit, new_data = test_normalized),
    predict(mlp_fit, new_data = test_normalized, type = "prob")
  )

test_results

test_results %>% 
  accuracy(truth = Class, .pred_class)

test_results %>% 
  conf_mat(truth = Class, .pred_class)


# xgboost  -------------------------------------------------------

bt_cls_spec <- 
  boost_tree(trees = 15) %>% 
  # This model can be used for classification or regression, so set mode
  set_mode("classification") %>% 
  set_engine("xgboost")
bt_cls_spec

set.seed(1)
bt_cls_fit <- bt_cls_spec %>% fit(Class ~ ., data = training_tbl)
bt_cls_fit

xboost_result <- test_tbl %>%
  select(Class) %>%
  bind_cols(
  predict(bt_cls_fit, test_tbl),
  predict(bt_cls_fit, test_tbl, type = "prob")
)

xboost_result %>%
  accuracy(truth = Class, .pred_class)

xboost_result %>% 
  conf_mat(truth = Class, .pred_class)


# cnn ---------------------------------------------------------------------

training_tbl <- here("data", "mitbih_train.csv") %>%
  read_csv(col_names = F) 

training_tbl_shuffled <- training_tbl %>%
  slice(sample(1:nrow(training_tbl), nrow(training_tbl)))
  
test_tbl <- here("data", "mitbih_test.csv") %>%
  read_csv(col_names = F)

xtrain <- training_tbl_shuffled %>% 
  select(-length(training_tbl)) %>%
  as.matrix

ytrain <- training_tbl_shuffled %>% 
  select(length(training_tbl)) %>%
  as.matrix 

xtest <- test_tbl %>% 
  select(-length(training_tbl)) %>%
  as.matrix

ytest <- test_tbl %>% 
  select(length(training_tbl)) %>%
  as.matrix 

batch_size <- 128
num_classes <- 5
epochs <- 50

input_shape <- c(187, 1)

ytrain <- to_categorical(ytrain, num_classes)
ytest <- to_categorical(ytest, num_classes)

# define model structure 
cnn_model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = c(5), activation = 'relu', input_shape = input_shape) %>% 
  layer_max_pooling_1d(strides = 2, pool_size = c(5)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_1d(filters = 32, kernel_size = c(5), activation = 'relu', input_shape = input_shape) %>% 
  layer_max_pooling_1d(strides = 2, pool_size = c(5)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dense(units = num_classes, activation = 'softmax')

cnn_model

# Compile model
cnn_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


# set checkpoints callback ------------------------------------------------

cp_callback <- callback_model_checkpoint(
  filepath = "class_5/all_checkpoints.ckpt",
  save_weights_only = TRUE,
  verbose = 1
)

# Train model
cnn_history <- cnn_model %>% fit(
  xtrain, ytrain,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2,
  callbacks = list(cp_callback)
)

plot(cnn_history) + theme_minimal()
cnn_model %>% evaluate(xtest, ytest)

cnn_pred <- cnn_model %>% 
  predict(xtest) %>%
  k_argmax %>%
  as.array

head(cnn_pred, n=50)
sum(cnn_pred != test_tbl$X188)


# results -----------------------------------------------------------------

test_results <- test_tbl %>%
  select(Class = X188) %>%
  bind_cols(
    predict(cnn_model, xtest) %>%
      k_argmax %>%
      as.array,
    predict(cnn_model, xtest)
  ) %>%
  rename(
    Class_predicted = 2,
    prob_0 = 3,
    prob_1 = 4,
    prob_2 = 5,
    prob_3 = 6,
    prob_4 = 7
  ) %>%
  mutate(across(contains("Class"), as.factor))

test_results

test_results %>% 
  accuracy(truth = Class, Class_predicted)

test_results %>% 
  conf_mat(truth = Class, Class_predicted)

test_results %>% 
  conf_mat(truth = Class, Class_predicted) %>%
  autoplot()

# make it a three class problem -------------------------------------------

training_tbl_shuffled_3class <- training_tbl_shuffled %>%
  mutate(X188 = case_when(X188 %in% c(0,3,4) ~ 0,
                         T ~ X188))

test_tbl_3class <- test_tbl %>%
  mutate(X188 = case_when(X188 %in% c(0,3,4) ~ 0,
                          T ~ X188))

xtrain <- training_tbl_shuffled_3class %>% 
  select(-length(training_tbl_shuffled_3class)) %>%
  as.matrix

ytrain <- training_tbl_shuffled_3class %>% 
  select(length(training_tbl_shuffled_3class)) %>%
  as.matrix 

xtest <- test_tbl_3class %>% 
  select(-length(test_tbl_3class)) %>%
  as.matrix

ytest <- test_tbl_3class %>% 
  select(length(test_tbl_3class)) %>%
  as.matrix 

batch_size <- 128
num_classes <- 3
epochs <- 50

input_shape <- c(187, 1)

ytrain <- to_categorical(ytrain, num_classes)
ytest <- to_categorical(ytest, num_classes)

# define model structure 
cnn_model_3class <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = c(5), activation = 'relu', input_shape = input_shape) %>% 
  layer_max_pooling_1d(strides = 2, pool_size = c(5)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_1d(filters = 32, kernel_size = c(5), activation = 'relu', input_shape = input_shape) %>% 
  layer_max_pooling_1d(strides = 2, pool_size = c(5)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dense(units = num_classes, activation = 'softmax')

cnn_model_3class

# Compile model
cnn_model_3class %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


# set checkpoints callback ------------------------------------------------

cp_callback <- callback_model_checkpoint(
  filepath = "class_3/all_checkpoints.ckpt",
  save_weights_only = TRUE,
  verbose = 1
)

# Train model
cnn_3_class_history <- cnn_model_3class %>% fit(
  xtrain, ytrain,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2,
  callbacks = list(cp_callback)
)


# results 3 class ---------------------------------------------------------

plot(cnn_3_class_history) + theme_minimal()
cnn_model_3class %>% evaluate(xtest, ytest)

cnn_pred <- cnn_model_3class %>% 
  predict(xtest) %>%
  k_argmax %>%
  as.array

head(cnn_pred, n=50)
sum(cnn_pred != test_tbl_3class$X188)


# results -----------------------------------------------------------------

test_3_class_results <- test_tbl_3class %>%
  select(Class = X188) %>%
  bind_cols(
    predict(cnn_model_3class, xtest) %>%
      k_argmax %>%
      as.array,
    predict(cnn_model_3class, xtest)
  ) %>%
  rename(
    Class_predicted = 2,
    prob_0 = 3,
    prob_1 = 4,
    prob_2 = 5
  ) %>%
  mutate(across(contains("Class"), as.factor))

test_3_class_results

test_3_class_results %>% 
  accuracy(truth = Class, Class_predicted)

test_3_class_results %>% 
  conf_mat(truth = Class, Class_predicted)



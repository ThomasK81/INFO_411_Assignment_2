library(tidyverse)
library(here)
library(ggthemes)
library(tidymodels)
library(echarts4r)
library(Rtsne)
library(tictoc)

# EDA --------------------------------------------------------------------

# 1. Compare Training and Testing Data ------------------------------------
# Remark: All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.


training_data <- here("data", "mitbih_train.csv") %>% read_csv(col_names = F)
testing_data <- here("data", "mitbih_test.csv") %>% read_csv(col_names = F)

all_data <- bind_rows(training_data %>% mutate(data = "training"),
                      testing_data %>% mutate(data = "testing")) %>%
  rename(class = X188)

# check class distribution ------------------------------------------------------

all_data %>%
  ggplot(aes(class, col = data)) +
  geom_density()

# the density graph for testing and training look comparable

# absolute numbers
all_data %>%
  ggplot(aes(class, fill = data)) +
  geom_histogram(bins = 5)

# proporation
all_data %>%
  group_by(data) %>%
  count(class) %>%
  mutate(prop = n / sum(n))

# vis 
all_data %>%
  group_by(data) %>%
  count(class) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(class, prop, fill = data)) +
  geom_col(position = "dodge2")


# from here on we jus look at the training set (otherwise we cheat) --------

# 2. Check summary --------------------------------------------------------


training_data <- training_data %>%
  rename(class = X188) %>%
  mutate(class = as.integer(class))

training_data %>% 
  ggplot(aes(X99)) +
  geom_boxplot() +
  facet_wrap(~class)

training_data_summary <- training_data %>% 
  group_by(class) %>%
  summarise(across(everything(), .fns = list(mean = mean,
                                             median = median,
                                             min = min, 
                                             max = max,
                                             sd = sd))) %>%
  ungroup %>%
  pivot_longer(-class, names_to = "names", values_to = "values") %>%
  separate(names, into = c("feature", "function")) %>%
  pivot_wider(names_from = `function`, values_from = values)

training_data_summary %>%
  mutate(class = as.character(class),
         feature = str_remove_all(feature, "X") %>% 
           formatC(width = 3, flag = 0)) %>%
  ggplot(aes(y = feature)) +
  geom_col(aes(x = mean, fill = class)) +
  facet_wrap(~class, ncol = 5)

training_data_summary %>%
  mutate(class = as.character(class),
         feature = str_remove_all(feature, "X") %>% 
           formatC(width = 3, flag = 0)) %>%
  ggplot(aes(y = feature)) +
  geom_col(aes(x = sd, fill = class)) +
  facet_wrap(~class, ncol = 5)


# 2a. look at heartbeats --------------------------------------------------

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 0) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .5) +
  geom_smooth()

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 1) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .5) +
  geom_smooth()

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 2) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .5) +
  geom_smooth()

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 3) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .5) +
  geom_smooth()

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 4) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .5) +
  geom_smooth()


# remove 0s ---------------------------------------------------------------

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 0) %>%
  filter(value != 0) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .7) +
  geom_smooth(col = "red") +
  labs(title = "Class 0")

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 1) %>%
  filter(value != 0) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .7) +
  geom_smooth(col = "red")+
  labs(title = "Class 1")

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 2) %>%
  filter(value != 0) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .7) +
  geom_smooth(col = "red")+
  labs(title = "Class 2")

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 3) %>%
  filter(value != 0) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .7) +
  geom_smooth() +
  labs(title = "Class 3")

training_data %>% 
  pivot_longer(-class, names_to = "reading", values_to = "value") %>%
  mutate(reading = str_remove_all(reading, "X") %>% as.integer()) %>%
  filter(class == 4) %>%
  filter(value != 0) %>%
  ggplot(aes(reading, value)) +
  geom_hex(alpha = .7) +
  geom_smooth(col = "red")+
  labs(title = "Class 4")

# 3. PCA ---------------------------------------------------------------------

# produce matrix
training_data_mat <- training_data %>%
  select(-class) %>%
  as.matrix

rownames(training_data_mat) <- training_data$class

training_pca <- prcomp(training_data_mat, center = T, scale. = T)

training_pca$x[,1:2] %>%
  as_tibble() %>%
  mutate(class = training_data$class %>% as.character()) %>%
  ggplot(aes(PC1, PC2, col = class)) +
  geom_point(alpha = 0.3) +
  theme_minimal()

training_pca$x[,1:3] %>%
  as_tibble() %>%
  mutate(class = training_data$class %>% as.character()) %>%
  group_by(class) %>%
  e_charts(PC1) %>%
  e_scatter_3d(PC2, PC3) %>%
  e_tooltip()

# 4. t-sne -------------------------------------------------------------------

tic()
training_tsne <- Rtsne(training_data_mat, pca_scale = T, verbose = T, perplexity = 30, max_iter = 5000)
toc()

training_tsne$Y %>%
  as_tibble() %>%
  mutate(class = training_data$class %>% as.character()) %>%
  ggplot(aes(V1, V2, col = class)) +
  geom_point(alpha = 0.3) +
  theme_minimal()

tic()
training_tsne_unscaled <- Rtsne(training_data_mat, pca_scale = F, verbose = T, perplexity = 30, max_iter = 5000)
toc()

training_tsne_unscaled$Y %>%
  as_tibble() %>%
  mutate(class = training_data$class %>% as.character()) %>%
  ggplot(aes(V1, V2, col = class)) +
  geom_point(alpha = 0.3) +
  theme_minimal()


# tsne_all_data -----------------------------------------------------------

training_data_mat_all <- bind_rows(training_data %>% filter(class == 0), testing_data %>% filter(X188 == 0) %>% rename(class = X188)) %>%
  select(-class) %>%
  as.matrix

tic()
training_tsne_all <- Rtsne(training_data_mat_all, pca_scale = T, verbose = T, perplexity = 30, max_iter = 5000)
toc()

tsne_data_names <- bind_rows(training_data %>% filter(class == 0) %>% mutate(data_name = "training"), 
                             testing_data %>% filter(X188 == 0) %>% rename(class = X188) %>% mutate(data_name = "testing")) %>%
  pull(data_name)

training_tsne_all$Y %>%
  as_tibble() %>%
  mutate(data_set = tsne_data_names) %>%
  ggplot(aes(V1, V2, col = data_set)) +
  geom_point(alpha = 0.3) +
  theme_minimal()

training_data_mat_all_class_3 <- bind_rows(training_data %>% filter(class == 3), 
                                           testing_data %>% filter(X188 == 3) %>% rename(class = X188)) %>%
  select(-class) %>%
  as.matrix

tic()
training_tsne_all_class_3 <- Rtsne(training_data_mat_all_class_3, pca_scale = T, verbose = T, perplexity = 30, max_iter = 5000)
toc()

tsne_data_names <- bind_rows(training_data %>% filter(class == 3) %>% mutate(data_name = "training"), 
                             testing_data %>% filter(X188 == 3) %>% rename(class = X188) %>% mutate(data_name = "testing")) %>%
  pull(data_name)

training_tsne_all_class_3$Y %>%
  as_tibble() %>%
  mutate(data_set = tsne_data_names) %>%
  ggplot(aes(V1, V2, col = data_set)) +
  geom_point(alpha = 0.3) +
  theme_minimal()

# drop initial PCA --------------------------------------------------------
# increase perplexity -----------------------------------------------------

tic()
training_tsne_no_pca <- Rtsne(training_data_mat, initial_dims = 187, pca = F, verbose = T, perplexity = 50, max_iter = 5000)
toc()

training_tsne_no_pca$Y %>%
  as_tibble() %>%
  mutate(class = training_data$class %>% as.character()) %>%
  ggplot(aes(V1, V2, col = class)) +
  geom_point(alpha = 0.3) +
  theme_minimal()



# 5. some models ----------------------------------------------------------




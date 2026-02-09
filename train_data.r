library(readr)
flair_features <- read_csv("fz/flair_features.csv")
t1_features <- read_csv("fz/t1_features.csv")
t1c_features <- read_csv("fz/t1c_features.csv")
t2_features <- read_csv("fz/t2_features.csv")
label <- read_csv("fz/label.csv")

merged_df <- Reduce(
  function(x, y) merge(x, y, by = "Patient", all = TRUE),
  list(flair_features, t1_features, t1c_features, t2_features,label)
)

data_label <- na.omit(merged_df)
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
train_data_label <- data_label
# Select numeric columns and drop non-numeric columns
numeric_columns <- sapply(data_label, is.numeric)
data_df_num <- data_label[, numeric_columns]
X <- data_df_num
variance <- apply(X, 2, var)
non_zero_variance_indices <- which(variance > 1e-8)
X <- X[, non_zero_variance_indices]
library(caret)
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled_clinical <- predict(scaler, X)
X_scaled <- X_scaled_clinical[,-c(1:3)]
remove_duplicate_columns <- function(df) {
  keep_columns <- c()
  
  for (i in 1:ncol(df)) {
    is_duplicate <- FALSE
    for (j in keep_columns) {
      if (all(df[, i] == df[, j])) {
        is_duplicate <- TRUE
        break
      }
    }
    
    if (!is_duplicate) {
      keep_columns <- c(keep_columns, i)
    }
  }
  
  return(df[, keep_columns])
}

X_scaled_clean <- remove_duplicate_columns(X_scaled)
X_scaled_clean_multicener <- X_scaled_clean


y=data_label[,c(1:4)]          #！！！！！！！！！！！！！！———————

data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 

train <- data[,-(1:3)]
significant_features <- list()
alpha <- 0.05

for (column in 2:ncol(train)) {
  feature_name <- colnames(train)[column]
  feature <- unlist(train[, column])
  
  label_0 <- feature[train[, 1] == 0]
  label_1 <- feature[train[, 1] == 1]
  
  shapiro_test <- shapiro.test(feature)
  if (shapiro_test$p.value < alpha) {
    mannwhitney_test <- wilcox.test(label_0, label_1)
    p_value <- mannwhitney_test$p.value
  } else {
    t_test <- t.test(label_0, label_1)
    p_value <- t_test$p.value
  }
  
  if (p_value < alpha) {
    significant_features[[feature_name]] <- p_value
  }
}

significant_features_df <- data.frame(
  Feature = names(significant_features),
  P_Value = unlist(significant_features)
)

significant_features_df <- significant_features_df[order(significant_features_df$P_Value), ]
X_train <- as.matrix(train[, significant_features_df$Feature])
y_train <- train[1]
y_train <- y_train[,1]
train_data <- cbind.data.frame(train[1],X_train)

set.seed(42)
library(glmnet)

X_train <- as.matrix(X_train)  
y_train <- as.factor(y_train)                             

cv.lasso <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial", nfolds = 10, maxit = 50000, type.measure = "class")
lasso.coef1 <- coef(cv.lasso, s = "lambda.min")


selected_features <- rownames(lasso.coef1)[lasso.coef1[, 1] != 0]
selected_features <- selected_features[-1]  

selected_feature_names <- selected_features



lasso_weights = lasso.coef1@x[-1]  
results4 <- data.frame(
  Feature = selected_feature_names,
  Weight = lasso_weights
)



X_train_lasso_selected <- X_train[, selected_feature_names]
y_train

train_data <- cbind.data.frame(y_train,X_train_lasso_selected)


X_train_lasso_selected <- train_data[,-1]
y_train <- train_data$y_train
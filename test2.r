library(readr)
flair_features <- read_csv("bj/flair_features.csv")
t1_features <- read_csv("bj/t1_features.csv")
t1c_features <- read_csv("bj/t1c_features.csv")
t2_features <- read_csv("bj/t2_features.csv")
label <- read_csv("bj/label.csv")

merged_df <- Reduce(
  function(x, y) merge(x, y, by = "Patient", all = TRUE),
  list(flair_features, t1_features, t1c_features, t2_features,label)
)

data_label <- na.omit(merged_df)
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
test_data_label_dy <- data_label
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
y=data_label[,c(1:4)]
data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 
test <- data

X_test_lasso_selected <- test[, selected_feature_names]
y_test_tt<- test[,4]
test_data_tt <- cbind.data.frame(y_test_tt,X_test_lasso_selected)



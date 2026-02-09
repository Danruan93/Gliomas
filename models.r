library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(pROC)
library(ggplot2)


### 1. Random Forest ###########___________---------------------_________________________
X <- X_train_lasso_selected
y <- y_train

set.seed(123)
train_control <- trainControl(method = "cv", number = 5)
rf_model <- train(X, y, method = "rf", trControl = train_control)
print(rf_model)


finelmodel <- rf_model[["finalModel"]]
predict_prop <- finelmodel[["votes"]][,1]
logreg_roc_train <- roc(y, predict_prop)
auc_value <- auc(logreg_roc_train)
auc_ci <- ci.auc(logreg_roc_train)


logreg_conf_matrix_train <- confusionMatrix(finelmodel[["predicted"]], as.factor(y), positive = "1")
tn_train <- logreg_conf_matrix_train$table[1, 1]
fn_train <- logreg_conf_matrix_train$table[1, 2]
fp_train <- logreg_conf_matrix_train$table[2, 1]
tp_train <- logreg_conf_matrix_train$table[2, 2]
kappa_value <- logreg_conf_matrix_train[["overall"]][["Kappa"]]
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int
sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int
specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

cat("Logistic Regression Model (Training Set) Evaluation Metrics:\n")
cat("Sensitivity:", sensitivity_train, "\n")
paste(tp_train, "/", tp_train+fn_train)
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")

cat("Specificity:", specificity_train, "\n")
paste(tn_train, "/", tn_train + fp_train)
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")

ppv_train <- tp_train / (tp_train + fp_train)
ppv_ci_train <- binom.test(tp_train, tp_train + fp_train)$conf.int
cat("ppv:", ppv_train, "\n")
paste(tp_train, "/", tp_train + fp_train)
cat("95% CI for ppv:", ppv_ci_train[1], "-", ppv_ci_train[2], "\n")

npv_train <- tn_train / (tn_train + fn_train)
npv_ci_train <- binom.test(tn_train, tn_train + fn_train)$conf.int
cat("npv:", npv_train, "\n")
paste(tn_train, "/", tn_train + fn_train)
cat("95% CI for npv:", npv_ci_train[1], "-", npv_ci_train[2], "\n")

cat("Accuracy:", accuracy_train, "\n")
c
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")

auc_value
auc_ci

kappa_value



#test_img_xm
# test_data_dy 
X_test_test <- test_data_dy[,-1]
y_test_test <- test_data_dy[,1]
#test_img_tt
# test_data_tt 
X_test_test <- test_data_tt[,-1]
y_test_test <- test_data_tt[,1]


#### testing
X_test <- X_test_test
y_test <- y_test_test
y_test <- as.factor(y_test)

predictions <- predict(rf_model, newdata = X_test)
accuracy <- sum(predictions == y_test) / length(y_test)

predictions <- predict(rf_model, newdata = X_test)
probabilities <- predict(rf_model, newdata = X_test, type = "prob")[, 1]
conf_matrix <- confusionMatrix(predictions, as.factor(y_test), positive = '1')
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
roc_obj <- roc(as.numeric(y_test), probabilities)
auc_test <- auc(roc_obj)
auc_test_ci <- ci.auc(roc_obj)


tn_train <- conf_matrix$table[1, 1]
fn_train <- conf_matrix$table[1, 2]
fp_train <- conf_matrix$table[2, 1]
tp_train <- conf_matrix$table[2, 2]
kappa_value <- conf_matrix[["overall"]][["Kappa"]]
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int
sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int
specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

cat("Logistic Regression Model (Training Set) Evaluation Metrics:\n")
cat("Sensitivity:", sensitivity_train, "\n")
paste(tp_train, "/", tp_train+fn_train)
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")

cat("Specificity:", specificity_train, "\n")
paste(tn_train, "/", tn_train + fp_train)
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")

ppv_train <- tp_train / (tp_train + fp_train)
ppv_ci_train <- binom.test(tp_train, tp_train + fp_train)$conf.int
cat("ppv:", ppv_train, "\n")
paste(tp_train, "/", tp_train + fp_train)
cat("95% CI for ppv:", ppv_ci_train[1], "-", ppv_ci_train[2], "\n")

npv_train <- tn_train / (tn_train + fn_train)
npv_ci_train <- binom.test(tn_train, tn_train + fn_train)$conf.int
cat("npv:", npv_train, "\n")
paste(tn_train, "/", tn_train + fn_train)
cat("95% CI for npv:", npv_ci_train[1], "-", npv_ci_train[2], "\n")

cat("Accuracy:", accuracy_train, "\n")
paste(tp_train + tn_train, "/" ,tp_train + tn_train + fp_train + fn_train)
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")

auc_test
auc_test_ci

kappa_value




##############XGBoost______________
### 2. XGBoost  ###########___________---------------------_________________________###########______
library(xgboost)
train_control <- trainControl(method = "cv", number = 5)
xgb_grid <- expand.grid(
  nrounds = 100,  
  eta = 0.05,      
  max_depth = 6,  
  gamma = 1,
  colsample_bytree = 0.7,
  min_child_weight = 0.7,
  subsample = 0.9
)

X <- X_train_lasso_selected
y <- y_train
levels(y) <- make.names(levels(y))
set.seed(123)
train_control <- trainControl(method = "cv", number = 5,savePredictions = "final",classProbs = TRUE )  
xgb_model <- train(X, y, method = "xgbTree", trControl = train_control,tuneGrid = xgb_grid, verbose = FALSE)
print(xgb_model)
predictions <- xgb_model$pred
library(caret)
conf_matrix <- confusionMatrix(predictions$pred, predictions$obs)
print(conf_matrix)



### 3.logistic regression  ###########___________---------------------_________________________

X <- X_train_lasso_selected
y <- y_train
levels(y) <- make.names(levels(y))
set.seed(123)
train_control <- trainControl(
  method = "cv",          
  number = 5,            
  savePredictions = "final",  
  classProbs = TRUE       
)

library(caret)
logistic_model <- caret::train(X, y, method = "glm", family = "binomial", trControl = train_control)
print(logistic_model)


predictions <- logistic_model$pred
library(caret)
conf_matrix <- confusionMatrix(predictions$pred, predictions$obs,positive = 'X1')
print(conf_matrix)



### 4. Naive Bayes ###########___________---------------------_________________________
library(e1071)
X <- X_train_lasso_selected
colnames(X) <- gsub("-", "_", colnames(X))
y <- y_train
y <- as.factor(y)
levels(y) <- make.names(levels(y))
set.seed(123)
train_control <- trainControl(method = "cv", number = 5,savePredictions = "final",classProbs = TRUE ) 
nb_model <- train(X, y, method = "nb", trControl = train_control)
print(nb_model)
predictions <- nb_model$pred
library(caret)
conf_matrix <- confusionMatrix(predictions$pred, predictions$obs)
print(conf_matrix)



### 5. SVM ###########___________---------------------_________________________
X <- X_train_lasso_selected
y <- y_train
levels(y) <- make.names(levels(y))
train_control <- trainControl(method = "cv", number = 5,savePredictions = "final",classProbs = TRUE ) 
set.seed(123)
svm_linear <- train(X, y, method = "svmLinear", trControl = train_control)    

print(svm_linear)

predictions <- svm_linear$pred
library(caret)
conf_matrix <- confusionMatrix(predictions$pred, predictions$obs)
print(conf_matrix)



### 6. KNN  ###########___________---------------------_________________________
X <- X_train_lasso_selected
y <- y_train
levels(y) <- make.names(levels(y))
set.seed(42)
train_control <- trainControl(method = "cv", number = 5,savePredictions = "final",classProbs = TRUE ) 
knn_model <- train(X, y, method = "knn", trControl = train_control)
print(knn_model)
predictions <- knn_model$pred
library(caret)
conf_matrix <- confusionMatrix(predictions$pred, predictions$obs)
print(conf_matrix)


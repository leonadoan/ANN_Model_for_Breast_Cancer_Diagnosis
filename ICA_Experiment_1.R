# To illustrate data strategy to predict breast cancer
# ML model
# Le Doan 07/01/2021

### Part 3: ML preparation - Experiment 1

# Load support package
library(fastDummies)
library(caret)
library(dplyr)
library(reticulate)
library(keras)
tensorflow::tf$random$set_seed(0) ## to obtain reproducible result

# Convert B and M values to 0 and 1 (B to 0, M to 1)
dummy <- dummy_cols(breast_cancer,remove_first_dummy = TRUE)
df <- dummy[,2:32]
head(df)

# Prepare subsets for training and testing(80:20)
set.seed(33) ## to obtain reproducible subsets
intrain <- createDataPartition(df$diagnosis_M, p=0.8, list=FALSE)
df.training.temp <- df[intrain,]
df.test <- df[-intrain,]

# Prepare subsets for training and validation
# Validation set 20 % training set
inval <- createDataPartition(df.training.temp$diagnosis_M, p=0.8, list=FALSE)
df.training <- df.training.temp[inval,]
df.val <- df.training.temp[-inval,]

# Create training, test, validation inputs and outputs
x_train <- df.training %>% select(-diagnosis_M)
y_train <- df.training$diagnosis_M
x_test <- df.test %>% select(-diagnosis_M) 
y_test <- df.test$diagnosis_M
x_val <- df.val %>% select(-diagnosis_M) 
y_val <- df.val$diagnosis_M

# Check balance of 3 subsets
prop_M_train <- sum(y_train)/length(y_train)
prop_M_test <- sum(y_test)/length(y_test)
prop_M_val <- sum(y_val)/length(y_val)

prop_M_train
prop_M_test
prop_M_val

# Normalise data using min-max method
norm_train <- preProcess(x_train, method = c("range"))
x_train <- predict(norm_train, x_train)
norm_test <- preProcess(x_test, method = c("range"))
x_test <- predict(norm_test, x_test)
norm_val <- preProcess(x_val, method = c("range"))
x_val <- predict(norm_val, x_val)

# Check any missing value in the normalised subsets
anyNA(norm_train)
anyNA(norm_test)
anyNA(norm_val)

# Visualise data distribution for training set (figure 7)
mean_train <- round(apply(x_train,2,mean),3)
par(las=2, mar=c(9,3,2,2))
barplot(mean_train, main = 'Data distribution for training set', 
        col = 'blue',cex.names=0.8, ylim = c(0,0.4))

# Visualise data distribution for test set (figure 7)
mean_test <- round(apply(x_test,2,mean),3)
par(las=2, mar=c(9,3,2,2))
barplot(mean_test, main = 'Data distribution for testing set', 
        col = 'blue',cex.names=0.8, ylim = c(0,0.5))

# Visualise data distribution for validating set (figure 7)
mean_val <- round(apply(x_val,2,mean),3)
par(las=2,mar=c(10,2,3,1))
barplot(mean_val, main = 'Data distribution for validating set', 
        col = 'blue',cex.names=0.8, ylim = c(0,0.5))

# Data transfer to matrix before conduct the model
x_train <- data.matrix(x_train)
x_test <- data.matrix(x_test)
x_val <- data.matrix(x_val)

### Part 4: Model design

# Design ANN model
model <- keras_model_sequential()

# Configure the layers
#Input
 layer_dense(model,units = 30, activation = "relu", input_shape =  ncol(x_test),
             kernel_regularizer = regularizer_l2(l = 0.02),
             bias_regularizer = regularizer_l2(l = 0.01))%>% 
 layer_dropout(rate = 0.4) %>% 

# Hidden layer
 layer_dense(units = 21, activation = "relu",
             kernel_regularizer = regularizer_l2(l = 0.02),
             bias_regularizer = regularizer_l2(l = 0.01)) %>%
 layer_dropout(rate = 0.4) %>%
# Output layer
 layer_dense(units = 1, activation = "sigmoid",input_shape =  21,
             kernel_regularizer = regularizer_l2(l = 0.02),
             bias_regularizer = regularizer_l2(l = 0.01))

# Neural Network configure
history <- model %>% compile(
 loss = "binary_crossentropy",
 optimizer = "adam",
 metrics = c("accuracy")
)

### Part 5: Training model

# Training data
training_result <- model %>% fit(
 x_train, y_train, 
 epochs = 200, 
 batch_size = 32,
 validation_data = list(x_val,y_val)
)
summary(model)

### Part 4: Performance result
# Training accuracy
train_acc <- model %>% evaluate(x_train, y_train, verbose = 0 )
train_acc

# Validation accuracy
val_acc <- model %>% evaluate(x_val, y_val, verbose = 0 )
val_acc

# Predict model
predictions <- model %>% predict_classes(x_test)

# Testing accuracy
test_acc <- model %>% evaluate(x_test, y_test, verbose = 0 )
test_acc

# Confusion Matrix
t <- table(predictions,y_test)
confusionMatrix(t, positive = "1")
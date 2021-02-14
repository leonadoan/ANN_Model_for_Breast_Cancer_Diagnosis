# To illustrate data strategy to predict breast cancer
# ML model
# Le Doan 07/01/2021

### Part 3: ML preparation - Experiment 3

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

# PCA
pca_train <- x_train
pca_test <- x_test
pca_val <- x_val

pca <- prcomp(x_train, scale = T )

# variance
pr_var <- ( pca$sdev )^2 

# % of variance
prop_varex <- pr_var / sum( pr_var )

# Plot
plot( prop_varex, xlab = "Principal Component", 
                  ylab = "Proportion of Variance Explained", type = "b" )

# Feature extract using PCA for 3 subsets
train <- data.frame(pca$x)
t <- as.data.frame( predict( pca, newdata = pca_test ))
v <- as.data.frame( predict( pca, newdata = pca_val ))

new_train <- train[, 1:10]
new_test <-  t[, 1:10]
new_val <- v[,1:10]


# Data transfer to matrix before conduct the model
x_train <- data.matrix(new_train)
x_test <- data.matrix(new_test)
x_val <- data.matrix(new_val)

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

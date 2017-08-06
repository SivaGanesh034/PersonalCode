rm(list=ls())
# Importing the dataset
dataset = read.csv("D:\\DataScience_Drive\\Churn_Modelling.csv")
View(dataset)
dataset = dataset[4:14]

## 
str(dataset)
library(dummies)

dataset1 = dummy.data.frame(dataset)
dataset1 = dataset1[,-c(2,5)]
rm(dataset)
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset1$Exited, SplitRatio = 0.8)
training_set = subset(dataset1, split == TRUE)
test_set = subset(dataset1, split == FALSE)

# Feature Scaling
training_set[-12] = scale(training_set[-12])
test_set[-12] = scale(test_set[-12])

# Fitting ANN to the Training set
# install.packages('h2o')
training_set$Exited = as.factor(training_set$Exited)
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(50,50),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-12]))
y_pred = y_pred[,1]
y_pred = as.vector(y_pred)



# Making the Confusion Matrix
table(test_set$Exited, y_pred)

# h2o.shutdown()


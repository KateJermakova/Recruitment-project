import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import tensorflow as tf
%matplotlib inline

#Importing the dataset and taking only our company "C"
dataset = pd.read_csv('C:/Users/kater/Downloads/Dataset1.csv')
companyC = dataset[(dataset.company == 'C')] 

# in this part we create target variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(companyC['decision'])

from keras.utils import to_categorical
output_category = to_categorical(y, num_classes=None)
output_category

#converting categorical data into indicator variables (0/1)
dataset_1= pd.get_dummies(dataset,
                          columns=['ind-programming_exp', 'ind-exact_study', 'ind-international_exp', 'ind-entrepeneur_exp'],
                          prefix= ['ind-programming_exp', 'ind-exact_study', 'ind-international_exp', 'ind-entrepeneur_exp'])
dataset_1.head(2)
print(dataset_1)

# creating input features, specific columns from dataset
X = dataset_1.iloc[:,11:]

#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= sc.fit_transform(X)

# creating training and test data. test data set to 0.3 of whole dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, output_category, test_size=0.3)


def get_model(X_train, y_train):
    # define model
    model = Sequential()
    model.add(Dense(5, activation='relu', kernel_initializer='uniform', input_dim=8))
    model.add(Dropout(rate=0.3))
    model.add(Dense(5, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(2, activation='sigmoid', kernel_initializer='uniform'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X_train, y_train, epochs=300, verbose=0)
    return model
    

#Output Layer
    model.add(Dense(4, activation='softmax', kernel_initializer=kernel))
    
    model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

model = get_model(X_train, y_train)
 
#prediction on test data
yhat_probs = (model.predict(X_test) > 0.5).astype("int32")
yhat_probs = yhat_probs[:, 0]

#10-fold cross validation and calculation of accuracy
classifier = KerasClassifier(build_fn=get_model(X_train, y_train))
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1, scoring='accuracy')
print('Mean accuracy for 10 fold cross validation:  %f' % accuracies.mean())

# precision tp / (tp + fp)
precision = precision_score(y_test[:, 0], yhat_probs)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test[:, 0], yhat_probs)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test[:, 0], yhat_probs)
print('F1 score: %f' % f1)


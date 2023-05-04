import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dataset = pd.read_csv('/Users/Angela/Downloads/recruitmentdataset-2022-1.3.csv')
companyC = dataset[(dataset.company == 'C')] 
descriptors = companyC[['gender', 'age', 'nationality', 'sport', 'decision']]
descriptors.head()

#Converting to numerical data
descriptors['gender'].replace(['male', 'female', 'other'], [0,1,2], inplace = True)
descriptors['nationality'].replace(['Dutch', 'Belgian', 'German'], [0,1,2], inplace=True)
descriptors['sport'].replace(['Football', 'Swimming', 'Rugby', 'Tennis', 'Chess', 'Running', 'Cricket', 'Golf'], [0,1,2,3,4,5,6,7], inplace = True)
descriptors['decision'].replace([False, True], [0,1], inplace=True )

descriptors.head()

X = descriptors.values[:, 0:4]  #independent variable: gender, age, nationality, sport
Y = descriptors.values[:, 4]    #dependent variable: decision

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state = 50, max_depth=4, min_samples_leaf=10)
clf_entropy.fit(X_train, y_train)

#prediciting the values with the test set
y_pred_en = clf_entropy.predict(X_test)
y_pred_en

print(("Accuracy is"), accuracy_score(y_test,y_pred_en)*100)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_en, labels=[0,1])
from sklearn.metrics import precision_score
print(("Percision is"), precision_score(y_test, y_pred_en))
from sklearn.metrics import recall_score
print(("Recall is"), recall_score(y_test, y_pred_en))

feature_names = ['gender', 'age', 'nationality', 'sport'] #selecting the independent variables (X-axis)
feature_importance = pd.DataFrame(clf_entropy.feature_importances_, index = feature_names)
feature_importance

#The feature importance in a histogram
feature_importance.head().plot(kind='bar')

#visualize tree
from sklearn import tree
from matplotlib import pyplot as plt
#False = Rejected, True = Hired
fig = plt.figure(figsize=(25,20))
_= tree.plot_tree(clf_entropy, feature_names=feature_names, class_names={0:'False', 1:'True'}, filled=True, fontsize = 10)

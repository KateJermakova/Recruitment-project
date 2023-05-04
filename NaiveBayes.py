from sklearn.naive_bayes import GaussianNB 
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import keras

# Reading the original dataset 
my_dataset = pd.read_csv(r'/Users/am/Desktop/recruitmentdataset-2022-1.3.csv')

df = pd.DataFrame(my_dataset)

# Renaming some columns for the sake of convenience
df = df.rename(columns={"ind-entrepeneur_exp": "ind_entrepreneur_exp", "ind-programming_exp": "ind_programming_exp", "ind-university_grade": "ind_university_grade", "ind-debateclub": "ind_debateclub", "ind-degree": "ind_degree", "ind-international_exp": "ind_international_exp", "ind-exact_study": "ind_exact_study","ind-languages": "ind_languages"})

# Changing categorical values in columns into numerical 
df['ind_entrepreneur_exp'].replace([True, False],[1, 0], inplace=True)
df['ind_programming_exp'].replace([True, False],[1, 0], inplace=True)
df['decision'].replace([True, False],[1, 0], inplace=True)
df['ind_international_exp'].replace([True, False],[1, 0], inplace=True)
df['ind_debateclub'].replace([True, False],[1, 0], inplace=True)
df['ind_exact_study'].replace([True, False],[1, 0], inplace=True)
df['ind_degree'].replace(['bachelor', 'master', 'phd'],[1, 2, 3], inplace=True)

# Converting the dataset into a file to re-read it as pandas Dataframe 
df.to_csv('new_recruitmentdataset.csv') 

new_dataset = pd.read_csv(r'/Users/am/Downloads/new_recruitmentdataset.csv')

# Getting rid of the Id column
dataset = new_dataset.drop(["Id"], axis = 1)
# Selecting the C company subset assigned to our group 
dataset = dataset[2001:3000]
#print(dataset)

# Stating with the same indicators as M1 
xx = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages']].values
yy = dataset['decision'].values

# Splitting the data into the train and the test set 
X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.3, random_state = 42) 
test_data = (X_test,y_test)
train_data = (X_train,y_train)

# Creating a model
nb = GaussianNB()
model = nb.fit(X_train, y_train)
# Predicting the y-values on the test dataset 
y_pred = nb.predict(X_test)

print("Naive Bayes score: ",nb.score(X_test, y_test))

# First look at the metrics 
print ('Indicators: ind_university_grade, ind_programming_exp, ind_entrepreneur_exp, ind_languages') 
print ('Accuracy:', accuracy_score(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,
                              average='weighted'))
print ('Precision:', precision_score(y_test, y_pred,
                                    average='weighted'))
highest_results = three_highest_accuracies(batches, dict_accuracies) 

batch_4 = batch4
new_batches = [batch2, batch_4, batch1] 

# Batch 4 with ind_debateclub replacing each of the values 
batch4 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_debateclub', 'ind_languages', 'ind_exact_study']]
batch5 = dataset[['ind_university_grade', 'ind_debateclub', 'ind_entrepreneur_exp', 'ind_languages', 'ind_exact_study']]
batch6 = dataset[['ind_debateclub', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_exact_study']]
batch7 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_debateclub', 'ind_exact_study']]
batch8 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_debateclub']]

# Batch 4 with ind_international_exp replacing each of the values 
batch9 = dataset[['ind_international_exp', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_exact_study']]
batch10 = dataset[['ind_university_grade', 'ind_international_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_exact_study']]
batch11 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_international_exp', 'ind_languages', 'ind_exact_study']]
batch12 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_international_exp', 'ind_exact_study']]
batch13 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_international_exp']]

# Batch 2 with ind_debateclub replacing each of the values 
batch14 = dataset[['ind_debateclub', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_degree']]
batch15 = dataset[['ind_university_grade', 'ind_debateclub', 'ind_entrepreneur_exp', 'ind_languages', 'ind_degree']]
batch16 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_debateclub', 'ind_languages', 'ind_degree']]
batch17 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_debateclub', 'ind_degree']]

# Batch 2 with ind_international_exp replacing each of the values 
batch18 = dataset[['ind_international_exp', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_degree']]
batch19 = dataset[['ind_university_grade', 'ind_international_exp', 'ind_entrepreneur_exp', 'ind_languages', 'ind_degree']]
batch20 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_international_exp', 'ind_languages', 'ind_degree']]
batch21 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_international_exp', 'ind_degree']]

# Batch 1 with ind_debateclub replacing each of the values 
batch22 = dataset[['ind_debateclub', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages']] 
batch23 = dataset[['ind_university_grade', 'ind_debateclub', 'ind_entrepreneur_exp', 'ind_languages']] 
batch24 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_debateclub', 'ind_languages']] 
batch25 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_debateclub']] 

# Batch 1 with ind_international_exp replacing each of the values 
batch26 = dataset[['ind_international_exp', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_languages']] 
batch27 = dataset[['ind_university_grade', 'ind_international_exp', 'ind_entrepreneur_exp', 'ind_languages']] 
batch28 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_international_exp', 'ind_languages']] 
batch29 = dataset[['ind_university_grade', 'ind_programming_exp', 'ind_entrepreneur_exp', 'ind_international_exp']] 

list2 = [batch4, batch5, batch6, batch7, batch8, batch9, batch10, batch11, batch12, batch13, batch14, batch15, batch16, batch17, batch18, batch19, batch20, batch21, batch22, batch23, batch24, batch25, batch26, batch27, batch28, batch29]
new_batches = new_batches + list2

# Training the model on the highest scoring batches and the new batches
new_dict_accuracies = training(new_batches) 
# Retrieving the highest results 
highest_results = three_highest_accuracies(new_batches, new_dict_accuracies) 

# Testing on batches of smaller sizes (all combinations of two indicators) 
# Ind_university grade x others 
small_b1 = dataset[['ind_university_grade', 'ind_programming_exp']]
small_b2 = dataset[['ind_university_grade', 'ind_debateclub']]
small_b3 = dataset[['ind_university_grade', 'ind_international_exp']]
small_b4 = dataset[['ind_university_grade', 'ind_entrepreneur_exp']]
small_b5 = dataset[['ind_university_grade', 'ind_languages']]
small_b6 = dataset[['ind_university_grade', 'ind_exact_study']]
small_b7 = dataset[['ind_university_grade', 'ind_degree']]

# Ind_programming_exp x others 
small_b8 = dataset[['ind_debateclub', 'ind_programming_exp']]
small_b9 = dataset[['ind_international_exp', 'ind_programming_exp']]
small_b10 = dataset[['ind_entrepreneur_exp', 'ind_programming_exp']]
small_b11 = dataset[['ind_languages', 'ind_programming_exp']]
small_b12 = dataset[['ind_exact_study', 'ind_programming_exp']]
small_b13 = dataset[['ind_degree', 'ind_programming_exp']]

# Ind_debateclub x others 
small_b14 = dataset[['ind_debateclub', 'ind_entrepreneur_exp']]
small_b15 = dataset[['ind_debateclub', 'ind_international_exp']]
small_b16 = dataset[['ind_debateclub', 'ind_languages']]
small_b17 = dataset[['ind_debateclub', 'ind_exact_study']]
small_b18 = dataset[['ind_debateclub', 'ind_degree']]

# Ind_languages x others 
small_b19 = dataset[['ind_international_exp', 'ind_languages']]
small_b20 = dataset[['ind_entrepreneur_exp', 'ind_languages']]
small_b21 = dataset[['ind_exact_study', 'ind_languages']]
small_b22 = dataset[['ind_degree', 'ind_languages']]

# Ind_exact_study x others 
small_b23 = dataset[['ind_exact_study', 'ind_international_exp']]
small_b24 = dataset[['ind_exact_study', 'ind_entrepreneur_exp']]
small_b25 = dataset[['ind_exact_study', 'ind_degree']]

#Ind_degree x others 
small_b26 = dataset[['ind_entrepreneur_exp', 'ind_degree']]
small_b27 = dataset[['ind_international_exp', 'ind_degree']]

# Ind_international_exp x ind_entrepreneur_exp
small_b28 = dataset[['ind_entrepreneur_exp', 'ind_international_exp']]

small_batches = [small_b1, small_b2, small_b3, small_b4, small_b5, small_b6, small_b7, small_b8, small_b9, small_b10]
list_s1 = [small_b11, small_b12, small_b13, small_b14, small_b15, small_b16, small_b17, small_b18, small_b19, small_b20]
list_s2 = [small_b21, small_b22, small_b23, small_b24, small_b25, small_b26, small_b27, small_b28]
small_batches = small_batches + list_s1 + list_s2 

# Training the model on the small batches
small_dict_accuracies = training(small_batches) 

highest_small_results = three_highest_accuracies(small_batches, small_dict_accuracies) 

# cross validation
from sklearn.model_selection import cross_val_score
clf = model
scores = cross_val_score(clf, X_train, y_train, cv=10)
print(scores)

np.savetxt("csvsubmission.csv", scores, delimiter=",")


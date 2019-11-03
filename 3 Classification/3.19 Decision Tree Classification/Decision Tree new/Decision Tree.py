#importing libraries 
import pandas as pd
import matplotlib.pyplot as plt

#reading the data
data=pd.read_csv('survivor.csv')

#checking missing values in the data
data.isnull().sum()

#seperating independent and dependent variables
y = data['Survived']
X = data.drop(['Survived'], axis=1)

#importing train_test_split to create validation set
from sklearn.model_selection import train_test_split
#creating the train and test set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 101, stratify=y, test_size=0.25)
#Explaining Stratify whic will make distribution in training set & validation set same
y_train.value_counts(normalize=True)
y_valid.value_counts(normalize=True)

#importing decision tree classifier 
from sklearn.tree import DecisionTreeClassifier
# how to import decision tree regressor
from sklearn.tree import DecisionTreeRegressor
#creating the decision tree function
dt_model = DecisionTreeClassifier(random_state=10)
#fitting the model
dt_model.fit(X_train, y_train)
#checking the training score
dt_model.score(X_train, y_train)
#checking the validation score
dt_model.score(X_valid, y_valid)
#predictions on validation set
y_pred = dt_model.predict(X_valid)
#checking the prediction score
dt_model.score(X_valid, y_pred)

dt_model.predict_proba(X_valid)
y_pred = dt_model.predict_proba(X_valid)[:,1]
# changing the threshold from 0.5 to 0.4
y_new = []
for i in range(len(y_pred)):
    if y_pred[i]<=0.4:
        y_new.append(0)
    else:
        y_new.append(1)
        
from sklearn.metrics import accuracy_score
accuracy_score(y_valid, y_new) # accuracy drops

#Changing the max_depth
train_accuracy = []
validation_accuracy = []
for depth in range(3,12):
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=10)
    dt_model.fit(X_train, y_train)
    train_accuracy.append(dt_model.score(X_train, y_train))
    validation_accuracy.append(dt_model.score(X_valid, y_valid))


frame = pd.DataFrame({'max_depth':range(3,12), 'train_acc':train_accuracy, 'valid_acc':validation_accuracy})

plt.figure(figsize=(10,5))
plt.plot(frame['max_depth'], frame['train_acc'], marker='o')
plt.plot(frame['max_depth'], frame['valid_acc'], marker='o')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
plt.legend(('train accuracy','validtion accuracy'))
# graph shows that best depth is 8
dt_model = DecisionTreeClassifier(max_depth=8, max_leaf_nodes=25, random_state=10)
#fitting the model
dt_model.fit(X_train, y_train)
#Training score
dt_model.score(X_train, y_train)
#Validation score
dt_model.score(X_valid, y_valid)

from sklearn import tree
#install graphviz
decision_tree = tree.export_graphviz(dt_model,out_file='tree.dot'
                                     ,feature_names=X_train.columns
                                     ,max_depth=8,filled=True)
# converting dot to png
import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
# display the png file
image = plt.imread('tree.png')
plt.figure(figsize=(28,18))
plt.imshow(image)
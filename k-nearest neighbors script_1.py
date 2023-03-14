import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
 

# Importing the dataset
dataset = pd.read_csv('HueValue.csv')
dataset_1=dataset[dataset['Level'] == 1]
dataset_2=dataset[dataset['Level'] == 2]
dataset_3=dataset[dataset['Level'] == 3]
dataset_4=dataset[dataset['Level'] == 4]
dataset_5=dataset[dataset['Level'] == 5]
#print(len(dataset_1))
#print(len(dataset_5))
frames=[dataset_1,dataset_2,dataset_3,dataset_4,dataset_5]
dataset_1_5=pd.concat(frames)
print(len(dataset_1_5))
print(dataset_1_5.head())

X = dataset_1_5.iloc[0:, 1:].values
y = dataset_1_5['Level'].values



from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(xTrain)
X_test = sc.transform(xTest)
 
# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train,yTrain) 
 
 
# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(yTest,y_pred)*100)
 
plot_decision_regions(xTrain, yTrain, clf = classifier, legend = 2)
plt.show()
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(yTest, y_pred)
print(cm)
report = classification_report(yTest, y_pred)
print(report)





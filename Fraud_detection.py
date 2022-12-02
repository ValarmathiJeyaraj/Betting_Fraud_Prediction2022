import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

import sklearn as sk

dataset = pd.read_csv("C:\Deployment_model\Valarmati/Sports.csv")
dataset.head()

dataset.columns
dataset.isna().sum()

dataset["Credibility"].value_counts()

dataset.hist(figsize = (15,15))

plt.boxplot(dataset.DA);plt.title("BoxPlotDA");plt.show()
plt.boxplot(dataset.Wallet);plt.title("Wallet");plt.show()
plt.boxplot(dataset.AvgDep);plt.title("AvgDep");plt.show()
plt.boxplot(dataset.Age);plt.title("BoxPlotAge");plt.show()

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.3, variables = ["Wallet"])
dataset["Wallet"] = winsor.fit_transform(dataset[["Wallet"]])

winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.3, variables = ["DA"])
dataset["DA"] = winsor.fit_transform(dataset[["DA"]])

plt.boxplot(dataset.DA);plt.title("BoxPlotDA");plt.show()
plt.boxplot(dataset.Wallet);plt.title("Wallet");plt.show()

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
y.value_counts()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_test.shape, X_train.shape


"""from sklearn.ensemble import RandomForestClassifier
Rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
Rfc.fit(X_train, y_train.values.ravel())
"""
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

## Prediction
y_pred=classifier.predict(X_test)
### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
### Create a Pickle file using serialization 
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
classifier.predict([[2,3,4,1]])

"""
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = Rfc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
y_pred_tr = Rfc.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_tr)
confusion_matrix(y_test, y_pred)
hyper = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')
hyper.fit(X_train, y_train.values.ravel())

import pickle

pickle.dump(hyper,open('hyper.pkl' , 'wb'))
model1 = pickle.load(open('hyper.pkl','rb'))
result = model1.score(X_test, y_test)
result1 = model1.score(X_train, y_train)
print(result, result1)
Rfc.predict([[2,3,4,1]])

"""
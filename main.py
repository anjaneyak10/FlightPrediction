from sklearn import linear_model
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

#preprocessing
irisDataset = genfromtxt('2002.csv', delimiter=',', dtype=None, encoding=None)
x = irisDataset[1:, :13]
encoder = OrdinalEncoder()
X_8 = x[:,8].reshape(-1, 1)
X_8_encoded = encoder.fit_transform(X_8)
X_9 = x[:,9].reshape(-1, 1)
X_9_encoded = encoder.fit_transform(X_9)
x = np.hstack((x, X_8_encoded, X_9_encoded))
x = np.delete(x, [8, 9], axis=1)

y =irisDataset[1:, 13]
y = np.where(y>0,1,0)


# splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

#decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
y_pred= clf.predict(x_test)
Metrics=metrics.classification_report(y_pred,y_test)
dt_fpr, dt_tpr, threshold = metrics.roc_curve(y_test, y_pred)
print(dt_fpr,dt_tpr,threshold)
dt_roc_auc = metrics.auc(dt_fpr, dt_tpr)
print(Metrics)

# logistic regression
lrModel = linear_model.LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=2000)
lrModel.fit(x_train,y_train)
print(lrModel.score(x_test,y_test))
y_pred= lrModel.predict(x_test)
Metrics=metrics.classification_report(y_pred,y_test)
lr_fpr, lr_tpr, threshold1 = metrics.roc_curve(y_test, y_pred)
lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)
print(Metrics)


# plotting it in matlab
plt. figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_roc_auc:.2f})')
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_roc_auc:.2f})')
plt. plot ([0, 1], [0, 1], 'k--')
plt.ylabel( 'True Positive Rate')
plt.xlabel("False Positive Rate")
plt. title( 'ROC Curves')
plt. legend (loc='best')
plt.grid (True)
plt.show()

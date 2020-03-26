
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA as ICA
import sklearn.model_selection as k
r=pd.read_csv('bank_contacts.csv')
x=r.drop('credit_application',axis=1)
y=r['credit_application']
train_x,test_x,train_y,test_y=k.train_test_split(x,y,test_size=0.2,random_state=42)
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)
ica=ICA(n_components=4,random_state=42)
train_x=ica.fit_transform(train_x,train_y)
test_x=ica.transform(test_x)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(train_x,train_y)
pred = classifier.predict(test_x)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(confusion_matrix(test_y,pred))
print('Accuracy:',accuracy_score(test_y,pred))
plt.scatter(pred,test_x[:,0],marker='o')
plt.scatter(pred,test_x[:,1],marker='o')
plt.scatter(pred,test_x[:,2],marker='o')
plt.show()
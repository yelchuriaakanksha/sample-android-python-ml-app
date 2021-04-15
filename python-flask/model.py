import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import seaborn as sns
warnings.filterwarnings('ignore')

mydataset=pd.read_csv(".\glass.csv")
#print(mydataset)
#To check no of null values 
#l=mydataset.isnull().sum()
#print(l)






list_drop=['Id number']
mydataset.drop(list_drop,axis=1,inplace=True)
corr_matrix = mydataset.corr().abs()
upper = np.triu(np.ones_like(corr_matrix,dtype=bool))
k=corr_matrix.mask(upper)



features = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
label = ['Type of glass']

x = mydataset[features]

y = mydataset[label]

#names=mydataset.columns
#for i in names:
 #   mydataset[i]=mydataset[i].astype(np.int)
#x = mydataset.iloc[:,0:9].values 
#y = mydataset.iloc[:,4].values

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30, random_state=42)




stnd = StandardScaler()
mydataset = stnd.fit_transform(mydataset)

#svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
e_pred = clf.predict(x_test)
print('Accuracy of svm :',accuracy_score(y_test,e_pred)*100)
a6=accuracy_score(y_test,e_pred)*100

#linear regression
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
o_pred = clf.predict(x_test)
print('accuracy of linear regression:')
a1=r2_score(y_test,o_pred)*100
print(a1)



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 0)
trained_model=clf.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = clf.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm3 = confusion_matrix(y_test, y_pred)
print(cm3)




clf = RandomForestClassifier()
#Fitting model with trainig data
clf.fit(x_train,y_train)
# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))



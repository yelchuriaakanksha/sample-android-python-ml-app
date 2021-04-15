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


# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in k.columns if any(k[column] > 0.8)]
# Drop features
after_dropped=mydataset.drop(mydataset[to_drop], axis=1)
#print(to_drop)
#print(len(after_dropped.columns))

X = after_dropped.iloc[:, :-1].values # attributes to determine dependent variable / Class
y = after_dropped.iloc[:, -1].values# dependent variable / Class
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20, random_state=42)
#plt.figure(figsize = (8,8))
sns.heatmap(after_dropped.corr(), cmap = 'pink',linewidths=0.1,vmin=-1,vmax=1,annot = True)
#plt.title("Correlation Heatmap after Removing highly correlated values", fontsize=12)
#plt.show()


names=mydataset.columns
for i in names:
    mydataset[i]=mydataset[i].astype(np.int)
x = mydataset.iloc[:,:-1].values 
y = mydataset.iloc[:,4].values
#print('x is:',x)
#print('y is:',y)
#splitting into training and test sets
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30, random_state=0)
#print('x_train set:',x_train)



#standard scalar
#print("standard scalar")
stnd = StandardScaler()
mydataset = stnd.fit_transform(mydataset)
#print('standardization:',mydataset)



#svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
e_pred = clf.predict(x_test)
#print('Accuracy of svm :',accuracy_score(y_test,e_pred)*100)
a6=accuracy_score(y_test,e_pred)*100




clf = SVC()
#Fitting model with trainig data
clf.fit(x_train,y_train)
# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2,3,4,5,6,7,8,9]]))

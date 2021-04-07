import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
warnings.filterwarnings('ignore')

mydataset=pd.read_csv("E:\\glass.csv")
print(mydataset)
#To check no of null values 
l=mydataset.isnull().sum()
print(l)


outliers=['Na']
print(sns.boxplot(x=mydataset['Type of glass']))

def remove_outlier(mydataset, col_name):
    q1 = mydataset[col_name].quantile(0.25)
    q3 = mydataset[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    mydataset_out = mydataset.loc[(mydataset[col_name] > fence_low) | (mydataset[col_name] < fence_high)]
    return mydataset_out
mydataset=remove_outlier(mydataset,'Na')
mydataset.info()


list_drop=['Id number']
mydataset.drop(list_drop,axis=1,inplace=True)
#CORRELATION
plt.figure(figsize=(8,8))
corr=mydataset.corr()
corr.index=mydataset.columns
sns.heatmap(corr,cmap='pink',linewidths=0.1,vmin=-1,vmax=1,annot=True)
plt.title("Correlation Heatmap before Removing highly correlated values", fontsize=12)
plt.show()

corr_matrix = mydataset.corr().abs()
upper = np.triu(np.ones_like(corr_matrix,dtype=bool))
k=corr_matrix.mask(upper)
# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in k.columns if any(k[column] > 0.8)]
# Drop features
after_dropped=mydataset.drop(mydataset[to_drop], axis=1)
print(to_drop)
print(len(after_dropped.columns))

X = after_dropped.iloc[:, :-1].values # attributes to determine dependent variable / Class
y = after_dropped.iloc[:, -1].values# dependent variable / Class
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20, random_state=42)
plt.figure(figsize = (8,8))
sns.heatmap(after_dropped.corr(), cmap = 'pink',linewidths=0.1,vmin=-1,vmax=1,annot = True)
plt.title("Correlation Heatmap after Removing highly correlated values", fontsize=12)
plt.show()


names=mydataset.columns
for i in names:
    mydataset[i]=mydataset[i].astype(np.int)
x = mydataset.iloc[:,:-1].values 
y = mydataset.iloc[:,4].values
print('x is:',x)
print('y is:',y)
#splitting into training and test sets
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30, random_state=0)
print('x_train set:',x_train)


#Applying PCA function on training and testing set of X component 
from sklearn.decomposition import PCA 
pca = PCA() 
x_train = pca.fit_transform(x_train) 
x_test = pca.transform(x_test) 
explained_variance = pca.explained_variance_ratio_

mydataset['RI'] = mydataset.RI.astype(int)
mydataset['Na'] = mydataset.Na.astype(int)
mydataset['Mg'] = mydataset.Mg.astype(int)
mydataset['Al'] = mydataset.Al.astype(int)
mydataset['Si'] = mydataset.Si.astype(int)
mydataset['K'] = mydataset.K.astype(int)
mydataset['Ca'] = mydataset.Ca.astype(int)
mydataset['Ba'] = mydataset.Ba.astype(int)
mydataset['Fe'] = mydataset.Fe.astype(int)
mydataset.dtypes

#standard scalar
print("standard scalar")
stnd = StandardScaler()
mydataset = stnd.fit_transform(mydataset)
print('standardization:',mydataset)

#linear regression
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
o_pred = clf.predict(x_test)
print('accuracy of linear regression:')
a1=r2_score(y_test,o_pred)*100
print(a1)

#decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
t_pred = clf.predict(x_test)
print('Accuracy of decision tree:', (metrics.accuracy_score(y_test,t_pred)*100))
a2=metrics.accuracy_score(y_test,t_pred)*100



#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 50)
clf.fit(x_train, y_train)
th_pred = clf.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
result = confusion_matrix(y_test, th_pred)
print("Confusion Matrix:")
print(result)
#result1 = classification_report(y_test, th_pred)
result2 = accuracy_score(y_test,th_pred)
print("Accuracy of random forest classifier:",result2*100)
a3=result2*100


#naive bayes
from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB()
gauss.fit(x_train, y_train)
f_pred = gauss.predict(x_test)
#Accuracy
print("Accuracy of naive bayes:",(metrics.accuracy_score(y_test, f_pred)*100))
a4=metrics.accuracy_score(y_test, f_pred)*100

#knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
#train the model using the trainingsets
knn.fit(x_train,y_train)
#predict the response for test dataset
ypred=knn.predict(x_test)
#accuracy
print('Accuracy of KNN classifier:',(metrics.accuracy_score(y_test,ypred)*100))
a5=metrics.accuracy_score(y_test,ypred)*100

#svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
e_pred = clf.predict(x_test)
print('Accuracy of svm :',accuracy_score(y_test,e_pred)*100)
a6=accuracy_score(y_test,e_pred)*100

# x-coordinates of left sides of bars 
left = [0,20,40,60,80,100] 
# heights of bars 
height = [a1,a2,a3,a4,a5,a6] 
# labels for bars 
tick_label = ['LinearRegression', 'DecisionTree', 'RandomForestClassifier','NaiveBayes', 'KNeighbors', 'SupportVectorMachine',] 
plt.figure(figsize=(15, 8))
# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label, 
		width = 15, color = ['lightpink','lightblue','yellow','orange','lightgreen','purple']) 
# naming the x-axis 
plt.xlabel('algorithm') 
# naming the y-axis 
plt.ylabel('accuracy') 
# plot title 
plt.title('accuracy score for different models with correlation and PCA') 
# function to show the plot 
plt.show() 


clf = SVC()
#Fitting model with trainig data
clf.fit(x_train,y_train)
# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2,3,4,5,6,7,8,9]]))


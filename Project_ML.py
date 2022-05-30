
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot") 

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
#read CSV File
df =pd.read_csv("./dataset/smoking.csv")

# look the data set
print(df.head(10))
print('====================================\n')

print(df.shape)
print('====================================\n')

print(df.info())
print('====================================\n')

print(df.describe().round(2))
print('====================================\n')

df['gender'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)
plt.show()
df['smoking'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)
plt.show()

summary=df.groupby(["gender","smoking"])["age","weight(kg)","height(cm)"].mean().round(0)
print(summary)
print('====================================\n')




X_train = pd.read_csv('./dataset/competition_format/x_train.csv')
X_test = pd.read_csv('./dataset/competition_format/x_test.csv')
y_train = pd.read_csv('./dataset/competition_format/y_train.csv')
y_test = pd.read_csv('./dataset/competition_format/y_test.csv')
method_names = []
method_scores = []

l = [X_train,X_test,y_train,y_test]
for i in l:
    i.info()
    print('====================================\n\n')

#pre-processing

li = [X_train,X_test]
ONE = OneHotEncoder(handle_unknown='ignore')
def oneHot(df,a):
    cat_encoder = OneHotEncoder()
    ec_cat=cat_encoder.fit_transform(df[[a]])
    return ec_cat.toarray()
X_train['gender'] = oneHot(X_train,'gender')
X_test['gender'] = oneHot(X_test,'gender')

ordinal_encoder = OrdinalEncoder(categories = [['N','Y']])
X_train["oral"] = ordinal_encoder.fit_transform(X_train[["oral"]])
X_test["oral"] = ordinal_encoder.fit_transform(X_test[["oral"]])
X_train["tartar"] = ordinal_encoder.fit_transform(X_train[["tartar"]])
X_test["tartar"] = ordinal_encoder.fit_transform(X_test[["tartar"]])

print(X_train.info())

y_train = y_train['smoking']
y_test = y_test['smoking']


#building model

DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)
y_pred = DTC.predict(X_test)
print("DecisionTreeClassifier")
print("Score the X-train with Y-train is : ", DTC.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", DTC.score(X_test,y_test))
print("Accuracy Score :",accuracy_score(y_test,y_pred)*100)
method_names.append("DecisionTree")
method_scores.append(accuracy_score(y_test,y_pred)*100)
print('====================================\n')
conf_mat = confusion_matrix(y_test,y_pred)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
sort = DTC.feature_importances_.argsort()
plt.barh(df.columns[sort], DTC.feature_importances_[sort])
plt.xlabel("Feature Importance")
plt.show()

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("K Neighbors Classifier")
print("Score the X-train with Y-train is : ", knn.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", knn.score(X_test,y_test))
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100)
method_names.append("KNeighbors")
method_scores.append(accuracy_score(y_test,y_pred)*100)
print('====================================\n')
conf_mat = confusion_matrix(y_test,y_pred)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()

nb = GaussianNB()
nb.fit(X_train, y_train)
ypred = nb.predict(X_test)
print("Naive Bayes")
print("Score the X-train with Y-train is : ", nb.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", nb.score(X_test,y_test))
print("Accuracy Score :",accuracy_score(y_test,ypred)*100)
method_names.append("Naive Bayes")
method_scores.append(accuracy_score(y_test,y_pred)*100)
print('====================================\n')
conf_mat = confusion_matrix(y_test,y_pred)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


plt.figure(figsize=(10,5))
plt.ylim([1,100])
plt.bar(method_names,method_scores,width=0.2)
plt.xlabel('Method Name')
plt.ylabel('Method Score')
plt.show()
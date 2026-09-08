import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

data=pd.read_csv("census_income.csv")
pd.set_option('display.max_columns', None)
#print(data.head())
#print(data.dtypes)
#print(data.isnull().sum())
#print(data.describe())
#print(data.columns)
features = ['Workclass','Education','Marital_status','Occupation','Relationship','Race','Sex','Native_country']
#print(data.nunique().to_frame("No of unique values"))
#print(data['Workclass'].unique())
#print(data['Workclass'].value_counts())
data['Workclass'] = data['Workclass'].str.strip()
data['Workclass']=data['Workclass'].replace('?',np.nan)
#print(data['Workclass'].value_counts(dropna=False))
#print(data.isnull().sum())
data.dropna(inplace=True)
#print(data.shape)
#print(data.isnull().sum())
#print(data['Education'].unique())
#print(data['Education'].value_counts())
#print(data['Marital_status'].value_counts())
#print(data['Occupation'].value_counts())
data['Occupation']=data['Occupation'].str.strip()
data['Occupation']=data['Occupation'].replace('?',np.nan)
data.dropna(inplace=True)
#print(data['Relationship'].value_counts())
#print(data['Race'].value_counts())
#print(data['Sex'].value_counts())
#print(data['Native_country'].value_counts())
#print(data['Native_country'].unique())
data['Native_country']=data['Native_country'].str.strip()
data['Native_country']=data['Native_country'].replace('?',np.nan)
data.dropna(inplace=True)
#print(data['Income'].unique())
grouped=data.groupby('Income')
#for name,group in grouped:
    #print(name)
    #print(group)
col= ['Workclass','Education','Marital_status','Occupation','Relationship','Race','Sex','Native_country','Income']
from sklearn.preprocessing import LabelEncoder
lab_enc=LabelEncoder()
data['Workclass']=lab_enc.fit_transform(data['Workclass'])
data['Education']=lab_enc.fit_transform(data['Education'])
data['Occupation']=lab_enc.fit_transform(data['Occupation'])
data['Marital_status']=lab_enc.fit_transform(data['Marital_status'])
data['Relationship']=lab_enc.fit_transform(data['Relationship'])
data['Race']=lab_enc.fit_transform(data['Race'])
data['Sex']=lab_enc.fit_transform(data['Sex'])
data['Native_country']=lab_enc.fit_transform(data['Native_country'])
data['Income']=lab_enc.fit_transform(data['Income'])
#print(data.head())
import warnings
warnings.filterwarnings('ignore')
print(data.columns)
count_cols=['Workclass','Education','Education_num','Marital_status','Occupation','Relationship','Race','Sex','Capital_gain','Capital_loss','Hours_per_week','Native_country','Income']
'''plt.figure(figsize=(40,40))
for i in range(len(count_cols)):
    plt.subplot(5,3,i+1)
    sns.countplot(x=data[count_cols[i]])
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()'''
'''plt.figure(figsize=(15,30))
plotnumber=1
for column in data:
    if plotnumber<=15:
        ax=plt.subplot(5,3,plotnumber)
        sns.histplot(data[column],ax=ax,kde=True,stat='density')
        plt.xlabel(column,fontsize=12)
    plotnumber+=1
plt.show()'''
'''print(abs(data.corr()['Income'].sort_values(ascending=True)))
data.drop('Income',axis=1).corrwith(data['Income']).plot(kind='bar',grid=True,figsize=(20,30),title='Correlaion with INCOME')
plt.show()

plt.figure(figsize=(30,30))
graph=1
for column in data:
    if graph<=15:
        ax=plt.subplot(5,3,graph)
        sns.boxplot(data=data[column],orient="v")
        plt.xlabel(column,fontsize=10)
    graph+=1
plt.show()

plt.figure(figsize=(25,22))
sns.heatmap(abs(data.corr()),annot=True)
plt.yticks(rotation=0)
plt.show()'''


'''sns.pairplot(data)
plt.show()'''

q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
print(IQR)

itv_high=(q3.Fnlwgt+1.5*IQR.Fnlwgt)
print(itv_high)
index=np.where(data['Fnlwgt']>itv_high)
data=data.drop(data.index[index])
print(data.shape)
data.reset_index()

itv_high=(q3.Age+1.5*IQR.Age)
print(itv_high)
index=np.where(data['Age']>itv_high)
data=data.drop(data.index[index])
print(data.shape)
data.reset_index()


features=['Age','Fnlwgt']
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer(method='box-cox')
data[features]=pt.fit_transform(data[features].values)
#print(data.head())


x=data.drop(columns=['Income'],axis=1)
y=data['Income']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x=pd.DataFrame(x_scaled,columns=x.columns)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['vif']=[variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
vif['Features']=x.columns

print(vif)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


'''for i in range(500,700):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=i)
    LR.fit(x_train,y_train)
    pred_train=LR.predict(x_train)
    pred_test=LR.predict(x_test)
    if round(accuracy_score(y_train,pred_train)*100,1)==round(accuracy_score(y_test,pred_test)*100,1):
       # print("At random state",i,"model performs well")
        #print("at random state",i)
        #print("Training accuracy score is",accuracy_score(y_train,pred_train)*100)
        #print("Testing accuracy score is",accuracy_score(y_test,pred_test)*100)
        continue
''' 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=562)

from imblearn.over_sampling import SMOTE
from collections import Counter

print("Training outcome \n",y_train.value_counts())
print(Counter(y_train))

ovsmp=SMOTE(sampling_strategy=0.80)
x_train_ns,y_train_ns=ovsmp.fit_resample(x_train,y_train)

print("the no of classes before fit",format(Counter(y_train)))
print("the no of classes afer fit",format(Counter(y_train_ns)))
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

lr.fit(x_train_ns,y_train_ns)
pred=lr.predict(x_test)
print(accuracy_score(y_test,pred)*100)
print(classification_report(y_test,pred))
print("Confusion matrix\n",confusion_matrix(y_test,pred))

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train_ns,y_train_ns)
pred_dt=dt.predict(x_test)
print(accuracy_score(y_test,pred_dt)*100)
print(classification_report(y_test,pred_dt))
print("Confusion matrix\n",confusion_matrix(y_test,pred_dt))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train_ns,y_train_ns)
pred_rf=rf.predict(x_test)
print(accuracy_score(y_test,pred_rf)*100)
print(classification_report(y_test,pred_rf))
print("Confusion matrix \n",confusion_matrix(y_test,pred_rf))

from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(x_train_ns,y_train_ns)
pred_gb=gb.predict(x_test)
print(accuracy_score(y_test,pred_gb)*100)
print(classification_report(y_test,pred_gb))
print("Confusion matrix \n",confusion_matrix(y_test,pred_gb))

from xgboost import XGBClassifier
xgbt=XGBClassifier()
xgbt.fit(x_train_ns,y_train_ns)
pred_xgbt=xgbt.predict(x_test)
print(accuracy_score(y_test,pred_xgbt)*100)
print(classification_report(y_test,pred_xgbt))
print("Confusion matrix \n",confusion_matrix(y_test,pred_xgbt))

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train_ns,y_train_ns)
pred_svc=svc.predict(x_test)
print(accuracy_score(y_test,pred_svc)*100)
print(classification_report(y_test,pred_svc))
print("Confusion matrix \n",confusion_matrix(y_test,pred_svc))
'''
from sklearn.model_selection import cross_val_score
scr=cross_val_score(lr,x,y,cv=5)
print("Cross validaion score from lr model is",scr.mean()*100)


from sklearn.model_selection import cross_val_score
scr=cross_val_score(dt,x,y,cv=5)
print("Cross validaion score from decisiion tree model is",scr.mean()*100)


from sklearn.model_selection import cross_val_score
scr=cross_val_score(rf,x,y,cv=5)
print("Cross validaion score from random forest model is",scr.mean()*100)


from sklearn.model_selection import cross_val_score
scr=cross_val_score(gb,x,y,cv=5)
print("Cross validaion score from gradient boost model is",scr.mean()*100)


from sklearn.model_selection import cross_val_score
scr=cross_val_score(xgbt,x,y,cv=5)
print("Cross validaion score from xgbt model is",scr.mean()*100)


from sklearn.model_selection import cross_val_score
scr=cross_val_score(svc,x,y,cv=5)
print("Cross validaion score from SVC model is",scr.mean()*100)'''

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

RocCurveDisplay.from_estimator(dt, x_test, y_test)
RocCurveDisplay.from_estimator(lr, x_test, y_test, ax=plt.gca())
RocCurveDisplay.from_estimator(rf, x_test, y_test, ax=plt.gca())
RocCurveDisplay.from_estimator(gb, x_test, y_test, ax=plt.gca())
RocCurveDisplay.from_estimator(xgbt, x_test, y_test, ax=plt.gca())
RocCurveDisplay.from_estimator(svc, x_test, y_test, ax=plt.gca())

plt.show()

from sklearn.model_selection import GridSearchCV

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=0)
params = {'learning_rate':np.arange(0.2,0.4,0.1),
          'n_estimators':range(10,12),
          'max_depth':range(6,12),
          }
GCV2 = GridSearchCV(xgbt,param_grid = params)
GCV2.fit(x_train,y_train)
print('best_pram', GCV2.best_params_)

rf=GCV2.best_estimator_

rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)

rf_confusion_mat = confusion_matrix(y_test,y_pred)

print('\nconfusion mat =>','\n',rf_confusion_mat )
print('\naccuracy_score =>','\n',accuracy_score(y_test,y_pred))

'''import joblib
joblib.dump(GCV2.best_estimator_,"CENSUS.pk1")
model=joblib.load("CENSUS.pk1")
y_preds=model.predict(x_test)
predicted=pd.DataFrame(y_preds,columns=['predicted'])
print(predicted)
'''
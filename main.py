import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv('framingham.csv')

columns=['cigsPerDay','totChol','BMI','heartRate','glucose']
for col in columns:
    data[col].fillna(data[col].median(),inplace=True)
data['education']=data['education'].fillna(data['education'].mode()[0])
data['BPMeds']=data['BPMeds'].fillna(data['BPMeds'].mode()[0])

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split

x=data.drop(columns=['TenYearCHD'],axis=1)
y=data['TenYearCHD']
cols=['cigsPerDay','totChol','BMI','heartRate','glucose','male','age','education','currentSmoker','prevalentStroke','prevalentHyp','diabetes','sysBP','diaBP','BPMeds']
preprocessing=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),cols),
    ],
    remainder='passthrough'
)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=Pipeline(steps=[
    ('preprocessor',preprocessing),
    ('regressor',LogisticRegression(max_iter=1000,class_weight='balanced'))
])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_proba=model.predict_proba(x_test)[:,1]
print('Accuracy Score : ',accuracy_score(y_test,y_pred))
print('confusion matrix : \n',confusion_matrix(y_test,y_pred))
print('classification report : \n',classification_report(y_test,y_pred))
print('roc-auc score : ',roc_auc_score(y_test,y_proba))

fpr,tpr,thresholds=roc_curve(y_test,y_proba)
plt.plot(fpr,tpr,label='LogReg(area=%0.2f)'%roc_auc_score(y_test,y_proba))
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate",fontweight='bold',fontsize=13)
plt.ylabel("True Positive Rate",fontweight='bold',fontsize=13)
plt.title("ROC-CURVE",fontweight='bold',fontsize=20)
plt.legend(loc='best')
plt.grid()
plt.show()
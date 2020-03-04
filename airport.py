import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def score_dataset(model,X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    me=mean_absolute_error(y_valid, preds)
    r_score = r2_score(y_valid,preds)
    preds = preds.tolist()
    for x in range(len(preds)):
        preds[x]= round(int(preds[x]))
        if preds[x]==0:
            preds[x]='Highly_Fatal_And_Damaging'
        elif preds[x]==3:
            preds[x]= 'Significant_Damage_And_Serious_Injuries'
        elif preds[x]==1:
            preds[x]='Minor_Damage_And_Injuries'
        elif preds[x]==2:
            preds[x]='Significant_Damage_And_Fatalities'
    output = pd.DataFrame({'Accident_ID': X_valid.Accident_ID,'Severity': preds})
    output.to_csv('submission.csv', index=False)
    return me,r_score


X = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

X.dropna(axis=0, subset=['Severity'], inplace=True)
y = X.Severity
X.drop(['Severity'], axis=1, inplace=True)

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

y = LabelEncoder().fit_transform(y)

train_X,val_X,train_Y,val_Y = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

# Define multiple models to check which works
# model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model = RandomForestRegressor(n_estimators=100, random_state=1)
# model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
# model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
# model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
#model  = linear_model.LinearRegression()
print(score_dataset(model,train_X,val_X,train_Y,val_Y))

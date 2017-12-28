from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import metrics
import pandas as pd

def bc_qual_prediction(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    clf = RandomForestRegressor(n_estimators=10, random_state=33)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    cv_score = cross_validation.cross_val_score(clf, X, y, cv=10)
    print ("score: ", score)
    print ("cv_score: ",cv_score)

    predicted_y = clf.predict(X_test)
    df_predicted_y = pd.DataFrame(predicted_y, columns=y.columns)
    
    qual2binary(df_predicted_y)
    qual2binary(y_test)
    bc_score = binary_score(df_predicted_y, y_test)
    print("bc_score: ", bc_score)

def qual2binary(qual):
    col = qual.columns
    qual[qual.columns] = qual[col]>=5.0
    
def binary_score(df1, df2):
    sum = 0
    for col in df1.columns:
        score = metrics.accuracy_score(df1[col], df2[col])
        sum += score
    bc_score = sum/len(df1.columns)
    return bc_score

def bc_binary_prediction(X, y):
    qual2binary(X)
    qual2binary(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    clf = RandomForestClassifier(n_estimators=10, random_state=33)
    clf.fit(X_train, y_train)
    
    predicted_y = clf.predict(X_test)
    df_predicted_y = pd.DataFrame(predicted_y==1.0, columns=y.columns)
    
    bc_score = binary_score(y_test, df_predicted_y)
    print("bc_score: ", bc_score)
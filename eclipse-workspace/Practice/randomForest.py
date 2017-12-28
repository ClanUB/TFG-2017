from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import UtilFunctions
from sklearn.model_selection import train_test_split, cross_val_score

#formatear el np.array output
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={"float_kind":float_formatter})

#===============================================================================
# Realizar el algoritmo de regresion de arbol arbitrario para la prediccion 
# Imprimir las precisiones del mismo (score y cross validation score)

# X(pd.DataFrame): las muestras
# y(pd.DataFrame): las etiquetas de clase
#===============================================================================
def rf_qual_prediction(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    clf = RandomForestRegressor(n_estimators=10, random_state=33)
    #===========================================================================
    # clf.fit(X_train, y_train)
    #===========================================================================
    
    #===========================================================================
    # score = clf.score(X_test, y_test) #Necesita el fit
    # print ("score: ", score)
    #===========================================================================
    for i in range(3,11):
        cv_score =  cross_val_score(clf, X, y, cv=i)
        print ("cv_score with cv = ", i, "\t", cv_score, cv_score.mean())

    predicted_y = clf.predict(X_test)
    df_predicted_y = pd.DataFrame(predicted_y, columns=y.columns)
    
    UtilFunctions.score2ranking(df_predicted_y)
    UtilFunctions.score2ranking(y_test)
    
    r_score = UtilFunctions.compute_accuracy(y_test, df_predicted_y)
    print("r_score: ", r_score)    


def rf_rank_prediction(X, y):
    copy_X = UtilFunctions.score2ranking(X)
    copy_y = UtilFunctions.score2ranking(y)
    X_train, X_test, y_train, y_test = train_test_split(copy_X, copy_y, test_size=0.20, random_state=0)
    clf = RandomForestRegressor(n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)
    
    predicted_y = clf.predict(X_test)
    df_predicted_y = pd.DataFrame(predicted_y, columns=y_test.columns, index=y_test.index)
    df_predicted_y = UtilFunctions.score2ranking(df_predicted_y, reverse=False)
    
    acc = UtilFunctions.compute_accuracy(y_test, df_predicted_y)
    accMargin = UtilFunctions.compute_accMargin(y_test, df_predicted_y)
    mae = UtilFunctions.compute_mae(y_test, df_predicted_y)
    print ("acc: ", acc, "\naccMargin: ", accMargin, "\nmae: ", mae)
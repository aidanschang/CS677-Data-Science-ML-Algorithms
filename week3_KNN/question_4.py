"""
Aidan Chang
Class: CS 677
Date: 07/24/2022
Homework Problem #4
Description of Problem: Conduct observation on the significance level of each 
features within k-NN classifier
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#train and predicts by using knn neighbors
def get_confusion_matrix(test_df,true_label, pred_label):
    TP, FN, FP, TN = confusion_matrix(test_df[true_label].tolist(), test_df[pred_label].tolist()).ravel()
    #print(f'TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}')
    return(f'Accuracy: {(TP+TN)/(FN+FP+TN+TP)}')
    # print(f'TPR: {TP/(TP+FN)}')
    # print(f'TNR: {TN/(TN+FP)}')

def get_knn_confusion_matrix(k_value,x_train, x_test, y_train, y_test):
    #train the knn model
    neigh = KNeighborsClassifier(n_neighbors=k_value)
    neigh.fit(x_train,y_train)
    
    #convert a series to a dataframe
    cm_df = pd.DataFrame(y_test)
    #Add a column name
    cm_df.columns= ['true_label']
    #Add a new column of prediction
    cm_df['pred_label']=neigh.predict(x_test).tolist()
    #get confusion matrix
    return(get_confusion_matrix(cm_df, 'true_label', 'pred_label'))

if __name__ == "__main__":
    #Preapres the dataframe
    banknote_auth = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
    banknote_auth.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

    """
    Problem 4.1
    """
    banknote_f1=banknote_auth.copy()
    banknote_f1.drop(['variance'],axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(banknote_f1.iloc[:, :3], banknote_f1.iloc[:, 3],test_size = 0.5, random_state= 3)
    f1_truncated=get_knn_confusion_matrix(3,x_train, x_test, y_train, y_test)
    print(f'f1 truncated: {f1_truncated}')

    banknote_f2=banknote_auth.copy()
    banknote_f2.drop(['skewness'],axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(banknote_f2.iloc[:, :3], banknote_f2.iloc[:, 3],test_size = 0.5, random_state= 3)
    f2_truncated=get_knn_confusion_matrix(3,x_train, x_test, y_train, y_test)
    print(f'f2 truncated: {f2_truncated}')

    banknote_f3=banknote_auth.copy()
    banknote_f3.drop(['curtosis'],axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(banknote_f3.iloc[:, :3], banknote_f3.iloc[:, 3],test_size = 0.5, random_state= 3)
    f3_truncated=get_knn_confusion_matrix(3,x_train, x_test, y_train, y_test)
    print(f'f3 truncated: {f3_truncated}')

    banknote_f4=banknote_auth.copy()
    banknote_f4.drop(['entropy'],axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(banknote_f4.iloc[:, :3], banknote_f4.iloc[:, 3],test_size = 0.5, random_state= 3)
    f4_truncated=get_knn_confusion_matrix(3,x_train, x_test, y_train, y_test)
    print(f'f4 truncated: {f4_truncated}')

    """
    Problem 4.2
    """
    # No, accuracy dropped slightly. Only f4 truncated accuracy maintained at the same level as before

    """
    Problem 4.3
    """
    # By removing f1 feature, the accuracy lost that most at 3%

    """
    Problem 4.4
    """
    #By removing f4 feature, the accuracy maintained at 99.85%

  
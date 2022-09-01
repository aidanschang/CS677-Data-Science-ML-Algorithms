"""
Aidan Chang
Class: CS 677
Date: 08/13/2022
Homework Problem #2
Description of Problem: pick on of the previous models and compares to SVM models
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix


def choose_class(ds, last_digit):
    if last_digit%3 == 0:
        ds_custom = ds.loc[ds['class'] < 3]
    elif last_digit%3 == 1:
        ds_custom = ds.loc[ds['class'] > 1]
    elif last_digit%3 == 2:
        ds_custom = ds.loc[ds['class'] != 2]
    return ds_custom


if __name__ == "__main__":
    ds = pd.read_csv('seeds_dataset.csv', sep= "\t", header= None)
    ds.columns = ["area", "perimeter", "compactness", "length", "width", "asymmetry_coefficient", "ker_groove_length", "class"]
    ds = ds.astype({'area': 'float', 'perimeter': 'float','compactness': 'float','length': 'float','width': 'float','asymmetry_coefficient': 'float','ker_groove_length': 'float','class': 'int'})

    bu_id=6
    ds_custom = choose_class(ds, 6)
    x_train, x_test, y_train, y_test = train_test_split(ds_custom.iloc[:, :7], ds_custom.iloc[:, 7], test_size= 0.5, random_state=3)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # print(len(x_train.index))
    # print(len(y_train.index))
    # print(len(x_test.index))
    # print(len(y_test.index))
    # print(x_test, y_test)

    """
    Problem 2.1
    """
    # Using logistic regression model to comare to SVM
    lgr = LogisticRegression(random_state=0).fit(x_train, y_train)
    y_pred=lgr.predict(x_test).tolist()
    print("Logistic Regression")
    print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(lgr, x_test, y_test)
    # TP, FN, FP, TN = confusion_matrix(y_test, y_pred).ravel()
    # print(f'TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}')
    plt.title('Logistic Regression')
    plt.show()
    print(classification_report(y_test,y_pred))
  
    # TP:33, TN:34, FP:2, FN:1
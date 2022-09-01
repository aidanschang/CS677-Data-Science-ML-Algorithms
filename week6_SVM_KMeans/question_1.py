"""
Aidan Chang
Class: CS 677
Date: 08/13/2022
Homework Problem #1
Description of Problem: using three types of SVM models to predict the test data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
import matplotlib.pyplot as plt
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

    # Set custom class labels according to BU ID
    bu_id=7
    ds_custom = choose_class(ds, 6)
    x_train, x_test, y_train, y_test = train_test_split(ds_custom.iloc[:, :7], ds_custom.iloc[:, 7], test_size= 0.5, random_state=3)
    
    # Scale the data
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    """
    Problem 1.1
    """
    linear_svc = svm.SVC(kernel='linear')
    # print(x_train, y_train)
    linear_svc.fit(x_train, y_train)
    y_pred = linear_svc.predict(x_test)
    print("linear SVM")
    # print(confusion_matrix(y_test,y_pred))
    plot_confusion_matrix(linear_svc, x_test, y_test) 
    plt.title('linear SVM')
    plt.show()
    print(classification_report(y_test,y_pred))
    """
    Problem 1.2
    """
    linear_svc = svm.SVC(kernel='rbf')
    linear_svc.fit(x_train, y_train)
    y_pred = linear_svc.predict(x_test)
    print("RBF/Gaussian SVM")
    plot_confusion_matrix(linear_svc, x_test, y_test)
    plt.title('RBF/Gaussian SVM')
    plt.show()
    print(classification_report(y_test,y_pred))


    """
    Problem 1.3
    """
    linear_svc = svm.SVC(kernel='poly')

    linear_svc.fit(x_train, y_train)
    y_pred = linear_svc.predict(x_test)
    print("Poly SVM")
    plot_confusion_matrix(linear_svc, x_test, y_test)
    plt.title('Poly SVM') 
    plt.show()
    print(classification_report(y_test,y_pred))

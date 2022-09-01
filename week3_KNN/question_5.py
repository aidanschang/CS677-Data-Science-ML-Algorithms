"""
Aidan Chang
Class: CS 677
Date: 07/24/2022
Homework Problem #5
Description of Problem: apply logistic regression model and compares the 
result with your simple classifer and k-NN
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#This functions returns the mean and std of each of the four columns from the input_df

#train and predicts by using knn neighbors
def get_logistic_regression_confusionM(x_train, x_test, y_train, y_test):
    #convert a series to a dataframe
    cm_df = pd.DataFrame(y_test)
    #Add a column name
    cm_df.columns= ['true_label']
    #train the data set
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    
    #Add a new column of prediction
    cm_df['pred_label']=clf.predict(x_test).tolist()
    #get confusion matrix
    get_confusion_matrix(cm_df, 'true_label', 'pred_label')
        
def get_confusion_matrix(test_df,true_label, pred_label):
    TP, FN, FP, TN = confusion_matrix(test_df[true_label].tolist(), test_df[pred_label].tolist()).ravel()
    print(f'TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}')
    print(f'Accuracy: {(TP+TN)/(FN+FP+TN+TP)}')
    print(f'TPR: {TP/(TP+FN)}')
    print(f'TNR: {TN/(TN+FP)}')

def simple_classifier(test_df):
    test_df['true_label']='na'
    for index, row in test_df.iterrows():
        if row['variance'] <= -2 or row['skewness'] <= -5 or row['curtosis'] >= 7.5:
            test_df.loc[index, 'true_label']= 1
        else:
            test_df.loc[index, 'true_label']= 0 
    return(test_df)

def logistic_regression(x_train, x_test, y_train):
    #train the data set
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    #returns a list of predicted outcomes
    result=clf.predict(x_test).tolist()
    return(result)
    
if __name__ == "__main__":
    #Preapres the dataframe
    banknote_auth = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
    banknote_auth.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    #Adding 'color' column
    banknote_auth["color"]=np.where(banknote_auth['class']== 0, 'green','red')
    #custom split the dataframe
    x_train, x_test, y_train, y_test = train_test_split(banknote_auth.iloc[:, :4], banknote_auth.iloc[:, 4],test_size = 0.5, random_state= 3)

    """
    Problem 5.1 & 5.2
    """
    y_values=get_logistic_regression_confusionM(x_train, x_test, y_train, y_test)

    """
    Problem 5.3
    """
    # Yes, the accuracy of logistic regression is 99.13% vs 77% of my simple classifier

    """
    Problem 5.4
    """
    #No, the k-NN classifier return a 99.85% of accuracy vs 99.13% from logistic regression

    """
    Problem 5.5
    """
    BU_ID= [9,1,1,6]

    #insert BU_ID as a new row
    x_test.loc[len(banknote_auth)] = BU_ID
    simple_test = x_test.copy()
    print(f'simple prediction result(legit is 0, fake is 1): {simple_classifier(simple_test)["true_label"][len(simple_test)-1]}')
    logistic_test= x_test.copy()
    print(f'k-NN prediction result(legit is 0, fake is 1): {logistic_regression(x_train, logistic_test, y_train)[-1]}')

    #Both predicted legit(0)

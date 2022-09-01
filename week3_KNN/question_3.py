"""
Aidan Chang
Class: CS 677
Date: 07/24/2022
Homework Problem #3
Description of Problem: apply k-NN classifier and compares the 
result with your simple classifer
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#train and predicts by using knn neighbors
def knn_neigh( k_values, x_train, x_test, y_train, y_test):
    accuracy_list=[]
    for k in k_values:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train,y_train)
        
        y_accu = neigh.score(x_test, y_test)
        accuracy_list.append(y_accu)
        print(f'Prediction Accuracy for k={k}: {y_accu}')
    return accuracy_list
    
def get_confusion_matrix(test_df,true_label, pred_label):
    TP, FN, FP, TN = confusion_matrix(test_df[true_label].tolist(), test_df[pred_label].tolist()).ravel()
    print(f'TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}')
    print(f'Accuracy: {(TP+TN)/(FN+FP+TN+TP)}')
    print(f'TPR: {TP/(TP+FN)}')
    print(f'TNR: {TN/(TN+FP)}')

def get_knn_confusion_matrix( k_value, x_train, x_test, y_train, y_test):
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
    get_confusion_matrix(cm_df, 'true_label', 'pred_label')

def simple_classifier(test_df):
    test_df['true_label']='na'
    
    for index, row in test_df.iterrows():
        if row['variance'] <= -2 or row['skewness'] <= -5 or row['curtosis'] >= 7.5:
            test_df.loc[index, 'true_label']= 1
        else:
            test_df.loc[index, 'true_label']= 0 
    return(test_df)

def knn_classifier(x_train, x_test, y_train):
    #train the knn model
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train,y_train)
    
    #Add a new column of prediction
    result=neigh.predict(x_test).tolist()
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
    Problem 3.1
    """
    k_values= [3,5,7,9,11]
    y_values=knn_neigh(k_values,x_train, x_test, y_train, y_test)

    """
    Problem 3.2
    """
    y = y_values
    x= k_values
    plt.plot(x,y)
    plt.title('knn Accuracy')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
    # Answer: The optimal k value for this test set is when k=3,7,9,11

    """
    Problem 3.3
    """
    y_values=get_knn_confusion_matrix(3, x_train, x_test, y_train, y_test)

    """
    Problem 3.4
    """
    #Yes, the k-NN classifier predicts nearly 100% vs my simple classifier at 77%

    """
    Problem 3.5
    """
    BU_ID= [9,1,1,6]
    x_test.loc[len(banknote_auth)] = BU_ID

    simple_test = x_test.copy()
    print(f'simple prediction result(legit is 0, fake is 1): {simple_classifier(simple_test)["true_label"][len(simple_test)-1]}')

    knn_test= x_test.copy()
    print(f'k-NN prediction result(legit is 0, fake is 1): {knn_classifier(x_train, knn_test, y_train)[-1]}')
    

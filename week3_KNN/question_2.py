"""
Aidan Chang
Class: CS 677
Date: 07/24/2022
Homework Problem #2
Description of Problem: split the data set into training and testing 
then construct and test your simple classifier based on the confusion matrix
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



#This functions returns the mean and std of each of the four columns from the input_df
def compute_banknote(input_df):
    f1_m=0
    f1_sd=0
    f2_m=0
    f2_sd=0
    f3_m=0
    f3_sd=0
    f4_m=0
    f4_sd=0
    results=[]
    f1_m=round(input_df['variance'].mean(),2)
    f1_sd=round(input_df['variance'].std(),2)
    f2_m=round(input_df['skewness'].mean(),2)
    f2_sd=round(input_df['skewness'].std(),2)
    f3_m=round(input_df['curtosis'].mean(),2)
    f3_sd=round(input_df['curtosis'].std(),2)
    f4_m=round(input_df['entropy'].mean(),2)
    f4_sd=round(input_df['entropy'].std(),2)
    results.extend([f1_m,f1_sd,f2_m,f2_sd,f3_m,f3_sd,f4_m,f4_sd])
    #print(results)
    return results

def simple_classifier(test_df):
    test_df['predicted_label']='na'

    for index, row in test_df.iterrows():
        if row['variance'] <= -2 or row['skewness'] <= -5 or row['curtosis'] >= 7.5:
            test_df.loc[index, 'predicted_label']= 1
        else:
            test_df.loc[index, 'predicted_label']= 0 
    print(test_df)

def get_confusion_matrix(test_df):
    TP, FN, FP, TN = confusion_matrix(test_df['Class'].tolist(), test_df['predicted_label'].tolist()).ravel()
    print(f'TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}')
    print(f'Accuracy: {(TP+TN)/(FN+FP+TN+TP)}')
    print(f'TPR: {TP/(TP+FN)}')
    print(f'TNR: {TN/(TN+FP)}')
   



if __name__ == "__main__":
    """
    Problem 2.1
    """
    banknote_auth = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
    banknote_auth.columns = ["variance", "skewness", "curtosis", "entropy", "Class"]

    banknote_auth["color"]=np.where(banknote_auth['Class']== 0, 'green','red')
    #custom split
    #x_train, x_test, y_train, y_test = train_test_split(banknote_auth.iloc[:, :3], banknote_auth.loc[:, 'class'],test_size = 0.5, random_state= 3)
    train, test = train_test_split(banknote_auth, test_size= 0.5, random_state=3) # random_state number returns the same data set everytime you run it
    
    training_good_df=train[train['Class']==0].copy()
    training_fake_df= train[train['Class']==1].copy()
    
    train_figure= sns.pairplot(training_good_df)
    plt.show()

    train_figure= sns.pairplot(training_fake_df)
    plt.show()
    """
    Problem 2.2
    """
   
    # if row['variance'] <= -2 or row['skewness'] <= -5 or row['curtosis'] >= 7.5:
    #     test_df.loc[index, 'predicted_label']= 1
    # else:
    #     test_df.loc[index, 'predicted_label']= 0 
    
    """
    Problem 2.3
    """
    print("Q2.3 Compute predicted class labels from simple classifier")
    simple_classifier(test)
    

    """
    Problem 2.4 & 2.5
    """
    get_confusion_matrix(test)

    """
    Problem 2.6
    """
    # Yes, my simple classifier give me an accuracy of 77%
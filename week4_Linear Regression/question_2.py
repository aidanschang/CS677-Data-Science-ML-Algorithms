"""
Aidan Chang
Class: CS 677
Date: 07/29/2022
Homework Problem #2
Description of Problem: using a number of different linear system models to analyze the correlations
"""
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def get_loss_func(x_train, x_test, patient_type):
    # Creating x and y feature based on facility group number
    x= 'serum_sodium'
    y = 'serum_creatinine'
    type=''
    
    if patient_type == 0:
        type="Survived Event"
    else:
        type="Death Event"
        
    """
    Problem 2.1
    """
    # Obtain the coeifficient for linear regression
    y_coeif = np.polyfit(x_train[x], x_train[y],1)
    print(f'a: {y_coeif[0]}, b: {y_coeif[1]}')
    # Obtain the model for linear regression
    p=np.poly1d(y_coeif)

    # Create a new column in testing dataset and create the value for L
    x_test['Pred_linear_y']= p(x_test[x]) 
    x_test['L_linear']= (x_test[y]-x_test['Pred_linear_y'])**2
    print(x_test)
    

    y_pred = x_test['Pred_linear_y']
    y_actual= x_test[y]
    x_value= x_test[x]
    plt.scatter(x_value,y_pred)
    plt.scatter(x_value,y_actual)
    plt.title(f'{type} linear')
    plt.xlabel(x)
    plt.ylabel(f'{y} level')
    plt.legend(['Pred_log_y', 'y_actual'])
    plt.show()
    
    """
    Problem 2.2
    """
    # Obtain the coeifficient for linear regression
    y_coeif = np.polyfit(x_train[x], x_train[y],2)
    print(f'a: {y_coeif[0]}, b: {y_coeif[1]}, c: {y_coeif[2]}')
    # Obtain the model for linear regression
    p=np.poly1d(y_coeif)

    # Create a new column in testing dataset and create the value for L
    x_test['Pred_quad_y']= p(x_test[x])
    x_test['L_quad']= (x_test[y]-x_test['Pred_quad_y'])**2
    print(x_test)
    

    y_pred = x_test['Pred_quad_y']
    y_actual= x_test[y]
    x_value= x_test[x]
    plt.scatter(x_value,y_pred)
    plt.scatter(x_value,y_actual)
    plt.title(f'{type} quadratic')
    plt.xlabel(x)
    plt.ylabel(f'{y} level')
    plt.legend(['Pred_log_y', 'y_actual'])
    plt.show()

    """
    Problem 2.3
    """
    # Obtain the coeifficient for linear regression
    y_coeif = np.polyfit(x_train[x], x_train[y],3)
    print(f'a: {y_coeif[0]}, b: {y_coeif[1]}, c: {y_coeif[2]}, d: {y_coeif[3]}')
    # Obtain the model for linear regression
    p=np.poly1d(y_coeif)

    # Create a new column in testing dataset and create the value for L
    x_test['Pred_cubic_y']= p(x_test[x])
    x_test['L_cubic']= (x_test[y]-x_test['Pred_cubic_y'])**2
    print(x_test)
    

    y_pred = x_test['Pred_cubic_y']
    y_actual= x_test[y]
    x_value= x_test[x]
    plt.scatter(x_value,y_pred)
    plt.scatter(x_value,y_actual)
    plt.title(f'{type} cubic')
    plt.xlabel(x)
    plt.ylabel(f'{y} level')
    plt.legend(['Pred_log_y', 'y_actual'])
    plt.show()

    """
    Problem 2.4
    """
    # Obtain the coeifficient for linear regression
    y_coeif = np.polyfit(np.log(x_train[x]), x_train[y],1)
    print(f'a: {y_coeif[0]}, b: {y_coeif[1]}')
    # Obtain the model for linear regression
    p=np.poly1d(y_coeif)

    # Create a new column in testing dataset and create the value for L
    x_test['Pred_log_y']= p(np.log(x_test[x]))
    x_test['L_log']= (x_test[y]-p(np.log(x_test[x])))**2
    print(x_test)
    
    
    y_pred = x_test['Pred_log_y']
    y_actual= x_test[y]
    x_value= x_test[x]
    plt.scatter(x_value,y_pred)
    plt.scatter(x_value,y_actual)
    plt.title(f'{type} GLM_1')
    plt.xlabel(x)
    plt.ylabel(f'{y} level')
    plt.legend(['Pred_log_y', 'y_actual'])
    plt.show()

    """
    Problem 2.5
    """
    # Obtain the coeifficient for linear regression
    y_coeif = np.polyfit(np.log(x_train[x]), np.log(x_train[y]),1)
    print(f'a: {y_coeif[0]}, b: {y_coeif[1]}')

    # Obtain the model for linear regression
    p=np.poly1d(y_coeif)

    # Create a new column in testing dataset and create the value for L

    x_test['Pred_y']= math.e**(p(np.log(x_test[x])))
    x_test['L_logy']= (x_test[y]-x_test['Pred_y'])**2
    print(x_test)

    #Plot
    y_pred = x_test['Pred_y']
    y_actual= x_test[y]
    x_value= x_test[x]
    plt.scatter(x_value,y_pred)
    plt.scatter(x_value,y_actual)
    plt.title(f'{type} GLM_2')
    plt.xlabel(x)
    plt.ylabel(f'{y} level')
    plt.legend(['Pred_y', 'y_actual'])
    plt.show()

    print(f'loss function for linear: {x_test["L_linear"].sum()}')
    print(f'loss function for quadratic: {x_test["L_quad"].sum()}')
    print(f'loss function for cubic: {x_test["L_cubic"].sum()}')
    print(f'loss function for GLM_1: {x_test["L_log"].sum()}')
    print(f'loss function for GLM_2: {x_test["L_logy"].sum()}')
    
if __name__ == "__main__":
    #Reading csv into df with columns added
    heartfailure_clinical_records = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=",", header=None)
    heartfailure_clinical_records.columns = ["age", "anaemia", "CPK_level", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time", "death_event"]

    #Remove the first row which contains the headers
    heartfailure_clinical_df= heartfailure_clinical_records.iloc[1:]
    #convert dtype from object to int or float
    heartfailure_clinical_df= heartfailure_clinical_df.astype({"CPK_level":"int","serum_creatinine":"float","serum_sodium":"int","platelets":"float"})
    
    
    #loc method indices a df by labels, in this case, when 'death'=='0' or '1'. The second [] passed in to create the columns that I wanted to create
    df_survived= heartfailure_clinical_df.loc[(heartfailure_clinical_df.death_event=='0'), ["serum_creatinine","serum_sodium"]]
    df_deceased= heartfailure_clinical_df.loc[(heartfailure_clinical_df.death_event=='1'), ["serum_creatinine","serum_sodium"]]
 
    # Split the sruvived patient records 50/50
    train_0, test_0= train_test_split(df_survived.iloc[:,:2], test_size = 0.5, random_state= 3)
    get_loss_func(train_0, test_0, 0)

    # Split the deceased patient records 50/50
    train_1, test_1= train_test_split(df_deceased.iloc[:,:2], test_size = 0.5, random_state= 3)
    get_loss_func(train_1, test_1, 1)
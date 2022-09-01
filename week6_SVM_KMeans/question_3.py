"""
Aidan Chang
Class: CS 677
Date: 08/13/2022
Homework Problem #3
Description of Problem: use k means clusters to solved the problems
"""
from cmath import sqrt
import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def set_cluster(ds,centroid):
    num_1 = 0
    num_2 = 0
    num_3 = 0

    #Add the numbers of class labels
    for i in ds.index:
        # print(ds['class'][i])
        if ds['class'][i]==1:
            num_1+=1
        elif ds['class'][i]==2:
            num_2+=1
        elif ds['class'][i]==3:
            num_3+=1
    # print(num_1, num_2, num_3)

    # Choose the highest number of class
    if num_1 > num_2 and num_1 > num_3:
        print( f'Centroid {centroid} assigned label is 1')
    elif num_2 > num_1 and num_2 > num_3:
        print( f'Centroid {centroid} assigned label is 2')
    elif num_3 > num_2 and num_3 > num_1:
        print( f'Centroid {centroid} assigned label is 3')

def find_cloeset_centroid(centroids, ds):
        # Set three centroids
        label_1_coord = centroids[2]
        label_2_coord = centroids[0]
        label_3_coord = centroids[1]
      
        for i in ds.index:
            # Calculate the distance from each data point to each centroids
            distance_to_1 = sqrt((abs(label_1_coord[0]-ds[5][i]))**2 + (abs(label_1_coord[1]-ds[1][i]))**2)
            distance_to_2 = sqrt((abs(label_2_coord[0]-ds[5][i]))**2 + (abs(label_2_coord[1]-ds[1][i]))**2)
            distance_to_3 = sqrt((abs(label_3_coord[0]-ds[5][i]))**2 + (abs(label_3_coord[1]-ds[1][i]))**2)

            # Choose the shortest distance and add to "euclid_pred"
            if distance_to_1.real < distance_to_2.real and distance_to_1.real < distance_to_3.real:
                ds.loc[i, 'euclid_pred']= int(1)
                #print(1, distance_to_1.real, distance_to_2.real, distance_to_3.real)
            elif distance_to_2.real < distance_to_1.real and distance_to_2.real < distance_to_3.real:
                ds.loc[i, 'euclid_pred']= int(2)
                #print(2, distance_to_1.real, distance_to_2.real, distance_to_3.real)
            elif distance_to_3.real < distance_to_2.real and distance_to_3.real < distance_to_1.real:
                ds.loc[i, 'euclid_pred']= int(3)
                #print(3, distance_to_1.real, distance_to_2.real, distance_to_3.real)
            
        return(ds)
  
def choose_class(ds, last_digit):
    if last_digit%3 == 0:
        ds_custom = ds.loc[ds['euclid_pred'] < 3]
    elif last_digit%3 == 1:
        ds_custom = ds.loc[ds['euclid_pred'] > 1]
    elif last_digit%3 == 2:
        ds_custom = ds.loc[ds['euclid_pred'] != 2]
    return ds_custom

def choose_classes(ds, last_digit):
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
    class_labels = ds.iloc[:,7]
    class_label=class_labels.copy()
    ds = ds.iloc[:, :7]

    # Data Scaling
    scaler = StandardScaler().fit(ds)
    ds_scaled = scaler.transform(ds)
    ds_scaled = pd.DataFrame(ds_scaled)
   
    # print(len(x_train.index))
    # print(len(y_train.index))
    # print(len(x_test.index))
    # print(len(y_test.index))
    # print(x_test, y_test)

    """
    Problem 3.1
    """
    inertia_list = []
    for k in range(1,9):
        kmeans = KMeans(n_clusters=k)
        y_kmeans = kmeans.fit_predict(ds_scaled)
        inertia = kmeans.inertia_
        inertia_list.append(inertia)

    # print(inertia_list)
    fig, ax = plt.subplots(1, figsize = (7,5))
    plt.plot(range(1,9), inertia_list, marker = 'o', color='green')

    plt.xlabel('number of clusters: k')
    plt.ylabel('inertia')
    plt.tight_layout()
    plt.show()
    
    #By using the knee method, when k=3 is the last k value that has significant drops
    """
    Problem 3.2
    """
    #Create random features
    random.seed(10)
    f_i= random.randint(1,7)
    loop = True
    while loop:
        f_j= random.randint(1,7)
        if f_i != f_j:
            loop = False

    #Create a new df based on fi and fj
    
    ds_random_features = ds_scaled.iloc[:,[f_i, f_j]]
  
    #Create KMeans model with 3 clusters
    kmeans = KMeans(n_clusters = 3)
    y_kmeans= kmeans.fit_predict(ds_random_features)

    #Create Clustors
    centroids = kmeans.cluster_centers_
    
    #Plotting the graph
    fig, ax = plt.subplots(1,figsize =(7,5))
    d_0= ds_random_features[y_kmeans==0]
    d_1= ds_random_features[y_kmeans==1]
    d_2= ds_random_features[y_kmeans==2]
    
    plt.scatter(d_0.iloc[:,0], d_0.iloc[:,1], s = 75, c ='red', label = 'class-1')
    plt.scatter(d_1.iloc[:,0], d_1.iloc[:,1], s = 75, c ='green', label = 'class-2')
    plt.scatter(d_2.iloc[:,0], d_2.iloc[:,1], s = 75, c ='blue', label = 'class_3')
    plt.scatter(centroids[:, 0], centroids[:,1] , s = 200 , c = 'black', marker = "x", label = 'Centroids')

    plt.legend()
    plt.xlabel(ds.columns[f_i])
    plt.ylabel(ds.columns[f_j])
    plt.tight_layout()
    plt.show()

    #Centroids are in the middle of each colored clusters, and colored classes has a visual distinctive separations from each other

    """
    Problem 3.3
    """
    class_labels=pd.DataFrame(class_labels)
    class_labels['predict_cluster']= y_kmeans
    ds_0 = class_labels.loc[class_labels['predict_cluster']==0]
    ds_1 = class_labels.loc[class_labels['predict_cluster']==1]
    ds_2 = class_labels.loc[class_labels['predict_cluster']==2]

    set_cluster(ds_0, centroids[0])
    set_cluster(ds_1, centroids[1])
    set_cluster(ds_2, centroids[2])

    # Answer
    # Centroid [-0.0342861   1.27072831] assigned label is 2
    # Centroid [ 0.88715143 -0.88097843] assigned label is 3
    # Centroid [-0.88027002 -0.27544567] assigned label is 1


    """
    Problem 3.4
    """
    # print(ds_random_features)
    euc_pred = find_cloeset_centroid(centroids, ds_random_features)
    # print(euc_pred['euclid_pred'])
    # print(class_labels)
    print(f'Accuracy for KMeans: {accuracy_score(class_label, euc_pred["euclid_pred"])}')
    # Answer
    # 0.87619

    """
    Problem 3.5
    """
    # Change object type from complex to int
    euc_pred = euc_pred.astype({'euclid_pred': 'int'})

    # Create an additional column with true_labels data
    euc_pred['true_label'] = class_label

    #Create an empty dataframe with column names
    final_df=pd.DataFrame(columns=[f_i, f_j, 'euclid_pred', 'true_label'])

    q1_class = [1,2]
    #Append data rows from euc_pred to final_df by matching true_label to my custom class labels, in this case, 1 and 2
    for i in euc_pred.index:
        if (euc_pred['euclid_pred'][i] in q1_class) and (euc_pred['true_label'][i] in q1_class):
                final_df = pd.concat([final_df, euc_pred.loc[[i]]], axis=0, ignore_index=True)
    
    # Compute for CM and accuracy
    true_label = final_df['true_label'].tolist()
    euclid_pred = final_df['euclid_pred'].tolist()

    print("KMeans Prediction")
    print(confusion_matrix(true_label, euclid_pred))
    print(classification_report(true_label, euclid_pred))

    #Answer: the accuracy is on the same level as SVM predictions, and is higher than the logistic regression. The CM is also identical to SVM models. I would say there is no significant differences between the outcome of SVM and KMeans methods
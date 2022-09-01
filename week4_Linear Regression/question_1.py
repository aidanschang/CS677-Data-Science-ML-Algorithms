"""
Aidan Chang
Class: CS 677
Date: 07/29/2022
Homework Problem #1
Description of Problem: Find correlations between surviving and deceased patient records
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #Reading csv into df with columns added
    heartfailure_clinical_records = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=",", header=None)
    heartfailure_clinical_records.columns = ["age", "anaemia", "CPK_level", "diabetes", "ejection_fraction", 
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time", "death"]
    
    #Remove the first row which contains the headers
    heartfailure_clinical_df= heartfailure_clinical_records.iloc[1:]

    """
    Problem 1.1
    """
    #convert dtype from object to int or float
    heartfailure_clinical_df= heartfailure_clinical_df.astype({"CPK_level":"int","serum_creatinine":"float","serum_sodium":"int","platelets":"float"})
    
    #loc method indices a df by labels, in this case, when 'death'=='0' or '1'. The second [] passed in to create the columns that I wanted to create
    df_0= heartfailure_clinical_df.loc[(heartfailure_clinical_df['death']=='0'), ["CPK_level","serum_creatinine","serum_sodium","platelets"]]
    df_1= heartfailure_clinical_df.loc[(heartfailure_clinical_df['death']=='1'), ["CPK_level","serum_creatinine","serum_sodium","platelets"]]
    
    """
    Problem 1.2
    """
    # print(df_survived.dtypes)
    sns.set_theme(style="dark")

    # Compute the correlation matrix
    corr_0 = df_0.corr()
    
    # Generate a mask for the upper triangle
    mask_0 = np.triu(np.ones_like(corr_0, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title('M0')
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_0, mask=mask_0, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    #Repeat same process for m_1
    corr_1 = df_1.corr()
    mask_1 = np.triu(np.ones_like(corr_1, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title('M1')
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_1, mask=mask_1, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    """
    Problem 1.3
    """
    #1.3.a CPK levels and serum sodium has the highest correlation in the surviving patients
    #1.3.b serum sodium with serum_creatinnine has the lowest correlation in the surviving patients
    #1.3.c serum sodium and platelets has the highest correlation for deceased patients
    #1.3.d serum creatinnine and serum sodium has the lowest correlation for deceased patients
    #1.3.e results are very different between m0 and m1 where m1 has many postive correlations and m0 has many negative correlations


    
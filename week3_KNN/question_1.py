"""
Aidan Chang
Class: CS 677
Date: 07/24/2022
Homework Problem #1
Description of Problem: compute the data set's mean and sd of each features
"""
import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    """
    Problem 1.1
    """
    banknote_auth = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
    banknote_auth.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

    banknote_auth["color"]=np.where(banknote_auth['class']== 0, 'green','red')
    print(banknote_auth)

    """
    Problem 1.2
    """
    zero_computation=[]
    one_computation =[]
    total_computation=[]
    answer_list=[]

    #make df copies for legit and fake banknotes
    zero_df=banknote_auth[banknote_auth['class']==0].copy()
    one_df=banknote_auth[banknote_auth['class']==1].copy()

    #Obtain computation results in a list
    zero_computation= compute_banknote(zero_df)
    one_computation= compute_banknote(one_df)
    total_computation=compute_banknote(banknote_auth)

    #Create a df that consist of 3 lists that obtained above
    answer_list.extend([zero_computation, one_computation, total_computation])
    answer_df=pd.DataFrame(answer_list)

    #Create column and row names
    answer_df.columns=["μ(f1)","σ(f1)","μ(f2)","σ(f2)","μ(f3)","σ(f3)","μ(f4)","σ(f4)"]
    answer_df.index= ['0', '1', 'all']
    print(["variance", "skewness", "curtosis", "entropy"])
    print(answer_df)

    """
    Problem 1.3
    """
    # 1. fake bills has a negative mean variance and skewness versus good bills has positive variance and skewness. 
    # 2. f4 has the closest mean and standard deviation when compares to other three features.

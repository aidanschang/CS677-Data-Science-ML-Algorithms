# Machine-Learning-Algorithms

## Overview
In my Data Science course, I learned the principle of some common supervised ML algorithms such as linear/logistic regression, KNN, SVM, Decision Tree, Random Forest, and KMeans unsupervised algorithm. For my term project, I took the initiative to learn Convolutional Neural Networks and trained a CNN on the MNIST(handwritten numbers) dataset from Keras.

## Weekly Assignments

### Week3- K Nearest Neighbor
Conducted banknote classification using the "banknote authentication Data Set" from the UCI Machine Learning Repository. The data set consists of 4 features ("variance", "skewness", "curtosis", and "entropy") and a true label column. 

In this assignment, I conducted a pair plot from Seaborn to visualize the relationship between each feature as below.

![image](https://user-images.githubusercontent.com/84875731/188002680-77fbb28d-c733-4378-8cd8-c6adacf8004d.png)

Based on the pair plot, we can identify the plot to be categorical rather than continuous. Given that, I trained the dataset with KNN and obtained nearly 100% accuracy.

<img width="461" alt="image" src="https://user-images.githubusercontent.com/84875731/188003909-429d77e3-487d-4210-b2b7-a3a793f6e436.png">

### Week4- Linear Regression
Conducted heart failure regression models by selecting 2 of the 13 features included in the heart failure clinical records data set from the UCI.

In this assignment, I created two heatmaps to visually see the correlations between each feature through colors, one for deceased patients (M1), and one for survived patients (M0). 

![heatmap](https://user-images.githubusercontent.com/84875731/188007872-18093c19-ed74-422c-a8eb-e891897e678f.png)
![m0 heatmap](https://user-images.githubusercontent.com/84875731/188008819-b8a2fdae-18b2-41f4-b1a3-e8ad38d8447e.png)

Then, select two features, x = 'serum_sodium' and y = 'serum_creatinine', and by using the same training data, I trained linear, quadratic, cubic, generalized linear models, and logged-generalized linear models for both surviving and deceased patients to calculate the loss functions of each model as below.

![linear](https://user-images.githubusercontent.com/84875731/188009313-e3f899b6-af44-4762-9c48-3970d8bf1c92.png)
![quadratic](https://user-images.githubusercontent.com/84875731/188009329-6130571d-c015-4ce3-a4de-f72c7090ed3b.png)
![cubic](https://user-images.githubusercontent.com/84875731/188009360-05d1bacc-483e-42c5-81bb-5cfc3b93634e.png)
![GLM1](https://user-images.githubusercontent.com/84875731/188009374-9aefc4d6-7db9-43a8-95cb-e1845aecaa18.png)
![GLM2](https://user-images.githubusercontent.com/84875731/188009383-cad0070a-780c-480f-b713-004f7367c794.png)

**Results**<br />

**The loss function for surviving models are:**<br />
loss function for linear: 44.85216850356206<br />
loss function for quadratic: 44.77588771121759<br />
loss function for cubic: 48.95618821877799<br />
loss function for GLM_1: 44.60868945048049<br />
loss function for GLM_2: 43.70152945464571<br />

**The loss function for deceased models are:**<br />
loss function for linear: 133.0705870179978<br />
loss function for quadratic: 135.98257303316836<br />
loss function for cubic: 135.2028692678449<br />
loss function for GLM_1: 132.87370607293298<br />
loss function for GLM_2: 141.1165997705449

### Week5- Naive Bayesian, Decision Tree, and Random Forest
Conducted algorithm comparisons between Naive Bayesian, Decision Tree, and Random Forest with the "fetal cardiotocography data set" from UCI.

In this assignment, I conducted the comparison based on 4 features, FHR baseline, ALTV (percentage of time with abnormal long-term variability), minimum of FHR histogram, and average of histogram.

First, I made a Random Forest Classifier Error Plot to determine the number of trees (N) and the max depth of each subtree (d). In this case, I am using N = 10 and d = 4 to minimize the error rate.

![RF error plot](https://user-images.githubusercontent.com/84875731/188034554-c25046d7-a40f-425d-82f3-17a97479b6eb.png)

Second, I used N = 10 and d = 4 to train the Random Forest then trained the Naive Beyesian and Decision Tree models with the same data set as Random Forest and obtained the results as below.

![Screen Shot 2022-09-01 at 5 36 31 PM](https://user-images.githubusercontent.com/84875731/188034983-11581e1a-477c-473c-99d7-2185ae55c26f.png)

### Week6- SVM and KMeans Clusters
Conducted various type of SVM models and KMeans Clusters with the "seeds data set" from UCI to determin the variety of wheat.

In the first part of the assignment, I was just using 2 out of 3 class labels and I picked Logistic Regression algorithm to compared the results with linear SVM, Gaussian SVM, and polynomial SVM. See below for results.

![Screen Shot 2022-09-01 at 6 31 24 PM](https://user-images.githubusercontent.com/84875731/188039907-2298c82a-ddc2-4ff5-9c66-d3a1ad90daa6.png)

For the second part of the assignment. I used all three class labels for the KMeans Clustering algorithm. First, I used the "knee" methond to determine best k, which is 3 as you can see below.

<img width="365" alt="image" src="https://user-images.githubusercontent.com/84875731/188040101-9f565e3c-42e6-4920-b92b-5ab223f47bd7.png">

Then, I re-ran my clustering with 3 clusters and constructed a scatter plot with their centroids. Finally, I obtained the accuracy of KMean Clustering to be 87.2%. 

Accuracy for KMeans: 0.8761904761904762
![clusterings](https://user-images.githubusercontent.com/84875731/188040360-9535affb-bb7d-4835-b60c-443489409a1e.png)


Interestingly, if I extracted the predictions for the two class labels for SVMs, I would obtained 96% accuracy.

<img width="391" alt="Screen Shot 2022-09-01 at 6 50 33 PM" src="https://user-images.githubusercontent.com/84875731/188041772-5343c06a-95f9-4d5b-b44f-641a22d8f1d6.png">

## Course Summary
I found fascinating with data science and how machine learning algorithms are implemented and used in different senerios. As results, I self studied the Neural Network in the concentrations on propagations, loss functions, and activation functions. Ultimately, I conducted my term project on the topic of Convolution Neural Network wiht the MNIST data set from Keras and will continue concentrate in the field of computer vision/object detection.



# K-Nearest Neighbor from scratch
 
## Description

In this notebook I implemented the K Nearest Neighbor algorithm from scratch using Python and the Numpy library. Then I used scikit-learn's breast cancer dataset with a train-test split of 75% and 25% to train and test my model. By testing the accuracy of the model with different K, I found k=3 gives the best train set accuracy and a test set accuracy of **97.20%**. This is identical to scikit-learn's built-in K Nearest Neighbor classifier's accuracy with a K of 3. 


## Performance

The training set accuracy for different K is as follows:

<img src='https://github.com/RandomY-2/K-Nearest-Neighbor-from-scratch/blob/main/images/different_k.png'>

From it we see that K=3 gives the best train set accuracy, and the test set accuracy with K=3 is:

<img src='https://github.com/RandomY-2/K-Nearest-Neighbor-from-scratch/blob/main/images/my_final_accu.png'>

which is same as scikit-learn's K Nearest Neighbor classifier:

<img src='https://github.com/RandomY-2/K-Nearest-Neighbor-from-scratch/blob/main/images/scikit_final_accu.png'>

This shows that our implementation is successful.

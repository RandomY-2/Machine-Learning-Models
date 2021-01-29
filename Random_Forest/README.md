# Random Forest Classifier from sratch
 
## Description

In this notebook, I will implement a random forest classifier from scratch in Python and test it using scikit-learn's built-in random forest classifier and the red wine dataset.

### Logic of Random Forest:

The general idea of random forest is not very sophisticated. We are essentially ensembling a group of decision trees, each trained using a random subset of the data and features, and making a voting-based prediction. The reason we don't want all decision trees to be trained using the same data is because we want random variation between the trees, which may help the model to overcome noises in the data.

So My procedure in this notebook will be:

1. Implement the Decision Tree classifier
2. Implement Bootstrap and Random Subspace(used to randomly select data and features for each decision tree)
3. Implement the Random Forest Algorithm
4. Test my implementation with scikit-learn's random forest classifier

A lot of the decision tree implementations are based on my [decision tree classifier](https://github.com/RandomY-2/ML_Models_From_Scratch/tree/main/Decision_Tree). The new methods in this notebook are **Bootstraping** and **Random Subspacing**.

### Bootstrap

Bootstrap is a key component of random forest algorithm. Since we want each decision tree to be different, we will not use the entire train set to train each decision tree. In contrast, we will randomly select a subset of data from the train set for each tree. This process of randomly selecting is bootstrapping.

### Random Subspace

So like I previously mentioned, we don't want each decision tree to be trained using same data. So for each tree there will be a parameter to control which features can be used to partition the data.

## Result

To test my implementation, I used the red wine dataset and scikit-learn's RandomForestClassifier. My model obtained a **71.25%** accuracy while scikit-learn's model obtained a **74.0625%** accuracy. The final accuracy shows that our random forest classifier is pretty close to scikit-learn's random forest classifier, which would show the effectiveness of our implementation.

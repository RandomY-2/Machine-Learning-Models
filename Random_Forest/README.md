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

A lot of the decision tree implementations are based on my [decision tree classifier](https://github.com/RandomY-2/ML_Models_From_Scratch/tree/main/Decision_Tree). The new features in 


## Bootstrap & Random Subspace

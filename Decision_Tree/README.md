# Decision_Tree_from_Scratch

## Description

In this notebook, I created a decision tree classifier from scratch using pandas and numpy libraries. Then, I will use the iris dataset to test my model against the scikit-learn built-in decision tree classifier. My decision tree model will partition the data based on lowest entropy, and I created a model with a **min sample leaf of 2 and a max depth of 3**.

## Helper Functions

The idea of a decision tree is that on every node, the tree will split the data in a way that the two splitted dataset will have lowest entropy(meaning the data will be the purest) and finally predict the class of leaves based on fequency. To achieve this, I implemented the following helper functions:

1. Check Purity: check if the data is now pure(having only one kind of label)
2. Predict Class: predict the class of current leaf(return which label is most frequent)
3. Get Potential Splits: get all the potential splits(every split inbetween every value for every column)
4. Get Best Split: get the best split that leads to the lowest entropy
5. Calculate Entropy: calculate the entropy

## Final Model

I combined all the helper functions to create the final decision tree model. Specifically, I will make a recursive decision tree algorithm that will split the data based on best split until at least one of the three following conditions is met:

1. All samples on the leaf are belong to the same class
2. the leaf has only 2 or less samples
3. the depth of the tree is 5

If one of these condition is met, the algorithm will predict the class based on the most frequent class label on the leaf. 

After training, the final model after training has the following structure:

<img src='https://github.com/RandomY-2/ML_Models_From_Scratch/blob/main/Decision_Tree/img/tree_structure.png'>

and the scikit-learn tree has the following structure:

<img src='https://github.com/RandomY-2/ML_Models_From_Scratch/blob/main/Decision_Tree/img/scikit_structure.png'>

Our model learns the exact same features as the scikit-learn model, which shows the success of our implementation. Our model obtained a final accuracy of **94.7368%** on classifying the flower, and this performance is identical to the scikit-learn model. 

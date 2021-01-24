# Logistic_Regression_From_Scratch

## Description

In this notebook, I created a Logistic Regression model with gradient descent from scratch using Python and the Numpy library. Then I used scikit-learn's breast cancer dataset with a **train-test split of 75% and 25%** to train and test my model. My model achieves a **96%** accuracy, whille scikit-learn's built-in Logistic Regression model has a **99%** accuracy. 

## Breast Cancer Dataset

Scikit-learn's breast cancer dataset is a classic and very easy binary classification dataset that contains data for malignant and benign tumors. The dataset has 569 samples with a class distribution as follows:

<p align="left">
  <img width="700" height="200" src="https://github.com/RandomY-2/Logistic_Regression_From_Scratch/blob/main/images/class_distribution.jpg">
</p>

## My Model

I built a simple Logistic Regression model with gradient descent to optimize parameters. My model achieves a 96% test accuracy, and the detailed performance of the model after fitting the training set is presented bellow:

<p align="center">
  <img width="600" height="200" src="https://github.com/RandomY-2/Logistic_Regression_From_Scratch/blob/main/images/model_confusion.jpg">
</p>

The scikit-learn built-in Logistic Regression model has a test accuracy of 99%, and the detailed performance is presented below:

<p align="center">
  <img width="600" height="200" src="https://github.com/RandomY-2/Logistic_Regression_From_Scratch/blob/main/images/scikit_confusion.jpg">
</p>

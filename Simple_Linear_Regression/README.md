# Linear_Regression_From_Scratch
 
## Description

In this notebook, I created a simple single-variable Linear Regression model with gradient descent using Python and the numpy library. I then used Kaggle's House Price dataset with a **train-test split of 75% and 25%** to train and evaluate the model. Specifically, I trained the model to predict house prices using the dataset's **GrLivArea** column. The model is compared to the scikit-learn's built-in Linear Regression model, and **all results are almost identical**.

## Dataset

The dataset is from [Kaggle's House Price dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and contains information about 1460 houses and their prices. For simplicity, I only used two of the columns, which are GrLivArea and Price columns. 

## My Model

I built a single-variable Linear Regression model with input being the above grade (ground) living area square feet and output be the price of that house. The performance of my model is described below:

- Parameters: the two graphs show that my model obtains parameters that are very close to scikit-learn's Linear Regression model. (The first parameter is different because I included a bias unit into my model)

   **My Model**:
   <p align="left">
     <img width="700" height="100" src="https://github.com/RandomY-2/Linear_Regression_From_Scratch/blob/main/images/my_parameters.jpg">
   </p>
   
   **Scikit-learn Model**:
   <p align="left">
     <img width="700" height="100" src="https://github.com/RandomY-2/Linear_Regression_From_Scratch/blob/main/images/scikit_parameters.jpg">
   </p>
   
- Regression Line:

   **My Model**:
   <p align="left">
     <img width="600" height="300" src="https://github.com/RandomY-2/Linear_Regression_From_Scratch/blob/main/images/my_regression_line.jpg">
   </p>
   
   **Scikit-learn Model**:
     <p align="left">
       <img width="500" height="300" src="https://github.com/RandomY-2/Linear_Regression_From_Scratch/blob/main/images/scikit_regression_line.jpg">
     </p>

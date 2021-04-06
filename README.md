# Churn Prediction Project Overview

Created a tool that predicts churn rates to help businesses make savings and perform better.

## Code and Resources Used
**Python Version:** `3.8.6`

**Packages:** `pandas, numpy, matplotlib, seaborn, sklearn, tensorflow, shap, imblearn`

**SMOTE Github:** https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/14_imbalanced/handling_imbalanced_data.ipynb

**SHAP Article:** https://towardsdatascience.com/churn-prediction-using-neural-networks-and-ml-models-c817aadb7057


## Data Cleaning
After loading the data, I needed to clean it up so that it was usable for our model. I made the following changes:
* removed redundant features,
* checked for any null values

## EDA
I looked at the percentage of churn in the dataset to find out that our dataset is imbalanced.
I ploted a histogram to find out that majority of customers in the age range 50-60 are leaving. My idea behind it is: other banks are offering money 
for opening an account. Therefore retired people, having more time and willing to get extra money, are more likely to go for the offer.

## Model Building
First, I performed nominal encoding - transformed the categorical variables into dummy variables. I then scaled numerical features for better performance using MinMaxScaler. After that I balanced the data using SMOTE. I also split the data into train and test subsets with a test size of 20%.

I built a ANN model using relu activation function for the first 2 layers, and softmax for the last one.

## Model performance
On the train and test subsets I managed to achieve accuracy of nearly 85% and around 82% respectively. F1-score (2*(precision*recall)/(precision+recall)) for the 0 class (no churn) equals 0.83, and for the 1 class (churn) 0.81. Using shap I realized that 'age' is the most important feature. My advice to businesses would be (e.g.) to offer to the 50-60  age group customers discounts to prevent them from leaving the business.

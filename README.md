# Predicting Loan Repayment: A Deep Learning Journey with Lending Club Data

This project focuses on analyzing Lending Club loan data, building a machine learning model to predict loan repayment, and evaluating its performance. The analysis involves data cleaning, exploratory data analysis (EDA), feature engineering, and the creation of a neural network model using TensorFlow.

### About Data:

LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

- Loan Data: [lending_club_loan_data.csv](https://drive.google.com/file/d/1ExYuPifN1v1mna2W2APz1V7CMDvf1nHu/view?usp=sharing)
- Data Description: [lending_club_info.csv](https://drive.google.com/file/d/1qLvLQOVfGbgyYleSiPlbhboCOZP2e6Sw/view?usp=sharing)

### Our Goal:
Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model that can predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. Keep in mind classification metrics when evaluating the performance of your model!

### Table of Contents:
1. [Data Description Setup](#data-description-setup)
2. [Importing Libraries & Loading Data](#importing-libraries-and-loading-data)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Cleaning](#data-cleaning)
5. [Feature Engineering & Handling Categorical Variables](#feature-engineering-and-handling-categorical-variables)
6. [Train Test Split & Data Normalization](#train-test-split-and-data-normalization)
7. [Creating & Training the Model](#creating-and-training-the-model)
8. [Evaluating Model Performance](#evaluating-model-performance)
9. [Predicting Output](#predicting-output)
10. [Conclusion](#conclusion)

## Data Description Setup

To understand the data, we start by loading a data dictionary that provides descriptions for each feature. This helps us gain insights into what each feature represents and its potential impact on the target variable.

## Importing Libraries and Loading Data

We import the necessary libraries, including -
- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `matplotlib` and `seaborn` for data visualization.
- `tensorflow` for building the deep learning model.

We load the Lending Club loan data into a pandas DataFrame. This data includes various features related to loan applications, such as loan amount, interest rate, employment length, and loan status.

## Exploratory Data Analysis

### Target Variable Analysis
We analyze the target variable, 'loan_status', to understand the distribution of fully paid and charged off loans.

### Feature Distribution
We visualize the distribution of key features, such as loan amount, using histograms. This helps us understand the range and frequency of loan amounts in the dataset.

### Correlation Analysis
We examine the correlations between continuous features using a heatmap. This helps us identify highly correlated features that may impact our model.

## Data Cleaning

### Handling Missing Values
We identify and handle missing values in the dataset. For instance, we explore features like 'emp_title' and 'emp_length' to decide whether to impute or drop them.

### Removing Irrelevant Features
We drop features that are not relevant or have too many unique values, such as 'title' and 'emp_title'.

## Feature Engineering and Handling Categorical Variables

- We engineer new features, such as 'loan_repaid', which converts the loan status into a numeric binary variable for modeling purposes.
  
- We convert categorical variables into dummy/indicator variables to make them suitable for machine learning algorithms. For example, we transform the 'home_ownership' feature into dummy variables representing different ownership statuses.

## Train Test Split and Data Normalization

- We split the data into training and testing sets to evaluate our model's performance on unseen data. This involves defining the feature matrix 'X' and target vector 'y', and then using a train/test split function.

- To ensure that all features contribute equally to the model, we normalize the feature data using MinMaxScaler. This scales the features to a range between 0 and 1.

## Creating and Training the Model

- We build a sequential neural network model using TensorFlow. The model consists of multiple layers, including input, hidden, and output layers. Dropout layers are added to prevent overfitting.

- We train the model on the training data and validate it on the testing data. The training process includes multiple epochs to optimize the model parameters.

## Evaluating Model Performance

- We evaluate the model's performance using metrics like validation loss, classification report, and confusion matrix. These metrics help us understand the model's accuracy and precision.
- Plots training and validation loss curves to assess model learning behavior.

## Predicting Output

We test the model by making predictions on new, unseen data. This involves selecting a random sample from the dataset and predicting whether the loan will be repaid.

## Conclusion

This project demonstrates the application of machine learning to predict loan repayment status. By following a structured approach to data analysis, cleaning, and modeling, we can build a reliable model to assist financial institutions in assessing loan applications.



